"""Reward_Executor — validation, sandboxed execution, and MJX/JAX wrapping.

MuJoCo-track analog of the Isaac project's Reward_Executor (Req 4, 5, 6). The
structural validation (``ast.parse`` + top-level ``compute_reward`` check) is
framework-agnostic and carried over verbatim. The deltas for this track:

  - the sandbox allowlist exposes ``jax`` / ``jax.numpy`` (as ``jnp``), NOT
    ``torch`` (Req 5.1 delta);
  - the wrapped reward conforms to the JAX goal-reward contract
    ``compute_reward(data, action, goal_xy, success_radius) -> (reward,
    components)`` consumed by ``src/envs/goal_env.py`` (Req 6 delta);
  - ``wrap`` additionally compiles the reward under ``jax.jit`` and surfaces a
    JAX trace/concretization failure (a non-jittable reward) as an
    ``ExecutionError`` so the Orchestrator can re-prompt with the trace error
    (Req 5.5);
  - the ``_MjxDataAccessProxy`` guards the ``data.*`` accessor surface so a
    hallucinated attribute fails with actionable feedback (Req 12.1).

The ``validate`` surface is import-light (no JAX) so it runs and unit-tests on
the controller host; ``jax``/``jnp`` are imported lazily inside the sandbox
builder, only added to the allowlist when importable.
"""

from __future__ import annotations

import ast
import builtins as _builtins
import math
import threading
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.exceptions import (
    DivergenceError,
    ExecutionError,
    TimeoutError,
    ValidationError,
)

__all__ = [
    "ValidationResult",
    "SandboxConfig",
    "RewardExecutor",
    "GoalRef",
    "WrappedReward",
    "REWARD_FUNCTION_NAME",
    "ALLOWED_BUILTINS",
    "DATA_ACCESSOR_HINT",
]

REWARD_FUNCTION_NAME = "compute_reward"

# Restricted builtins allowlist (Req 5.1) — identical to the Isaac track; the
# difference is the *modules* exposed (jax/jnp, not torch), handled below.
ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        "True", "False", "None",
        "abs", "min", "max", "sum", "round", "pow", "divmod",
        "float", "int", "bool", "complex",
        "len", "range", "enumerate", "zip", "map", "filter",
        "sorted", "reversed", "all", "any",
        "list", "dict", "tuple", "set", "frozenset", "str",
        "isinstance", "issubclass", "getattr", "hasattr",
        "Exception", "ValueError", "TypeError",
        "ZeroDivisionError", "ArithmeticError", "OverflowError",
    }
)


@dataclass
class ValidationResult:
    ok: bool
    error: str | None = None


@dataclass
class SandboxConfig:
    time_limit_s: float = 5.0
    extra_allowed_modules: tuple[str, ...] = field(default_factory=tuple)
    # Whether wrap() compiles the reward under jax.jit to catch non-traceable
    # code up front (Req 5.5). Disabled in non-JAX environments automatically.
    jit_check: bool = True


@runtime_checkable
class GoalRef(Protocol):
    """Duck-typed Goal handle: exposes ``position_xy`` + ``success_radius_m``."""

    position_xy: tuple[float, float]
    success_radius_m: float


# Documented MJX state-access surface advertised to the model in the prompts.
# The proxy names these when generated code references an unknown attribute, so
# a hallucinated accessor fails with actionable feedback (Req 12.1).
DATA_ACCESSOR_HINT = (
    "Valid MJX data accessors: data.{qpos, qvel, xpos, actuator_force, "
    "cfrc_ext}. qpos[0:3]=base xyz, qpos[3:7]=quat(w,x,y,z), qpos[7:]=joints; "
    "qvel[0:3]=base lin vel, qvel[3:6]=base ang vel, qvel[6:]=joint vel. "
    "The Goal is passed as goal_xy (2,) and success_radius (scalar)."
)


class _MjxDataAccessProxy:
    """Read-through guard around the ``mjx.Data`` passed to generated code.

    Forwards every existing attribute straight through (so valid accessors like
    ``data.qpos`` behave identically) but raises an ``AttributeError`` naming the
    valid surface when an attribute is genuinely absent — captured by the
    executor as an ``ExecutionError`` for actionable re-prompting (Req 12.1)."""

    __slots__ = ("_data",)

    def __init__(self, data: Any) -> None:
        object.__setattr__(self, "_data", data)

    def __getattr__(self, name: str) -> Any:
        data = object.__getattribute__(self, "_data")
        try:
            return getattr(data, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Generated reward referenced unknown mjx.Data attribute "
                f"'{name}'. {DATA_ACCESSOR_HINT}"
            ) from exc

    def __getitem__(self, key: Any) -> Any:
        return object.__getattribute__(self, "_data")[key]


class RewardExecutor:
    """Validates, sandboxes, and wraps generated JAX reward code."""

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config if config is not None else SandboxConfig()

    # ---------------------------- validate (Req 4) --------------------- #
    def validate(self, code: str) -> ValidationResult:
        """Structurally validate generated reward code (Req 4.1-4.4).

        Parse/inspect only — rejected code is NEVER executed."""
        try:
            module = ast.parse(code)
        except SyntaxError as exc:
            location = ""
            if exc.lineno is not None:
                location = f" at line {exc.lineno}"
                if exc.offset is not None:
                    location += f", column {exc.offset}"
            reason = exc.msg or "invalid syntax"
            return ValidationResult(
                ok=False,
                error=(
                    f"Generated reward code is not parseable Python: "
                    f"{reason}{location}."
                ),
            )
        except (ValueError, TypeError) as exc:
            return ValidationResult(
                ok=False,
                error=f"Generated reward code could not be parsed: {exc}.",
            )

        has_top_level_reward = any(
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == REWARD_FUNCTION_NAME
            for node in module.body
        )
        if not has_top_level_reward:
            return ValidationResult(
                ok=False,
                error=(
                    f"Generated reward code does not define a top-level "
                    f"'{REWARD_FUNCTION_NAME}' function."
                ),
            )
        return ValidationResult(ok=True, error=None)

    # ---------------------------- sandbox (Req 5) ---------------------- #
    def build_sandbox_globals(self) -> dict[str, object]:
        """Build the restricted-globals namespace (Req 5.1).

        Exposes the allowlisted builtins plus ``math`` always, and ``jax`` /
        ``jnp`` when importable (the MuJoCo-track delta — no ``torch``).

        A restricted ``__import__`` is provided that resolves ONLY the
        allowlisted modules (``jax``, ``jax.numpy``, ``math``) — so a generated
        reward that writes ``import jax.numpy as jnp`` (the natural idiom, shown
        in the prompt) works, while any other import raises ``ImportError``
        (captured as an ExecutionError). Everything else stays denied."""
        restricted_builtins = {
            name: getattr(_builtins, name)
            for name in ALLOWED_BUILTINS
            if hasattr(_builtins, name)
        }
        sandbox: dict[str, object] = {
            "math": math,
        }
        allowed_modules: dict[str, object] = {"math": math}
        try:  # pragma: no cover - exercised only where jax is installed
            import jax
            import jax.numpy as jnp

            sandbox["jax"] = jax
            sandbox["jnp"] = jnp
            allowed_modules["jax"] = jax
            allowed_modules["jax.numpy"] = jnp
        except ImportError:
            pass

        for module_name in self.config.extra_allowed_modules:
            try:
                module = __import__(module_name)
            except ImportError:
                continue
            sandbox[module_name.split(".")[0]] = module
            allowed_modules[module_name] = module

        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
            # Mirror CPython __import__ semantics for the allowlisted modules.
            # importlib.import_module(name) imports the submodule (side effect),
            # and we return what the bytecode expects: the top-level package
            # when fromlist is empty (``import jax.numpy as jnp`` binds via
            # attribute lookup on the returned ``jax``), or the named module
            # itself when a fromlist is present (``from jax import numpy``).
            root = name.split(".")[0]
            if root not in ("jax", "math") and name not in allowed_modules:
                raise ImportError(
                    f"import of '{name}' is not allowed in the reward sandbox; "
                    f"jnp and jax are already available without import"
                )
            import importlib

            module = importlib.import_module(name)
            if not fromlist:
                return importlib.import_module(root)
            return module

        restricted_builtins["__import__"] = _restricted_import
        # NOTE: providing a dict (not the module) disables implicit access to
        # the full builtin set inside exec'd code.
        sandbox["__builtins__"] = restricted_builtins
        return sandbox

    def compile_reward(self, code: str):
        """Validate, compile, and exec ``code`` into the restricted namespace."""
        result = self.validate(code)
        if not result.ok:
            raise ValidationError(
                result.error or "Generated reward code failed validation."
            )
        sandbox = self.build_sandbox_globals()
        try:
            compiled = compile(code, filename="<generated_reward>", mode="exec")
            exec(compiled, sandbox)  # noqa: S102 - restricted namespace by design
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"Generated reward code failed to define "
                f"'{REWARD_FUNCTION_NAME}': {type(exc).__name__}: {exc}"
            ) from exc
        func = sandbox.get(REWARD_FUNCTION_NAME)
        if not callable(func):
            raise ExecutionError(
                f"'{REWARD_FUNCTION_NAME}' is not callable after execution."
            )
        return func

    def execute_reward(self, func, *args, **kwargs):
        """Invoke a sandboxed reward under the wall-clock timeout (Req 5.2, 5.3)."""
        result_box: dict[str, object] = {}
        error_box: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result_box["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001
                error_box["error"] = exc

        worker = threading.Thread(target=_worker, name="reward-sandbox", daemon=True)
        worker.start()
        worker.join(self.config.time_limit_s)

        if worker.is_alive():
            raise TimeoutError(
                f"Reward execution exceeded the sandbox time limit of "
                f"{self.config.time_limit_s}s."
            )
        if "error" in error_box:
            exc = error_box["error"]
            raise ExecutionError(
                f"Reward function raised during execution: "
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return result_box.get("value")

    def check_finite(self, output) -> None:
        """Raise DivergenceError on non-finite output (Req 5.4)."""
        if not self._is_all_finite(output):
            raise DivergenceError(
                "Reward function returned a non-finite value (NaN or infinity)."
            )

    @staticmethod
    def _is_all_finite(output) -> bool:
        if isinstance(output, bool):
            return True
        if isinstance(output, (int, float)):
            return math.isfinite(output)
        if isinstance(output, complex):
            return math.isfinite(output.real) and math.isfinite(output.imag)
        # jax/numpy arrays: duck-type without importing jax.
        isfinite = getattr(output, "__array__", None)
        if isfinite is not None:
            try:
                import numpy as _np

                return bool(_np.isfinite(_np.asarray(output)).all())
            except Exception:  # noqa: BLE001
                pass
        if isinstance(output, (str, bytes)):
            return True
        try:
            iterator = iter(output)
        except TypeError:
            return True
        return all(RewardExecutor._is_all_finite(item) for item in iterator)

    # ---------------------------- wrap (Req 6, 5.5) -------------------- #
    def wrap(self, code: str, goal_ref: "GoalRef | None" = None) -> list["WrappedReward"]:
        """Validate, sandbox-compile, and wrap ``code`` as a JAX goal reward.

        Returns a single-element list (mirroring the design's list signature) of
        a :class:`WrappedReward` conforming to the goal_env reward contract
        ``compute_reward(data, action, goal_xy, success_radius)``.

        When ``jit_check`` is set and JAX is importable, the wrapped reward is
        compiled under ``jax.jit`` against a tiny synthetic input to catch
        non-traceable code (value-dependent control flow) up front and surface
        it as an ExecutionError for re-prompt (Req 5.5)."""
        func = self.compile_reward(code)
        wrapped = WrappedReward(func=func, goal_ref=goal_ref, executor=self)
        if self.config.jit_check:
            wrapped.maybe_jit_check()
        return [wrapped]


class WrappedReward:
    """JAX goal-reward callable backing a generated ``compute_reward``.

    Conforms to ``compute_reward(data, action, goal_xy, success_radius) ->
    (reward, components)`` (Req 6.1 delta). On each call it guards the mjx.Data
    surface, runs the generated function, records named components (Req 6.3,
    6.4), and (outside jit) checks finiteness (Req 5.4)."""

    def __init__(self, func, goal_ref: "GoalRef | None", executor: "RewardExecutor") -> None:
        self._func = func
        self._goal_ref = goal_ref
        self._executor = executor
        self._last_components: dict[str, Any] = {}

    @property
    def goal_ref(self) -> "GoalRef | None":
        return self._goal_ref

    @property
    def last_components(self) -> dict[str, Any]:
        return dict(self._last_components)

    def __call__(self, data, action, goal_xy, success_radius):
        """Evaluate the reward for one env (vmap adds the batch axis).

        Inside a jit/vmap trace this runs the generated function directly (no
        Python threads/finite-check, which would break tracing). Outside a trace
        it is exercised eagerly by the executor's sandbox + finite check."""
        guarded = _MjxDataAccessProxy(data)
        output = self._func(guarded, action, goal_xy, success_radius)
        reward, components = self._split_output(output)
        self._last_components = dict(components)
        return reward, components

    def run_eager(self, data, action, goal_xy, success_radius):
        """Eager, sandboxed single-call used by smoke tests / validation.

        Applies the wall-clock timeout, exception capture (Req 5.2, 5.3) and the
        non-finite check (Req 5.4). Not used inside the jitted training step."""
        guarded = _MjxDataAccessProxy(data)
        output = self._executor.execute_reward(
            self._func, guarded, action, goal_xy, success_radius
        )
        reward, components = self._split_output(output)
        self._executor.check_finite(reward)
        if components:
            self._executor.check_finite(list(components.values()))
        self._last_components = dict(components)
        return reward, components

    def maybe_jit_check(self) -> None:
        """Compile the reward under jax.jit on a synthetic input (Req 5.5).

        Catches non-traceable reward code (value-dependent Python control flow,
        in-place mutation, .item()) and surfaces it as an ExecutionError so the
        Orchestrator re-prompts. No-op when JAX is unavailable."""
        try:  # pragma: no cover - GPU/JAX host only
            import jax
            import jax.numpy as jnp
        except ImportError:
            return

        # A minimal duck-typed mjx.Data: just the attributes the prompt promises.
        class _FakeData:
            def __init__(self):
                self.qpos = jnp.zeros(26)
                self.qvel = jnp.zeros(25)
                self.xpos = jnp.zeros((21, 3))
                self.actuator_force = jnp.zeros(19)
                self.cfrc_ext = jnp.zeros((21, 6))

        def _traced(action, goal_xy, success_radius):
            reward, _ = self.__call__(_FakeData(), action, goal_xy, success_radius)
            return reward

        try:
            jax.eval_shape(
                _traced,
                jnp.zeros(19),
                jnp.zeros(2),
                jnp.float32(0.5),
            )
        except Exception as exc:  # noqa: BLE001
            raise ExecutionError(
                f"Generated reward is not jax.jit-traceable "
                f"({type(exc).__name__}: {exc}). Avoid value-dependent Python "
                f"control flow (use jnp.where/lax.select), in-place mutation, "
                f"and .item(); operate on jnp arrays only."
            ) from exc

    @staticmethod
    def _split_output(output) -> tuple[Any, dict[str, Any]]:
        if not isinstance(output, tuple):
            return output, {}
        if len(output) != 2:
            raise ExecutionError(
                f"Reward function must return either a reward or a "
                f"(reward, components) pair; got a {len(output)}-tuple."
            )
        reward, components = output
        if components is None:
            components = {}
        if not isinstance(components, dict):
            raise ExecutionError(
                f"Reward components must be a dict[str, array]; got "
                f"{type(components).__name__}."
            )
        for key in components:
            if not isinstance(key, str):
                raise ExecutionError(
                    f"Reward component names must be strings; got a "
                    f"{type(key).__name__} key."
                )
        return reward, components
