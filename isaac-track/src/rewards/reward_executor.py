"""Reward_Executor — validation, sandboxed execution, and Isaac Lab wrapping.

This module owns the handling of untrusted, language-model-generated reward
code for the YOUR_REPO system (design.md → Components →
Reward_Executor, Req 4, 5, 6).

Scope of THIS file (Tasks 4.1 + 4.2)
------------------------------------
* :class:`ValidationResult` — the result value object (design.md → Data Models).
* :meth:`RewardExecutor.validate` — structural validation of generated code
  (Req 4.1–4.4): ``ast.parse`` *before any execution*, then an AST walk for a
  **top-level** ``compute_reward`` ``FunctionDef`` (Task 4.1).
* The **sandbox execution layer** (Task 4.2, Req 5.1–5.4):
  :class:`SandboxConfig`, the restricted-globals builder
  (:meth:`RewardExecutor.build_sandbox_globals`),
  :meth:`RewardExecutor.compile_reward` (compile/exec a validated function into
  the restricted namespace), and :meth:`RewardExecutor.execute_reward` /
  :meth:`RewardExecutor.run_reward` (wall-clock timeout, exception capture, and
  non-finite detection).

Scope of THIS file additionally covers (Task 4.3)
-------------------------------------------------
* :class:`GoalRef` — a lightweight, duck-typed protocol for the Goal the
  generated reward reads (the authoritative ``Goal`` data model is owned by
  Task 8.1; this protocol keeps the wrapper self-contained and testable now).
* :class:`WrappedReward` — the callable backing each Isaac Lab ``RewTerm``. It
  conforms to ``fn(env, ...) -> torch.Tensor`` of shape ``(num_envs,)`` (Req
  6.1), surfaces the Goal to the generated ``compute_reward`` (Req 6.2), records
  every named reward component per step (Req 6.3), and exposes them through
  :pyattr:`WrappedReward.last_components` (Req 6.4).
* :meth:`RewardExecutor.wrap` — validates + sandbox-compiles the code (reusing
  :meth:`RewardExecutor.compile_reward`) and returns the wrapped ``RewTerm``
  callable(s).

The **live Isaac Lab registration** on the env's ``RewardManager`` is Task 9.1
and is intentionally NOT performed here; ``wrap`` is validated against a FAKE
``env`` exposing base pose/velocity, joint pos/vel, joint torques, foot contact
forces, and a Goal.

torch is import-light by design
-------------------------------
Reward bodies use ``torch``, but importing this module must NOT require it: the
``validate`` surface (and its property tests) run on the controller host without
torch. Therefore ``torch`` is imported **lazily**, inside the sandbox builder,
and only added to the allowlist when actually importable. The restricted-globals
rejection, timeout, exception-capture, and non-finite paths all work with plain
Python floats/lists, so the non-torch paths remain fully exercisable.

Design note (why validate RETURNS rather than RAISES)
-----------------------------------------------------
Per the design, ``validate`` reports its outcome as a :class:`ValidationResult`
(``ok`` / descriptive ``error``) so the Orchestrator can re-prompt the
Qwen_Client with the error text (Req 12.1). The sandbox entry points
(:meth:`compile_reward` / :meth:`run_reward`) instead RAISE — they must refuse
to execute invalid code — surfacing :class:`~src.exceptions.ValidationError`,
:class:`~src.exceptions.TimeoutError`, :class:`~src.exceptions.ExecutionError`,
and :class:`~src.exceptions.DivergenceError` from the shared taxonomy.

Timeout mechanism (documented choice)
-------------------------------------
The wall-clock bound (Req 5.2) is enforced with a **worker-thread watchdog**:
the reward runs in a daemon thread that the caller ``join``\\s with the time
limit. If the thread is still alive past the limit we raise
:class:`~src.exceptions.TimeoutError`. This is chosen over ``signal.alarm``
because signals only fire on the **main** thread, whereas PPO training drives
reward evaluation from worker threads; a thread watchdog works regardless of the
calling thread. The tradeoff (documented in design.md → *Sandbox Safety
Tradeoff*) is that a runaway thread cannot be force-killed in CPython — it is
abandoned as a daemon and dies with the process. This is an accepted guardrail
against accidental hangs, not OS-level isolation.
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
    "RewTermSpec",
    "LiveRewardBinding",
    "REWARD_FUNCTION_NAME",
    "ALLOWED_BUILTINS",
    "GOAL_ENV_ATTR",
    "ENV_ACCESSOR_HINT",
]

# The single function name the Language_Model is contracted to define
# (Reward_Function in the glossary; Req 4.3).
REWARD_FUNCTION_NAME = "compute_reward"


def _looks_like_module(obj: object) -> bool:
    """True if ``obj`` is a Python module (used to build the import allowlist)."""
    import types as _types  # noqa: PLC0415

    return isinstance(obj, _types.ModuleType)

# --------------------------------------------------------------------------- #
# Restricted-globals allowlist (Req 5.1)
# --------------------------------------------------------------------------- #
# The curated subset of builtins exposed inside the sandbox. Everything else —
# notably ``open``, ``eval``, ``exec``, ``compile``, ``__import__``, ``input``,
# ``exit``, ``globals``, ``vars`` — is intentionally EXCLUDED so that a reward
# body referencing such a name fails with a NameError (captured as an
# ExecutionError; Property 8). This is an allowlist, never a denylist.
ALLOWED_BUILTINS: frozenset[str] = frozenset(
    {
        # constants
        "True",
        "False",
        "None",
        # numeric / math helpers commonly used by reward shaping
        "abs",
        "min",
        "max",
        "sum",
        "round",
        "pow",
        "divmod",
        "float",
        "int",
        "bool",
        "complex",
        # sequence / iteration helpers
        "len",
        "range",
        "enumerate",
        "zip",
        "map",
        "filter",
        "sorted",
        "reversed",
        "all",
        "any",
        # container constructors
        "list",
        "dict",
        "tuple",
        "set",
        "frozenset",
        "str",
        # introspection that is safe and frequently needed
        "isinstance",
        "issubclass",
        "getattr",
        "hasattr",
        # exceptions a reward body may legitimately raise/catch
        "Exception",
        "ValueError",
        "TypeError",
        "ZeroDivisionError",
        "ArithmeticError",
        "OverflowError",
    }
)


@dataclass
class ValidationResult:
    """Outcome of :meth:`RewardExecutor.validate` (design.md → Data Models).

    Attributes:
        ok: ``True`` only when the code parses AND defines a top-level
            ``compute_reward`` function (Req 4.4).
        error: ``None`` when ``ok`` is ``True``; otherwise a non-empty,
            descriptive message explaining the rejection (Req 4.2, 4.3).
    """

    ok: bool
    error: str | None = None


@dataclass
class SandboxConfig:
    """Configuration for the restricted sandbox (design.md → Reward_Executor).

    Attributes:
        time_limit_s: Wall-clock bound for a single reward execution (Req 5.2).
            Sourced from ``Config.sandbox_time_limit_s``. Must be positive.
        extra_allowed_modules: Optional additional module names to expose inside
            the sandbox beyond the built-in defaults (``math`` and, when
            importable, ``torch``). Defaults to none.
    """

    time_limit_s: float = 5.0
    extra_allowed_modules: tuple[str, ...] = field(default_factory=tuple)


@runtime_checkable
class GoalRef(Protocol):
    """Lightweight, duck-typed handle to the Goal the reward reads (Req 6.2).

    The authoritative ``Goal`` data model (point B position + Success_Radius) is
    owned by Task 8.1, and the live binding onto the env's ``RewardManager`` is
    Task 9.1. To keep :meth:`RewardExecutor.wrap` self-contained and testable
    now — against a FAKE ``env`` and without Isaac Lab — this module accepts any
    object that *structurally* looks like a Goal: it exposes a ground-plane
    target ``position_xy`` and an arrival ``success_radius_m``.

    The wrapper does not interpret these fields itself; it simply makes the
    Goal reachable from the generated ``compute_reward`` (which reads it off the
    ``env``). Any object with these attributes — the future real ``Goal``
    dataclass, a test double, or a namedtuple — satisfies the protocol.
    """

    position_xy: tuple[float, float]
    success_radius_m: float


# Attribute under which the wrapper publishes the bound Goal onto ``env`` so the
# generated ``compute_reward`` can read it (mirrors the per-env Goal buffer the
# live env will carry; design.md → "Introducing the Goal ... into the env").
GOAL_ENV_ATTR = "goal"

# Documented, stable env-access surface advertised to the Language_Model in the
# prompt templates (prompts/initial_reward.txt → "STATE API"). The env-access
# guard (:class:`_EnvAccessProxy`) names these in its error when generated code
# references an attribute the env does not expose, so a hallucinated accessor
# (e.g. ``env.root_pos``) fails with actionable feedback the Orchestrator can
# re-prompt with (Req 12.1) instead of a cryptic ``AttributeError`` that burns a
# training iteration.
ENV_ACCESSOR_HINT = (
    "Valid env state accessors: "
    'env.scene["robot"].data.{root_pos_w, root_quat_w, root_lin_vel_b, '
    "root_ang_vel_b, joint_pos, joint_vel, applied_torque}; "
    'env.scene["contact_forces"].data.net_forces_w; '
    "env.goal.position_xy; env.goal.success_radius_m. "
    "Read base lin/ang velocity and joint pos/vel from the obs tensor by index "
    "when possible; use env accessors only for world-frame base position, "
    "torques, and contact forces."
)


def _resolve_live_obs(env: Any) -> Any:
    """Best-effort resolve the current observation tensor from a live env.

    Isaac Lab's ``RewardManager`` calls reward terms as ``fn(env)`` with no
    observation argument, and a manager-based env does not expose ``env.obs``.
    The policy observation is instead available via, in priority order:

    1. ``env.obs_buf`` — the cached observation dict the env fills each step,
       keyed by observation group (``"policy"`` is the proprio+goal vector the
       generated reward expects). Older single-tensor envs store the tensor
       directly here.
    2. ``env.obs`` — legacy/simple envs and the fake-env test harness.
    3. ``env.observation_manager.compute()`` — recompute on demand as a last
       resort (returns the same per-group dict).

    Returns the ``(num_envs, obs_dim)`` policy observation tensor, or ``None``
    when nothing is resolvable (the reward must then rely on env accessors).
    """
    # (1) The cached per-step observation buffer.
    obs = _extract_policy_group(getattr(env, "obs_buf", None))
    if obs is not None:
        return obs

    # (2) Legacy/simple single-tensor surface and the fake-env test harness.
    obs = _extract_policy_group(getattr(env, "obs", None))
    if obs is not None:
        return obs

    # (3) Recompute from the observation manager as a last resort.
    obs_manager = getattr(env, "observation_manager", None)
    compute = getattr(obs_manager, "compute", None) if obs_manager is not None else None
    if callable(compute):
        try:
            return _extract_policy_group(compute())
        except Exception:  # noqa: BLE001 - never let obs recompute crash the step
            return None
    return None


def _extract_policy_group(value: Any) -> Any:
    """Pull the ``"policy"`` group tensor out of an obs container (or pass through).

    Manager-based envs return observations as a dict keyed by observation group;
    the generated reward expects the ``"policy"`` vector. A plain tensor (single
    group, or the fake-env harness) is returned unchanged.
    """
    if value is None:
        return None
    if isinstance(value, dict):
        if "policy" in value:
            return value["policy"]
        # Single-group dict: return its only tensor.
        if len(value) == 1:
            return next(iter(value.values()))
        return None
    return value


def _resolve_live_actions(env: Any) -> Any:
    """Best-effort resolve the most-recent action tensor from a live env.

    Manager-based Isaac Lab envs expose the processed action on
    ``env.action_manager.action``; they do not expose ``env.actions``. Fall back
    to ``env.actions`` for legacy/simple envs and the fake-env test harness.
    Returns ``None`` when no action surface is available.
    """
    action_manager = getattr(env, "action_manager", None)
    if action_manager is not None:
        action = getattr(action_manager, "action", None)
        if action is not None:
            return action
    return getattr(env, "actions", None)


class _EnvAccessProxy:
    """Transparent, read-through guard around the env passed to generated code.

    The generated ``compute_reward(env, obs, actions)`` is untrusted and is
    prompted to use a specific Isaac Lab accessor surface. In practice the model
    occasionally invents a top-level attribute (observed: ``env.root_pos`` /
    ``env.prev_root_pos``). Passing the raw env means such a miss raises a bare
    ``AttributeError`` whose message names nothing useful; the Orchestrator then
    re-prompts with an unhelpful error and may waste the iteration.

    This proxy forwards **every** existing attribute and item access straight to
    the wrapped env (so all valid code — including the nested
    ``env.scene["robot"].data.root_pos_w`` pattern — behaves identically), but
    when an attribute is genuinely absent it raises an ``AttributeError`` whose
    message appends :data:`ENV_ACCESSOR_HINT`. The executor captures that as an
    :class:`~src.exceptions.ExecutionError` (Req 5.3), giving the refine loop a
    precise, actionable message (Req 12.1).

    It is deliberately thin: attribute *reads* are guarded; assignments and item
    access pass straight through. Only the top-level ``env.<attr>`` lookup is
    intercepted — nested objects (``.scene``, ``.data``) are the real ones, so
    this adds a guardrail without altering any valid behavior.
    """

    __slots__ = ("_env",)

    def __init__(self, env: Any) -> None:
        object.__setattr__(self, "_env", env)

    def __getattr__(self, name: str) -> Any:
        # __getattr__ is only called when normal lookup fails, i.e. for names
        # not on the proxy itself (everything except the slotted ``_env``).
        env = object.__getattribute__(self, "_env")
        try:
            return getattr(env, name)
        except AttributeError as exc:
            raise AttributeError(
                f"Generated reward referenced unknown env attribute "
                f"'{name}'. {ENV_ACCESSOR_HINT}"
            ) from exc

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(object.__getattribute__(self, "_env"), name, value)

    def __getitem__(self, key: Any) -> Any:
        return object.__getattribute__(self, "_env")[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        object.__getattribute__(self, "_env")[key] = value


class RewardExecutor:
    """Validates, sandboxes, and wraps generated reward code.

    Tasks 4.1 (:meth:`validate`) and 4.2 (the sandbox execution layer) are
    implemented here. ``wrap`` (Task 4.3) — live Isaac Lab ``RewTerm`` binding —
    is intentionally not present yet. Importing this module never requires
    ``torch``: it is imported lazily inside :meth:`build_sandbox_globals`.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        """Create an executor.

        Args:
            config: Sandbox configuration (timeout + allowed modules). When
                omitted, :class:`SandboxConfig` defaults are used so the pure
                ``validate`` surface stays usable without any wiring.
        """
        self.config = config if config is not None else SandboxConfig()

    def validate(self, code: str) -> ValidationResult:
        """Structurally validate generated reward code (Req 4.1–4.4).

        Validation is **parse/inspect only** — rejected code is NEVER executed
        (Req 4.1). The steps are:

        1. ``ast.parse(code)`` before any execution. A :class:`SyntaxError`
           yields ``ok=False`` with a descriptive error (Req 4.2).
        2. Walk the parsed module's **top-level** body for an
           :class:`ast.FunctionDef` (or ``async def``) named ``compute_reward``.
           Absence yields ``ok=False`` with a descriptive error (Req 4.3).
        3. When both hold, return ``ok=True`` (Req 4.4).

        Args:
            code: The candidate reward source returned by the Qwen_Client.

        Returns:
            A :class:`ValidationResult`. ``error`` is non-empty whenever
            ``ok`` is ``False``.
        """
        # --- Req 4.1 / 4.2: parse before any execution; reject unparseable. ---
        try:
            module = ast.parse(code)
        except SyntaxError as exc:
            # Build a descriptive, non-empty message including the location.
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
            # ast.parse can raise ValueError (e.g. embedded null bytes) or
            # TypeError (e.g. code is not a string). Treat as a parse failure
            # rather than letting it escape — still no execution occurs.
            return ValidationResult(
                ok=False,
                error=f"Generated reward code could not be parsed: {exc}.",
            )

        # --- Req 4.3: require a TOP-LEVEL `compute_reward` function. ---
        # Only inspect the module's direct children so a `compute_reward`
        # nested inside another function/class does not satisfy the contract.
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

        # --- Req 4.4: parses and defines compute_reward -> accept. ---
        return ValidationResult(ok=True, error=None)

    # ===================================================================== #
    # Task 4.2 — Sandbox execution layer (Req 5.1–5.4)
    # ===================================================================== #

    def build_sandbox_globals(self) -> dict[str, object]:
        """Build the restricted-globals namespace for reward execution (Req 5.1).

        The returned ``globals`` mapping exposes ONLY an allowlisted surface:

        * ``__builtins__`` restricted to :data:`ALLOWED_BUILTINS` — every other
          builtin (``open``, ``eval``, ``exec``, ``__import__``, ``os`` access
          helpers, etc.) is absent, so referencing one raises ``NameError``
          (captured downstream as :class:`~src.exceptions.ExecutionError`;
          Property 8 / Req 5.1).
        * ``math`` — always available (pure stdlib, no torch).
        * ``torch`` — added **only if importable**, imported lazily here so this
          module stays import-light on the controller host. When torch is
          unavailable a reward referencing it simply fails with a captured
          ``NameError`` rather than breaking the sandbox builder.
        * any ``extra_allowed_modules`` from :class:`SandboxConfig` that import
          successfully.

        Returns:
            A fresh ``dict`` suitable as the ``globals`` argument to ``exec``.
            A new mapping is returned on every call so executions never share
            mutable module-level state.
        """
        # Restrict builtins to the curated allowlist. We resolve each name off
        # the real ``builtins`` module; unknown names are skipped defensively.
        restricted_builtins = {
            name: getattr(_builtins, name)
            for name in ALLOWED_BUILTINS
            if hasattr(_builtins, name)
        }

        sandbox: dict[str, object] = {
            # NOTE: providing a dict (not the builtins module) is what disables
            # implicit access to the full builtin set inside exec'd code.
            "__builtins__": restricted_builtins,
            "math": math,
        }

        # Lazily expose torch ONLY when present (reward bodies use it, but the
        # controller host / validate path must not require it).
        try:  # pragma: no cover - exercised only where torch is installed
            import torch  # noqa: PLC0415 - intentional lazy import

            sandbox["torch"] = torch
        except ImportError:
            pass

        # Optional operator-approved extra modules.
        for module_name in self.config.extra_allowed_modules:
            try:
                module = __import__(module_name)
            except ImportError:
                continue
            # __import__ returns the top-level package; bind under its leaf-free
            # top name, which is what the allowlist entry refers to.
            sandbox[module_name.split(".")[0]] = module

        # Provide a RESTRICTED ``__import__`` so a generated reward that writes
        # ``import torch`` / ``import math`` inside the function works, while
        # arbitrary imports stay blocked. Many models emit an in-body import even
        # when told the modules are pre-bound; without this they fail at runtime
        # with "ImportError: __import__ not found" (the dict-__builtins__ has no
        # __import__). The shim resolves ONLY names already allowlisted into the
        # sandbox namespace and raises ImportError for anything else, so it does
        # not widen the sandbox surface.
        _allowed_modules = {
            name: obj
            for name, obj in sandbox.items()
            if name not in ("__builtins__",) and _looks_like_module(obj)
        }

        def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):  # noqa: A002
            top = name.split(".")[0]
            if top in _allowed_modules:
                return _allowed_modules[top]
            raise ImportError(
                f"import of {name!r} is not allowed in the reward sandbox; "
                f"available modules: {sorted(_allowed_modules)}"
            )

        restricted_builtins["__import__"] = _restricted_import

        return sandbox

    def compile_reward(self, code: str):
        """Validate, compile, and exec ``code`` into the restricted namespace.

        This is the sandbox entry point that turns a validated source string
        into a callable ``compute_reward`` bound to the restricted globals
        (Req 5.1). Unlike :meth:`validate`, it RAISES on bad input because it
        is on the execution path and must refuse to run invalid code.

        Args:
            code: Candidate reward source.

        Returns:
            The ``compute_reward`` callable, with ``__globals__`` set to the
            restricted sandbox namespace.

        Raises:
            ValidationError: If ``code`` does not pass :meth:`validate` (not
                parseable, or no top-level ``compute_reward``).
            ExecutionError: If executing the module body to define the function
                raises (e.g. an import or top-level statement referencing a
                forbidden name).
        """
        result = self.validate(code)
        if not result.ok:
            raise ValidationError(
                result.error or "Generated reward code failed validation."
            )

        sandbox = self.build_sandbox_globals()
        # Compile then exec the module body in the restricted namespace so that
        # the `def compute_reward` (and any allowlisted module references it
        # makes at definition time) bind against the restricted globals.
        try:
            compiled = compile(code, filename="<generated_reward>", mode="exec")
            exec(compiled, sandbox)  # noqa: S102 - restricted namespace by design
        except Exception as exc:  # noqa: BLE001 - capture ALL to surface as ExecutionError
            raise ExecutionError(
                f"Generated reward code failed to define "
                f"'{REWARD_FUNCTION_NAME}': {type(exc).__name__}: {exc}"
            ) from exc

        func = sandbox.get(REWARD_FUNCTION_NAME)
        if not callable(func):
            # Defensive: validate guarantees a top-level def, but a same-named
            # non-callable could shadow it via exotic code.
            raise ExecutionError(
                f"'{REWARD_FUNCTION_NAME}' is not callable after execution."
            )
        return func

    def execute_reward(self, func, *args, **kwargs):
        """Invoke a sandboxed reward ``func`` under the wall-clock timeout (Req 5.2, 5.3).

        Runs ``func(*args, **kwargs)`` in a daemon worker thread and waits up to
        :attr:`SandboxConfig.time_limit_s` for it to finish (see the module
        docstring for why a thread watchdog is used instead of ``signal``).

        Args:
            func: A callable previously produced by :meth:`compile_reward`.
            *args: Positional arguments forwarded to the reward (e.g. ``env``).
            **kwargs: Keyword arguments forwarded to the reward.

        Returns:
            Whatever the reward returns. Non-finite detection is the caller's
            responsibility via :meth:`check_finite` (or use :meth:`run_reward`).

        Raises:
            TimeoutError: If execution exceeds the configured time limit
                (Req 5.2). The runaway thread is abandoned as a daemon.
            ExecutionError: If the reward raises any exception during execution;
                the original failure is identified and chained (Req 5.3).
        """
        result_box: dict[str, object] = {}
        error_box: dict[str, BaseException] = {}

        def _worker() -> None:
            try:
                result_box["value"] = func(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - relay to caller thread
                error_box["error"] = exc

        worker = threading.Thread(
            target=_worker,
            name="reward-sandbox",
            daemon=True,
        )
        worker.start()
        worker.join(self.config.time_limit_s)

        if worker.is_alive():
            # Req 5.2: exceeded the wall-clock bound. We cannot force-kill the
            # thread in CPython; it is a daemon and dies with the process.
            raise TimeoutError(
                f"Reward execution exceeded the sandbox time limit of "
                f"{self.config.time_limit_s}s."
            )

        if "error" in error_box:
            exc = error_box["error"]
            # Req 5.3: capture the failure and surface a descriptive
            # ExecutionError rather than propagating the raw exception
            # (Property 9). This also covers NameError from forbidden names
            # excluded by the restricted globals (Property 8 / Req 5.1).
            raise ExecutionError(
                f"Reward function raised during execution: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

        return result_box.get("value")

    def check_finite(self, output) -> None:
        """Raise :class:`~src.exceptions.DivergenceError` on non-finite output (Req 5.4).

        Detects NaN/inf anywhere in the reward output. The check works with
        plain Python numbers and nested lists/tuples (no torch required); when
        the output is a torch tensor (or any object exposing ``isfinite`` /
        ``isnan`` semantics) it is inspected element-wise.

        Args:
            output: The value returned by a reward function.

        Raises:
            DivergenceError: If any scalar within ``output`` is NaN or infinite
                (Property 10 / Req 5.4).
        """
        if not self._is_all_finite(output):
            raise DivergenceError(
                "Reward function returned a non-finite value (NaN or infinity)."
            )

    @staticmethod
    def _is_all_finite(output) -> bool:
        """Return ``True`` iff every scalar in ``output`` is finite.

        Handles, in order: torch tensors / numpy-like arrays (via duck-typed
        ``isfinite``/``all``), plain numbers, and nested iterables of these.
        Unknown non-iterable, non-numeric objects are treated as finite (there
        is nothing numeric to diverge).
        """
        # --- Numbers: the base case used by the torch-free test paths. ---
        if isinstance(output, bool):
            return True
        if isinstance(output, (int, float)):
            return math.isfinite(output)
        if isinstance(output, complex):
            return math.isfinite(output.real) and math.isfinite(output.imag)

        # --- torch tensors / numpy arrays: duck-type without importing torch.
        # ``torch.isfinite(t).all()`` -> tensor; ``bool(...)`` collapses it.
        isfinite = getattr(output, "isfinite", None)
        if callable(isfinite):
            try:
                finite_mask = isfinite()
                all_fn = getattr(finite_mask, "all", None)
                if callable(all_fn):
                    return bool(all_fn())
                return bool(finite_mask)
            except Exception:  # noqa: BLE001 - fall through to iterable handling
                pass

        # --- Strings are iterable but carry no numeric divergence. ---
        if isinstance(output, (str, bytes)):
            return True

        # --- Nested iterables (lists/tuples/sets/generators) of the above. ---
        try:
            iterator = iter(output)
        except TypeError:
            # Non-numeric, non-iterable object: nothing to diverge.
            return True
        return all(RewardExecutor._is_all_finite(item) for item in iterator)

    def run_reward(self, code: str, *args, **kwargs):
        """End-to-end sandbox run: compile, execute under timeout, check finite.

        Convenience composition of the Task 4.2 primitives, in the order the
        ``wrap()`` callable (Task 4.3) will use them:

        1. :meth:`compile_reward` — validate + restricted-globals compile
           (Req 5.1).
        2. :meth:`execute_reward` — invoke under the wall-clock timeout, with
           exception capture (Req 5.2, 5.3).
        3. :meth:`check_finite` — reject non-finite output (Req 5.4).

        Args:
            code: Candidate reward source.
            *args: Positional arguments forwarded to ``compute_reward``.
            **kwargs: Keyword arguments forwarded to ``compute_reward``.

        Returns:
            The reward output once it has passed the finite check.

        Raises:
            ValidationError, ExecutionError, TimeoutError, DivergenceError:
                per the steps above.
        """
        func = self.compile_reward(code)
        output = self.execute_reward(func, *args, **kwargs)
        self.check_finite(output)
        return output

    # ===================================================================== #
    # Task 4.3 — Isaac Lab wrapping into RewTerm callables (Req 6.1–6.4)
    # ===================================================================== #

    def wrap(self, code: str, goal_ref: "GoalRef | None") -> list["WrappedReward"]:
        """Validate, sandbox-compile, and wrap ``code`` as Isaac Lab RewTerm(s).

        This is the Task 4.3 entry point. It reuses :meth:`compile_reward` to
        validate (Req 4.x) and bind the generated ``compute_reward`` into the
        restricted-globals namespace (Req 5.1), then returns a
        :class:`WrappedReward` callable that conforms to the Isaac Lab reward
        interface ``fn(env, ...) -> torch.Tensor`` of shape ``(num_envs,)``
        (Req 6.1) bound to ``goal_ref`` (Req 6.2).

        A ``list`` is returned to match the design's
        ``wrap(...) -> list[RewTerm]`` signature: the generated function emits a
        set of *named, weighted components* internally and the executor records
        each (Req 6.3, 6.4), so a single ``WrappedReward`` that sums and reports
        them backs the registered term. Returning a list keeps the door open for
        future per-component term registration without changing callers.

        The live registration on the env's ``RewardManager`` is Task 9.1 and is
        deliberately not done here; the returned callable is exercised against a
        FAKE ``env`` in the meantime.

        Args:
            code: Candidate reward source (already produced by the Qwen_Client).
            goal_ref: A duck-typed Goal handle (see :class:`GoalRef`) the wrapped
                reward surfaces to the generated function via the ``env``. May be
                ``None`` when the live env already owns its per-env Goal buffer
                (Task 9.1) — in that case the wrapper leaves the env's existing
                ``goal`` untouched (see :meth:`WrappedReward._bind_goal`).

        Returns:
            A single-element list containing the :class:`WrappedReward` callable.

        Raises:
            ValidationError: If ``code`` does not pass :meth:`validate`.
            ExecutionError: If executing the module body to define the function
                raises (forbidden name, bad top-level statement, ...).
        """
        func = self.compile_reward(code)
        wrapped = WrappedReward(func=func, goal_ref=goal_ref, executor=self)
        return [wrapped]


class WrappedReward:
    """Isaac Lab ``RewTerm`` callable backing a generated ``compute_reward``.

    Conforms to the manager-based reward-term signature
    ``fn(env, ...) -> torch.Tensor`` of shape ``(num_envs,)`` (Req 6.1). On each
    call it:

    1. Surfaces the bound :class:`GoalRef` onto ``env`` (``env.goal``) so the
       generated reward can read the Goal alongside base pose/velocity, joint
       pos/vel, joint torques, and foot contact forces (Req 6.2).
    2. Runs the generated ``compute_reward(env, obs, actions)`` through the
       executor's sandbox primitives — wall-clock timeout + exception capture
       (Req 5.2, 5.3) — and the non-finite check (Req 5.4).
    3. Interprets the return value. The contract (see ``prompts/initial_reward``)
       is ``return reward, components`` where ``reward`` is shape ``(num_envs,)``
       and ``components`` is ``dict[str, Tensor]`` each ``(num_envs,)``. A bare
       reward (no components) is also accepted defensively.
    4. Records the named components for the step and exposes them via
       :pyattr:`last_components` (Req 6.3, 6.4).

    ``obs`` / ``actions`` are read from ``env`` when present (``env.obs`` /
    ``env.actions``) or may be passed explicitly as keyword arguments; this keeps
    the callable usable both from Isaac Lab (which calls ``fn(env)``) and from a
    fake-``env`` test harness.
    """

    def __init__(
        self,
        func,
        goal_ref: "GoalRef | None",
        executor: "RewardExecutor",
    ) -> None:
        """Bind a compiled reward function to a Goal and an executor.

        Args:
            func: The ``compute_reward`` callable from :meth:`RewardExecutor.compile_reward`.
            goal_ref: The Goal handle surfaced to the reward each call (Req 6.2).
                When ``None``, the wrapper does NOT publish a Goal onto the env
                and instead relies on the live env already carrying its own
                per-env Goal buffer (the Task 9.1 live-binding case).
            executor: The owning :class:`RewardExecutor`, reused for its sandbox
                timeout/exception-capture and finite-check primitives.
        """
        self._func = func
        self._goal_ref = goal_ref
        self._executor = executor
        self._last_components: dict[str, Any] = {}

    @property
    def goal_ref(self) -> "GoalRef | None":
        """The Goal handle this term is bound to (Req 6.2).

        ``None`` when the term defers to the live env's own per-env Goal buffer
        (Task 9.1 live-binding case).
        """
        return self._goal_ref

    @property
    def last_components(self) -> dict[str, Any]:
        """Per-component reward values recorded on the most recent call (Req 6.3, 6.4).

        Keyed by the component names the generated reward emitted, each value
        the per-environment contribution (a ``torch.Tensor`` of shape
        ``(num_envs,)`` under Isaac Lab; plain numbers/sequences in torch-free
        tests). Empty until the term has been evaluated at least once. A copy is
        returned so callers cannot mutate the recorded snapshot.
        """
        return dict(self._last_components)

    def __call__(self, env, obs=None, actions=None, **kwargs):
        """Evaluate the reward for one step over all parallel envs (Req 6.1–6.4).

        Args:
            env: The (manager-based) environment. The Goal is surfaced onto it
                and the generated reward reads state from it.
            obs: Optional observation tensor ``(num_envs, obs_dim)``. When
                ``None``, resolved from the live env's ``obs_buf`` /
                ``observation_manager`` (manager-based env) or ``env.obs``
                (legacy/fake env) via :func:`_resolve_live_obs`.
            actions: Optional action tensor ``(num_envs, act_dim)``. When
                ``None``, resolved from ``env.action_manager.action`` (live env)
                or ``env.actions`` (legacy/fake env) via
                :func:`_resolve_live_actions`.
            **kwargs: Forwarded to the generated ``compute_reward`` (Isaac Lab
                may pass term-specific keyword arguments).

        Returns:
            The per-environment reward of shape ``(num_envs,)``.

        Raises:
            TimeoutError: Reward exceeded the sandbox wall-clock limit (Req 5.2).
            ExecutionError: Reward raised during execution (Req 5.3).
            DivergenceError: Reward (or any component) was non-finite (Req 5.4).
        """
        # Req 6.2: make the Goal reachable from the generated reward via env.
        self._bind_goal(env)

        # Isaac Lab's RewardManager invokes reward terms as ``fn(env)`` with no
        # obs/actions, and a manager-based env does NOT expose ``env.obs`` /
        # ``env.actions``. It carries the most-recent observation in the
        # ``env.obs_buf`` dict (keyed by observation group) and the processed
        # action on ``env.action_manager.action``. Resolve from those live
        # surfaces so the generated reward receives real tensors instead of
        # ``None`` (which previously surfaced as ``'NoneType' has no attribute
        # 'device'`` from the generated body).
        if obs is None:
            obs = _resolve_live_obs(env)
        if actions is None:
            actions = _resolve_live_actions(env)

        # Defense-in-depth: pass the generated (untrusted) reward a transparent
        # guard around env. Valid accessors pass straight through; a hallucinated
        # top-level attribute (e.g. env.root_pos) raises an AttributeError naming
        # the valid surface, captured as an ExecutionError so the Orchestrator can
        # re-prompt with actionable feedback (Req 12.1) instead of burning the
        # iteration on a cryptic error.
        guarded_env = _EnvAccessProxy(env)

        # Run inside the sandbox (timeout + exception capture; Req 5.2, 5.3).
        output = self._executor.execute_reward(
            self._func, guarded_env, obs, actions, **kwargs
        )

        reward, components = self._split_output(output)

        # Req 5.4: reject non-finite reward AND non-finite components.
        self._executor.check_finite(reward)
        if components:
            self._executor.check_finite(list(components.values()))

        # Req 6.3 / 6.4: record each named component for this step.
        self._last_components = dict(components)
        return reward

    def _bind_goal(self, env) -> None:
        """Publish the bound Goal onto ``env`` so the reward can read it (Req 6.2).

        Mirrors the per-env Goal buffer the live env will carry. When this term
        was wrapped with an explicit ``goal_ref`` (the pure-logic / fake-env
        case), the Goal is published onto ``env.goal``. When ``goal_ref`` is
        ``None`` (the Task 9.1 live-binding case), the live env already owns its
        per-env Goal buffer, so the wrapper leaves ``env.goal`` untouched.
        """
        if self._goal_ref is None:
            # Live env owns its Goal buffer; do not overwrite it.
            return
        try:
            setattr(env, GOAL_ENV_ATTR, self._goal_ref)
        except (AttributeError, TypeError):
            # Some env objects forbid attribute assignment (e.g. slotted/frozen).
            # The Goal remains accessible to the reward via the closure-bound
            # term in that case; this is a best-effort surface for fakes/tests.
            pass

    @staticmethod
    def _split_output(output) -> tuple[Any, dict[str, Any]]:
        """Normalize a reward return into ``(reward, components)``.

        Accepts the contracted ``(reward, components)`` pair, as well as a bare
        reward (treated as zero named components) so a minimal reward body still
        wraps cleanly.

        Raises:
            ExecutionError: If the return shape is neither a bare reward nor a
                ``(reward, dict)`` pair (Req 5.3 — a contract violation surfaced
                as an execution error).
        """
        # Bare reward, no components.
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
                f"Reward components must be a dict[str, tensor]; got "
                f"{type(components).__name__}."
            )
        # Defensive: component keys must be strings for named recording (Req 6.3).
        for key in components:
            if not isinstance(key, str):
                raise ExecutionError(
                    f"Reward component names must be strings; got a "
                    f"{type(key).__name__} key."
                )
        return reward, components


# ===================================================================== #
# Task 9.1 — Live Isaac Lab env binding (Req 6.1, 6.2, 6.3, 6.4)
# ===================================================================== #
#
# This layer takes the ``WrappedReward`` callables produced by
# :meth:`RewardExecutor.wrap` (Task 4.3, validated against a FAKE env) and binds
# them onto a LIVE manager-based Isaac Lab env's ``RewardManager``:
#
#   * registers each wrapped callable as a ``RewTerm`` on the manager (Req 6.1);
#   * gives every invocation access to the live environment state — base
#     pose/velocity, joint pos/vel, joint torques, foot contact forces — and the
#     live per-env Goal buffer the goal-conditioned env carries (Req 6.2);
#   * records the per-component reward values emitted each step (Req 6.3) and
#     exposes them for logging / Eureka-style textual feedback (Req 6.4).
#
# Isaac Lab (``isaaclab`` / ``RewardManager`` / ``RewardTermCfg``) is NOT
# importable in the controller/dev environment, so every Isaac Lab touchpoint is
# **lazy and guarded**: imports happen inside methods, and registration is
# duck-typed so the binding remains unit-testable against a fake env + fake
# reward manager that expose the same surface the live objects do.

# Stable names under which the binding surfaces live environment state to the
# generated reward (Req 6.2). These mirror the manager-based H1 env's
# articulation / contact-sensor data. A reward body may read them off the
# ``RewTermSpec`` passed in via ``params``/closure or off the env directly.
ENV_STATE_FIELDS: tuple[str, ...] = (
    "base_pose",            # base position + orientation (num_envs, 7)
    "base_velocity",        # base lin+ang velocity        (num_envs, 6)
    "joint_pos",            # joint positions              (num_envs, 19)
    "joint_vel",            # joint velocities             (num_envs, 19)
    "joint_torques",        # applied joint torques        (num_envs, 19)
    "foot_contact_forces",  # foot contact forces          (num_envs, n_feet, 3)
    "goal",                 # live per-env Goal buffer
)


@dataclass
class RewTermSpec:
    """A lab-agnostic description of a reward term to register (Req 6.1).

    This is the value object the binding hands to the (live or fake) reward
    manager. When Isaac Lab is importable, :meth:`LiveRewardBinding.register`
    converts each spec into a real ``RewardTermCfg`` (aliased ``RewTerm``);
    otherwise the spec itself is registered, which is sufficient for the fake
    manager used in tests and keeps the module import-light.

    Attributes:
        name: The term's name on the manager (the key its value is logged under).
        func: The ``fn(env, ...) -> torch.Tensor`` callable — a
            :class:`WrappedReward` — of shape ``(num_envs,)`` (Req 6.1).
        weight: Scalar weight applied by the manager. Defaults to ``1.0`` since
            the generated reward already encodes its own component weighting.
        params: Extra keyword params forwarded to the term on each call.
    """

    name: str
    func: Any
    weight: float = 1.0
    params: dict[str, Any] = field(default_factory=dict)


class LiveRewardBinding:
    """Binds wrapped reward terms onto a live env's ``RewardManager`` (Req 6.1–6.4).

    Lifecycle::

        binding = executor.bind_to_env(code, env)   # validate+compile+wrap+register
        ...                                          # PPO training drives the env
        log = binding.component_log                  # per-component values (Req 6.4)

    The binding registers one ``RewTerm`` per wrapped callable on the env's
    reward manager (Req 6.1). Each wrapped callable reads the live env (base
    pose/velocity, joint pos/vel, joint torques, foot contact forces) and the
    live per-env Goal buffer carried by the goal-conditioned env (Req 6.2); the
    binding does NOT publish its own Goal, deferring to the env's buffer. After
    each manager step the binding harvests every term's ``last_components`` into
    :pyattr:`component_log` (Req 6.3, 6.4).
    """

    #: Default term name used when registering a single wrapped reward.
    DEFAULT_TERM_NAME = "llm_generated_reward"

    def __init__(
        self,
        terms: list["WrappedReward"],
        executor: "RewardExecutor",
        term_names: list[str] | None = None,
    ) -> None:
        """Create a binding over already-wrapped reward terms.

        Args:
            terms: The :class:`WrappedReward` callables from
                :meth:`RewardExecutor.wrap` (bound with ``goal_ref=None`` so they
                read the live env's Goal buffer; Req 6.2).
            executor: The owning executor (kept for symmetry / future use).
            term_names: Optional explicit per-term names; defaults to
                ``llm_generated_reward`` (suffixed by index when >1 term).
        """
        self._executor = executor
        self._specs: list[RewTermSpec] = []
        for i, term in enumerate(terms):
            if term_names is not None and i < len(term_names):
                name = term_names[i]
            elif len(terms) == 1:
                name = self.DEFAULT_TERM_NAME
            else:
                name = f"{self.DEFAULT_TERM_NAME}_{i}"
            self._specs.append(RewTermSpec(name=name, func=term))
        self._registered = False
        self._manager = None

    @property
    def specs(self) -> list[RewTermSpec]:
        """The term specs this binding manages (copy)."""
        return list(self._specs)

    @property
    def wrapped_terms(self) -> list["WrappedReward"]:
        """The wrapped reward callables backing the registered terms."""
        return [spec.func for spec in self._specs]

    @property
    def component_log(self) -> dict[str, Any]:
        """Per-component reward values from the most recent evaluation (Req 6.3, 6.4).

        Aggregates ``last_components`` across all registered terms. When a single
        term is registered (the common case) its component names are surfaced
        directly; when multiple terms are registered, component keys are
        namespaced by term name (``"<term>/<component>"``) to keep them distinct.
        Empty until at least one term has been evaluated.
        """
        log: dict[str, Any] = {}
        single = len(self._specs) == 1
        for spec in self._specs:
            components = getattr(spec.func, "last_components", {}) or {}
            for comp_name, value in components.items():
                key = comp_name if single else f"{spec.name}/{comp_name}"
                log[key] = value
        return log

    def _resolve_reward_manager(self, env):
        """Locate the live env's ``RewardManager`` (duck-typed, guarded).

        Manager-based Isaac Lab envs expose ``env.reward_manager``; some wrap it
        under ``env.unwrapped.reward_manager``. We accept any object that exposes
        a registration surface (``add_term`` / ``set_term_cfg`` / a mutable term
        list). Returns the manager or raises ``ExecutionError`` when none is
        found, so a misconfigured env fails loudly rather than silently skipping
        registration.
        """
        for path in ("reward_manager", "_reward_manager"):
            manager = getattr(env, path, None)
            if manager is not None:
                return manager
        unwrapped = getattr(env, "unwrapped", None)
        if unwrapped is not None and unwrapped is not env:
            return self._resolve_reward_manager(unwrapped)
        raise ExecutionError(
            "Live env exposes no RewardManager to register the generated "
            "reward term on (looked for env.reward_manager / "
            "env.unwrapped.reward_manager)."
        )

    def _make_rew_term(self, spec: RewTermSpec):
        """Convert a :class:`RewTermSpec` into an Isaac Lab ``RewTerm`` if possible.

        Isaac Lab is imported lazily and only here. When it is unavailable (the
        dev/controller host) the plain :class:`RewTermSpec` is returned, which
        the fake manager in tests accepts. This keeps the module importable
        without ``isaaclab``.
        """
        try:  # pragma: no cover - exercised only where isaaclab is installed
            from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: PLC0415

            return RewTerm(func=spec.func, weight=spec.weight, params=dict(spec.params))
        except Exception:  # noqa: BLE001 - ImportError or cfg API drift -> fall back
            return spec

    def register(self, env) -> "LiveRewardBinding":
        """Register every term on the live env's ``RewardManager`` (Req 6.1).

        Registration is duck-typed so it works against both the live manager and
        the fake manager used in tests, trying, in order:

        1. ``manager.add_term(name, term)`` — explicit add API;
        2. ``manager.set_term_cfg(name, term)`` — Isaac Lab term-cfg setter;
        3. appending to a mutable ``active_terms`` / ``_term_cfgs`` list.

        Idempotent: calling twice does not double-register.

        Args:
            env: The live (goal-conditioned) manager-based env.

        Returns:
            ``self`` for chaining.

        Raises:
            ExecutionError: If no RewardManager or no registration surface is
                found on the env.
        """
        if self._registered:
            return self
        manager = self._resolve_reward_manager(env)
        for spec in self._specs:
            term = self._make_rew_term(spec)
            if not self._register_one(manager, spec.name, term):
                raise ExecutionError(
                    f"RewardManager {type(manager).__name__} exposes no "
                    f"supported registration surface (add_term / set_term_cfg / "
                    f"active_terms) for term '{spec.name}'."
                )
        self._manager = manager
        self._registered = True
        return self

    @staticmethod
    def _register_one(manager, name: str, term) -> bool:
        """Register one term on ``manager`` via the first supported surface.

        Returns ``True`` on success, ``False`` when no surface is available.

        Order of preference:
        1. ``add_term(name, term)`` — explicit add API (fakes/newer managers).
        2. ``set_term_cfg(name, term)`` — Isaac Lab term-cfg setter, but ONLY when
           ``name`` already exists (it raises for unknown terms).
        3. Append to the manager's internal term structures for a genuinely NEW
           term (the live Isaac Lab RewardManager case): extend ``_term_names`` /
           ``_term_cfgs`` and seed the per-term episode-sum buffer so the manager
           computes and logs it like any built-in term.
        4. Fallback mutable-container append (the fake manager in tests).
        """
        add_term = getattr(manager, "add_term", None)
        if callable(add_term):
            add_term(name, term)
            return True

        # Isaac Lab RewardManager: set_term_cfg only UPDATES an existing term and
        # raises for unknown names. Use it when the term already exists, OR when
        # the manager doesn't expose a discoverable term list (the fake-manager
        # test case, where set_term_cfg is an unconditional setter).
        existing_names = getattr(manager, "_term_names", None)
        set_term_cfg = getattr(manager, "set_term_cfg", None)
        if callable(set_term_cfg) and (
            not isinstance(existing_names, list) or name in existing_names
        ):
            set_term_cfg(name, term)
            return True

        # Genuinely new term on the live RewardManager: append to its internals.
        term_cfgs = getattr(manager, "_term_cfgs", None)
        if isinstance(existing_names, list) and isinstance(term_cfgs, list):
            existing_names.append(name)
            term_cfgs.append(term)
            num_envs = getattr(manager, "num_envs", None)
            device = getattr(manager, "device", None)
            try:  # pragma: no cover - requires torch + live manager
                import torch  # noqa: PLC0415
            except Exception:  # noqa: BLE001
                torch = None  # type: ignore[assignment]

            if torch is not None and num_envs is not None:
                # Seed the per-term episode-sum buffer the manager keeps for
                # logging (Req 6.4).
                episode_sums = getattr(manager, "_episode_sums", None)
                if isinstance(episode_sums, dict):
                    try:
                        episode_sums[name] = torch.zeros(
                            int(num_envs), dtype=torch.float, device=device
                        )
                    except Exception:  # noqa: BLE001
                        pass

                # CRITICAL: the manager pre-allocates the per-step reward buffer
                # ``_step_reward`` with shape ``(num_envs, num_terms)`` at init.
                # ``compute()`` writes ``_step_reward[:, term_idx] = value/dt`` for
                # every term, so a newly-appended term's column index now exceeds
                # the buffer width and raises ``IndexError: index N out of bounds
                # for dimension 1 with size N``. Widen the buffer to match the new
                # term count so the appended term has a column to write into.
                num_terms = len(existing_names)
                for buf_attr in ("_step_reward", "_episode_sum_reward"):
                    buf = getattr(manager, buf_attr, None)
                    if buf is None:
                        continue
                    try:
                        if buf.shape[1] < num_terms:
                            widened = torch.zeros(
                                buf.shape[0],
                                num_terms,
                                dtype=buf.dtype,
                                device=buf.device,
                            )
                            widened[:, : buf.shape[1]] = buf
                            setattr(manager, buf_attr, widened)
                    except Exception:  # noqa: BLE001 - best-effort buffer widen
                        pass
            return True

        for list_attr in ("active_terms", "_term_cfgs", "term_cfgs"):
            container = getattr(manager, list_attr, None)
            if isinstance(container, list):
                container.append(term)
                return True
            if isinstance(container, dict):
                container[name] = term
                return True
        return False


def bind_to_env(
    self,
    code: str,
    env,
    *,
    term_names: list[str] | None = None,
    register: bool = True,
) -> "LiveRewardBinding":
    """Validate, wrap, and bind generated reward ``code`` onto a live env (Req 6.1–6.4).

    This is the Task 9.1 entry point. It composes the Task 4.3 :meth:`wrap`
    (validate + sandbox-compile + wrap as ``RewTerm`` callables) with live
    registration on the env's ``RewardManager``. The wrapped terms are bound with
    ``goal_ref=None`` so they read the **live per-env Goal buffer** carried by the
    goal-conditioned env (Req 6.2) rather than a standalone fake Goal.

    Args:
        code: Candidate reward source (already produced by the Qwen_Client).
        env: The live (goal-conditioned) manager-based Isaac Lab env. Must expose
            a Goal buffer (``env.goal``) and a ``reward_manager``.
        term_names: Optional explicit names for the registered terms.
        register: When ``True`` (default), immediately register the terms on the
            env's RewardManager. Pass ``False`` to wrap+bind without registering
            (e.g. to inspect the specs first).

    Returns:
        A :class:`LiveRewardBinding` whose :pyattr:`~LiveRewardBinding.component_log`
        surfaces per-component reward values for logging/feedback (Req 6.3, 6.4).

    Raises:
        ValidationError: If ``code`` fails validation.
        ExecutionError: If the code cannot be compiled, or the env exposes no
            RewardManager / registration surface.
    """
    # Wrap with goal_ref=None: the live env owns its per-env Goal buffer (Req 6.2),
    # so WrappedReward._bind_goal leaves env.goal untouched.
    terms = self.wrap(code, None)
    binding = LiveRewardBinding(terms, executor=self, term_names=term_names)
    if register:
        binding.register(env)
    return binding


# Attach the live-binding entry point to RewardExecutor. Defined at module scope
# (rather than inside the class body above) so it sits alongside the Task 9.1
# binding types it returns; functionally identical to a method on the class.
RewardExecutor.bind_to_env = bind_to_env
