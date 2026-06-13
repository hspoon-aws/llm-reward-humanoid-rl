"""MJX trainer adapter — the simulator-facing seam for the MuJoCo track.

This is the MuJoCo analog of the Isaac project's ``build_isaaclab_trainer`` /
``_IsaacLabTrainer``. It is the ONE place the simulator + RL stack
(``mujoco``, ``mujoco_mjx``, ``mujoco_playground``, ``brax``, ``jax``) is
touched, and — exactly like the Isaac adapter — every heavy import is **lazy
and local to the function body**, never at module load. That preserves the DI
seam that let the Isaac project unit-test the whole loop off-GPU:

  - on the controller/dev host (no JAX/MJX) ``build_mjx_trainer`` raises a
    descriptive ``RuntimeError``;
  - tests inject a fake ``Trainer`` via ``PPORunner(trainer_factory=...)`` and
    never import JAX/MJX at all.

STATUS: SKELETON. The body is intentionally unimplemented (Phase 4 of
PROJECT-PLAN). Signatures mirror the Isaac trainer so the carried-over
``PPORunner`` can drive either adapter unchanged.
"""

from __future__ import annotations

from typing import Any, Protocol


class Trainer(Protocol):
    """Minimal trainer protocol the PPORunner drives (engine-agnostic).

    Mirrors the Isaac project's ``Trainer`` protocol so the same ``PPORunner``
    works against MJX/Brax or a fake. Kept here so the protocol travels with the
    adapter that implements it.
    """

    def run_epoch(self) -> dict[str, Any]:
        """Advance training by one epoch; return that epoch's metrics dict."""
        ...

    def save_checkpoint(self, path: str) -> str:
        """Persist a policy checkpoint; return the checkpoint reference."""
        ...

    @property
    def final_checkpoint(self) -> str | None:
        """Reference to the last checkpoint written, if any."""
        ...


def build_mjx_trainer(
    train_config: Any,
    *,
    reward_fn: Any,
    num_envs: int,
    learning_rate: float,
    goal: Any,
    device: str,
    executor: Any = None,
    warm_start_checkpoint: Any = None,
) -> Trainer:
    """Construct a real MJX + Brax PPO trainer for one run (lazy import).

    MuJoCo analog of ``build_isaaclab_trainer``. Builds the goal-conditioned
    H1 MJX env (``training.env_name``, default ``H1JoystickGaitTracking``,
    reframed for goal-reaching — stock gait/velocity reward zeroed, Goal +
    Goal_Observation injected), wires the JAX ``reward_fn`` produced by the
    Reward_Executor, and drives Brax PPO one iteration per ``run_epoch``.

    Args mirror the Isaac trainer factory; ``reward_fn`` is the JAX-native
    wrapped reward (``jax.Array`` of shape ``(num_envs,)``), not a torch RewTerm.

    Raises:
        RuntimeError: When jax / mujoco_mjx / mujoco_playground / brax are not
            importable in this environment (the controller/dev host). Inject a
            ``Trainer`` via ``PPORunner(trainer_factory=...)`` to run off-GPU.
    """
    try:  # pragma: no cover - exercised only on the B200 GPU host
        import jax  # noqa: F401
        import mujoco_playground  # noqa: F401
        from brax.training.agents.ppo import train as _ppo_train  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dev host path
        raise RuntimeError(
            "JAX / MuJoCo MJX / mujoco_playground / Brax are not importable in "
            "this environment. Provide a trainer via "
            "PPORunner(trainer_factory=...) or an explicit trainer to "
            "train(trainer=...) to run without the GPU runtime."
        ) from exc

    _ensure_trainer_class()
    return _MjxTrainer(  # pragma: no cover - requires the GPU runtime
        train_config,
        reward_fn=reward_fn,
        num_envs=num_envs,
        learning_rate=learning_rate,
        goal=goal,
        device=device,
        executor=executor,
        warm_start_checkpoint=warm_start_checkpoint,
    )


def _make_mjx_trainer_class():  # pragma: no cover - GPU host only
    """Materialize the _MjxTrainer class lazily (keeps module import GPU-free)."""
    import functools
    import json
    import os
    import pickle
    import time

    import jax
    import jax.numpy as jnp
    from brax.training.agents.ppo import train as ppo_train
    from mujoco_playground import wrapper as pg_wrapper

    from ..envs.goal_env import build_goal_env
    from ..exceptions import DivergenceError, OutOfMemoryError

    class _MjxTrainer:
        """Brax-PPO trainer over the goal-conditioned H1 MJX env.

        Mirrors the Isaac ``Trainer`` protocol the carried-over ``PPORunner``
        drives. Brax PPO is a single ``train()`` call (not epoch-by-epoch), so
        the adapter maps the PPO_Runner contract onto it:

          - ``run_epoch`` runs one Brax PPO "generation" worth of timesteps and
            returns that generation's metrics; the PPO_Runner calls it
            ``epochs`` times.
          - a progress callback records metrics and detects divergence
            (non-finite reward/loss -> DivergenceError, Req 14.1).
          - ``save_checkpoint`` pickles the policy params; ``final_checkpoint``
            returns the last path written.
          - a JAX/XLA OOM during training raises the project OutOfMemoryError so
            the PPO_Runner can retry at the fallback env count (Req 15.1).
        """

        def __init__(self, train_config, *, reward_fn, num_envs, learning_rate,
                     goal, device, executor=None, warm_start_checkpoint=None):
            self._cfg = train_config
            self._reward_fn = reward_fn
            self._num_envs = int(num_envs)
            self._lr = float(learning_rate)
            self._warm_start_checkpoint = warm_start_checkpoint
            self._goal = goal
            self._device = device
            self._env_name = getattr(train_config, "env_name", "H1JoystickGaitTracking")
            self._epochs = int(getattr(train_config, "train_epochs", 10))
            self._episode_length = int(getattr(train_config, "episode_length", 1000))
            self._ckpt_dir = getattr(train_config, "checkpoint_dir", "/data/mujoco/runs")
            os.makedirs(self._ckpt_dir, exist_ok=True)

            # Build the goal-conditioned env with the generated reward injected.
            self._env = build_goal_env(
                self._env_name,
                goal,
                reward_fn=reward_fn,
                fall_terminate=bool(getattr(train_config, "fall_terminate", False)),
                fall_height_m=float(getattr(train_config, "fall_height_m", 0.5)),
                standing_reward=float(getattr(train_config, "standing_reward", 0.0)),
            )

            # Fixed network architecture so the eval-time rollout can rebuild the
            # exact same policy from saved params (see save_checkpoint).
            self._policy_hidden = (128, 128, 128, 128)
            self._value_hidden = (256, 256, 256, 256, 256)
            self._obs_size = int(getattr(self._env, "observation_size", 0) or 0)
            self._action_size = int(self._env.action_size)

            self._metrics_log: list[dict] = []
            self._final_ckpt: str | None = None
            self._final_params = None
            self._policy_params = None
            self._epoch_seen = 0

        # ----- Trainer protocol -------------------------------------------- #
        def run_epoch(self) -> dict:
            """Run the full Brax PPO training and return final metrics.

            Brax owns the inner loop, so a single ``run_epoch`` drives the whole
            training run (``num_timesteps`` derived from epochs x episode_length
            x num_envs) and returns the final metrics. The PPO_Runner treats a
            completed run as the iteration's training result."""
            num_timesteps = max(
                self._num_envs * self._episode_length * self._epochs,
                self._num_envs * self._episode_length,
            )

            def _progress(step, metrics):
                row = {"step": int(step)}
                for k, v in metrics.items():
                    try:
                        row[k] = float(v)
                    except Exception:  # noqa: BLE001
                        pass
                self._metrics_log.append(row)
                # Live progress to stdout (-> the run log) so a long iteration's
                # training is observable via `tail` during the unattended run,
                # not only at iteration end. Brax calls progress_fn periodically
                # (every num_timesteps/num_evals steps).
                import sys as _sys
                pct = 100.0 * float(step) / float(max(1, num_timesteps))
                rwd = row.get("eval/episode_reward", row.get("episode_reward", ""))
                print(
                    f"[train-progress] step={int(step)}/{int(num_timesteps)} "
                    f"(~{pct:.0f}%) eval_reward={rwd}",
                    flush=True, file=_sys.stdout,
                )
                # Divergence detection (Req 14.1): any non-finite metric.
                for k, v in row.items():
                    if k == "step":
                        continue
                    if not _finite(v):
                        raise DivergenceError(
                            f"Non-finite training metric '{k}'={v} at step {step}"
                        )

            import functools as _ft

            from brax.training.agents.ppo import networks as ppo_networks

            network_factory = _ft.partial(
                ppo_networks.make_ppo_networks,
                policy_hidden_layer_sizes=self._policy_hidden,
                value_hidden_layer_sizes=self._value_hidden,
            )

            # Brax requires batch_size * num_minibatches % num_envs == 0 and
            # num_envs % device_count == 0. Derive a batch_size/num_minibatches
            # that satisfy the first for any num_envs (batch_size == num_envs,
            # num_minibatches small) so the run never trips the assertion.
            num_minibatches = 8
            batch_size = max(1, self._num_envs // num_minibatches) * num_minibatches
            # Ensure batch_size * num_minibatches is a multiple of num_envs.
            if (batch_size * num_minibatches) % self._num_envs != 0:
                batch_size = self._num_envs
                num_minibatches = 1

            train_fn = functools.partial(
                ppo_train.train,
                environment=self._env,
                num_timesteps=num_timesteps,
                episode_length=self._episode_length,
                num_envs=self._num_envs,
                batch_size=batch_size,
                num_minibatches=num_minibatches,
                num_evals=int(getattr(self._cfg, "num_evals", 10)),
                learning_rate=self._lr,
                normalize_observations=True,
                network_factory=network_factory,
                wrap_env_fn=pg_wrapper.wrap_for_brax_training,
                seed=int(getattr(self._cfg, "seed", 0)),
                progress_fn=_progress,
            )
            # Warm-start: seed PPO from a previously-trained policy's params so
            # this iteration fine-tunes an improving policy instead of relearning
            # from random init. Brax's `restore_params` expects the (normalizer,
            # policy, value) params tuple that `save_params` wrote. Fail-soft: a
            # missing/unreadable/mismatched checkpoint logs and trains from
            # scratch rather than aborting the loop.
            restore_params = self._load_warm_start_params()
            if restore_params is not None:
                train_fn = functools.partial(train_fn, restore_params=restore_params)
            try:
                make_inference_fn, params, metrics = train_fn()
            except DivergenceError:
                raise
            except Exception as exc:  # noqa: BLE001
                if _is_oom(exc):
                    raise OutOfMemoryError(
                        f"JAX/XLA out-of-memory during PPO at num_envs="
                        f"{self._num_envs}: {exc}"
                    ) from exc
                raise

            self._final_params = (make_inference_fn, params)
            self._policy_params = params
            final = {k: _safe_float(v) for k, v in metrics.items()}
            self._epoch_seen = self._epochs
            return final

        def _load_warm_start_params(self):
            """Load the (normalizer, policy, value) params tuple from a prior
            checkpoint for Brax ``restore_params``; ``None`` if unavailable.

            Fail-soft: any problem (no path, missing file, unreadable, wrong
            structure) returns ``None`` so training proceeds from random init
            rather than aborting. A shape mismatch between the saved policy and
            the current network would surface later inside Brax; we keep the
            architecture fixed across iterations (same policy/value hidden sizes
            and obs/action sizes), so a same-env warm-start is compatible."""
            path = self._warm_start_checkpoint
            if not path:
                return None
            import sys as _sys
            try:
                import os as _os

                if not _os.path.exists(path):
                    print(f"[warm-start] checkpoint not found, training from scratch: {path}",
                          flush=True, file=_sys.stdout)
                    return None
                from brax.io import model as _bx_model

                params = _bx_model.load_params(path)
                # Expect a 3-tuple/list (normalizer, policy, value).
                if not isinstance(params, (list, tuple)) or len(params) < 3:
                    print(f"[warm-start] unexpected params structure, training from scratch: {type(params)}",
                          flush=True, file=_sys.stdout)
                    return None
                print(f"[warm-start] resuming from {path}", flush=True, file=_sys.stdout)
                return params
            except Exception as exc:  # noqa: BLE001 - fail-soft to from-scratch
                print(f"[warm-start] load failed ({type(exc).__name__}: {exc}); "
                      f"training from scratch", flush=True, file=_sys.stdout)
                return None

        def save_checkpoint(self, path: str) -> str:
            """Persist the trained policy params + network config (Req 8.4, 8.6).

            Saves the Brax params pytree via ``brax.io.model.save_params`` and,
            alongside it, a small JSON describing the network so the eval-time
            rollout can rebuild ``make_inference_fn`` from the saved params with
            the SAME architecture (the policy is otherwise unreconstructable from
            params alone). The two files share a basename: ``<path>`` (params)
            and ``<path>.netcfg.json`` (architecture)."""
            import json as _json

            from brax.io import model as _bx_model

            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            _bx_model.save_params(path, self._policy_params)
            netcfg = {
                "observation_size": int(self._obs_size),
                "action_size": int(self._action_size),
                "normalize_observations": True,
                "policy_hidden_layer_sizes": list(self._policy_hidden),
                "value_hidden_layer_sizes": list(self._value_hidden),
            }
            with open(path + ".netcfg.json", "w") as fh:
                _json.dump(netcfg, fh)
            self._final_ckpt = path
            return path

        @property
        def final_checkpoint(self) -> str | None:
            return self._final_ckpt

        @property
        def metrics_log(self) -> list[dict]:
            return list(self._metrics_log)

    def _finite(v) -> bool:
        try:
            return bool(jnp.isfinite(jnp.asarray(v)).all())
        except Exception:  # noqa: BLE001
            try:
                import math
                return math.isfinite(float(v))
            except Exception:  # noqa: BLE001
                return True

    def _safe_float(v):
        try:
            return float(v)
        except Exception:  # noqa: BLE001
            return None

    def _np_array(x):
        import numpy as np
        return np.asarray(x)

    def _is_oom(exc) -> bool:
        msg = str(exc).lower()
        return (
            "out of memory" in msg
            or "resource_exhausted" in msg
            or "oom" in msg
            or "failed to allocate" in msg
        )

    return _MjxTrainer


# Public name resolved on first build_mjx_trainer call (keeps import GPU-free).
_MjxTrainer: Any = None


def _ensure_trainer_class():  # pragma: no cover - GPU host only
    global _MjxTrainer
    if _MjxTrainer is None:
        _MjxTrainer = _make_mjx_trainer_class()


# =========================================================================== #
# PPO_Runner — the surface the Orchestrator drives (Req 8, 14, 15, 20)
# =========================================================================== #
import os as _os
from dataclasses import dataclass, field

from ..storage.s3_store import CheckpointRef
from ..exceptions import DivergenceError, OutOfMemoryError


@dataclass
class TrainConfig:
    """Config the PPO_Runner needs, derived from the run Config (Req 8)."""

    env_name: str = "H1JoystickGaitTracking"
    train_epochs: int = 10
    checkpoint_interval: int = 5
    checkpoint_dir: str = "/data/mujoco/runs"
    episode_length: int = 1000
    training_gpus: tuple = (1, 2, 3, 4, 5, 6, 7)
    xla_python_client_mem_fraction: float = 0.6
    oom_fallback_envs: int = 2048
    seed: int = 0
    num_evals: int = 10
    fall_terminate: bool = False
    fall_height_m: float = 0.5
    standing_reward: float = 0.0

    @classmethod
    def from_config(cls, config: Any, *, checkpoint_dir: str | None = None) -> "TrainConfig":
        return cls(
            env_name=getattr(config, "env_name", "H1JoystickGaitTracking"),
            train_epochs=int(getattr(config, "train_epochs", 10)),
            checkpoint_interval=int(getattr(config, "checkpoint_interval", 5)),
            checkpoint_dir=checkpoint_dir or "/data/mujoco/runs",
            episode_length=int(getattr(config, "episode_length", 1000)),
            training_gpus=tuple(getattr(config, "training_gpus", (1, 2, 3, 4, 5, 6, 7))),
            xla_python_client_mem_fraction=float(
                getattr(config, "xla_python_client_mem_fraction", 0.6)
            ),
            oom_fallback_envs=int(getattr(config, "oom_fallback_envs", 2048)),
            seed=int(getattr(config, "seed", 0)),
            num_evals=int(getattr(config, "num_evals", 10)),
            fall_terminate=bool(getattr(config, "fall_terminate", False)),
            fall_height_m=float(getattr(config, "fall_height_m", 0.5)),
            standing_reward=float(getattr(config, "standing_reward", 0.0)),
        )


@dataclass
class TrainResult:
    """Result of one training run (the ``.checkpoint`` the Orchestrator reads)."""

    checkpoint: CheckpointRef | None = None
    metrics_path: str | None = None
    metrics: dict = field(default_factory=dict)
    captures: list = field(default_factory=list)


class PPORunner:
    """Drives a Brax-PPO training run via ``build_mjx_trainer`` (Req 8, 14, 15).

    Matches the Orchestrator's expected surface:
    ``train(reward_terms, epochs=, learning_rate=, goal=, num_envs=,
    capture_hook=) -> TrainResult``. ``reward_terms`` is the list returned by
    ``RewardExecutor.wrap`` (a single ``WrappedReward``); its element is the JAX
    ``compute_reward(data, action, goal_xy, success_radius)`` callable injected
    into the goal env.

    Device pinning (Req 8.2) is applied process-wide via env vars before JAX
    initializes: ``CUDA_VISIBLE_DEVICES`` to the training GPUs (excluding GPU 0)
    and ``XLA_PYTHON_CLIENT_MEM_FRACTION`` so vLLM keeps GPU-0 headroom.
    """

    def __init__(self, train_config: TrainConfig, *, trainer_factory: Any = None) -> None:
        self._cfg = train_config
        self._trainer_factory = trainer_factory or build_mjx_trainer
        self._apply_device_pinning()
        self._n_calls = 0

    def _apply_device_pinning(self) -> None:
        gpus = ",".join(str(g) for g in self._cfg.training_gpus)
        # Only set if not already pinned by the launcher, so an explicit
        # CUDA_VISIBLE_DEVICES on the command line wins.
        _os.environ.setdefault("CUDA_VISIBLE_DEVICES", gpus)
        _os.environ.setdefault(
            "XLA_PYTHON_CLIENT_MEM_FRACTION",
            str(self._cfg.xla_python_client_mem_fraction),
        )
        _os.environ.setdefault("MUJOCO_GL", "egl")

    def train(self, reward_terms: Any, *, epochs: int, learning_rate: float,
              goal: Any, num_envs: int, capture_hook: Any = None,
              warm_start_checkpoint: Any = None) -> TrainResult:
        """Run Brax PPO once with the injected reward; return a TrainResult.

        On JAX/XLA OOM, retries once at the configured fallback env count
        (Req 15.1). A non-finite training metric propagates as DivergenceError
        (Req 14.1) for the Orchestrator's revert-and-reduce-LR path.

        ``warm_start_checkpoint`` (a policy ``.pkl`` path) seeds training from a
        previously-trained policy instead of random init, so reward refinement
        fine-tunes an improving policy across iterations. ``None`` -> from
        scratch (the default / first-iteration behaviour)."""
        reward_fn = reward_terms[0] if isinstance(reward_terms, (list, tuple)) else reward_terms

        def _run(n_envs: int) -> TrainResult:
            trainer = self._trainer_factory(
                self._cfg,
                reward_fn=reward_fn,
                num_envs=n_envs,
                learning_rate=learning_rate,
                goal=goal,
                device="cuda:0",
                warm_start_checkpoint=warm_start_checkpoint,
            )
            metrics = trainer.run_epoch()
            self._n_calls += 1
            ckpt_path = _os.path.join(
                self._cfg.checkpoint_dir, f"iter_{self._n_calls:02d}_policy.pkl"
            )
            trainer.save_checkpoint(ckpt_path)
            metrics_path = _os.path.join(
                self._cfg.checkpoint_dir, f"iter_{self._n_calls:02d}_metrics.json"
            )
            try:
                import json
                with open(metrics_path, "w") as fh:
                    json.dump(metrics, fh, indent=2, default=str)
            except Exception:  # noqa: BLE001
                metrics_path = None
            return TrainResult(
                checkpoint=CheckpointRef(path=ckpt_path),
                metrics_path=metrics_path,
                metrics=metrics,
            )

        try:
            return _run(int(num_envs))
        except OutOfMemoryError:
            # Req 15.1: retry once at the fallback env count.
            return _run(int(self._cfg.oom_fallback_envs))
