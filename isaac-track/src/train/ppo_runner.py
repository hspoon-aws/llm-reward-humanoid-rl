"""PPO_Runner — RSL-RL PPO training on the goal-conditioned H1 environment.

This module owns the training half of the Eureka loop (design.md → Components →
PPO_Runner, Req 8, 14, 15, 20). It trains a PPO policy in Isaac Lab via RSL-RL
against an injected, LLM-generated reward and returns a reference to the final
policy checkpoint.

Scope of THIS file (Task 10.1 — ``train()`` core)
-------------------------------------------------
* :class:`TrainConfig` — the flat training configuration the runner consumes
  (derived from the validated :class:`src.config.Config` via
  :meth:`TrainConfig.from_config`).
* :func:`select_cuda_visible_devices` — pure device-selection logic that maps
  the configured training GPU set to a ``CUDA_VISIBLE_DEVICES`` string and
  **excludes GPU 0**, which is reserved for the Language_Model (Req 8.2).
* :func:`checkpoint_epochs` — pure checkpoint-interval scheduling: the epochs at
  which a checkpoint is written are exactly the multiples of the configured
  interval within the epoch budget (Req 8.4 / Property 25).
* :class:`PPORunner.train` — trains for the configured number of epochs (Req
  8.1) using the configured parallel env count (Req 8.3), restricting devices
  via ``CUDA_VISIBLE_DEVICES`` (Req 8.2), writing checkpoints at the configured
  interval (Req 8.4), and returning the final checkpoint reference (Req 8.6).

Added in Task 10.2 (Req 8.5)
----------------------------
* Training-metrics JSON export — on completion, :meth:`PPORunner.train`
  serializes the accumulated per-epoch metrics (plus the run summary) to a
  deterministic JSON file under the checkpoint directory and records its path on
  :attr:`TrainResult.metrics_path`. The payload builder
  (:func:`training_metrics_payload`) and writer (:func:`export_training_metrics`)
  are exposed so the export shape is unit-testable and round-trippable
  (design.md → Property 27).

Added in Task 10.3 (Req 14.1, 15.1)
-----------------------------------
* Divergence detection — after each epoch, :meth:`PPORunner._run_training_loop`
  inspects the trainer's per-epoch metrics for any non-finite (NaN/inf) loss or
  reward and raises :class:`~src.exceptions.DivergenceError` (Req 14.1 trigger).
  The pure predicates :func:`find_non_finite_metric` / :func:`assert_finite_metrics`
  make that check unit-testable.
* CUDA OOM fallback — :meth:`PPORunner.train` wraps the single
  :meth:`_run_training_loop` seam: if it raises a CUDA out-of-memory condition
  (detected duck-typed via :func:`is_cuda_oom_error`), the run is retried **once**
  at ``config.oom_fallback_envs`` (Req 15.1). A second OOM is re-raised as
  :class:`~src.exceptions.OutOfMemoryError`.

Added in Task 10.4 (Req 20.1, 20.3, 20.5)
-----------------------------------------
* Training-capture hook invocation — :meth:`PPORunner._run_training_loop` invokes
  the optional ``capture_hook`` once per epoch, passing the 1-based training
  epoch. The hook (in production, a :meth:`src.capture.TrainingCaptureProducer.maybe_capture`
  bound to the current iteration index by the Orchestrator) **self-gates** on the
  configured Capture_Interval, so the runner can safely call it every epoch and a
  capture is only produced at the configured cadence (Req 20.1, 20.5). The runner
  never passes the training env count: the capture env-subset bound lives entirely
  in the producer, so invoking the hook can never trigger a full-resolution
  all-env render (Req 20.3, 20.5).

Dependency injection (why RSL-RL / Isaac Lab is not imported at module load)
----------------------------------------------------------------------------
RSL-RL and Isaac Lab are only importable inside the Isaac Sim runtime, never on
the controller/dev host. So the heavy trainer is an **injected dependency**: the
runner drives a small :class:`Trainer` protocol (run an epoch, save a
checkpoint) and never imports Isaac Lab itself. A real trainer is built lazily
by :func:`build_isaaclab_trainer` only when training actually runs and no
trainer factory was injected. This keeps the device-selection and
checkpoint-scheduling logic fully unit-testable with an in-memory fake trainer,
with no GPU and no Isaac Lab present.
"""

from __future__ import annotations

import json
import math
import numbers
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Protocol, Sequence, runtime_checkable

from src.exceptions import DivergenceError, OutOfMemoryError
from src.storage.s3_store import CheckpointRef

__all__ = [
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "TrainerFactory",
    "CaptureHook",
    "PPORunner",
    "select_cuda_visible_devices",
    "checkpoint_epochs",
    "training_metrics_payload",
    "export_training_metrics",
    "build_isaaclab_trainer",
    "find_non_finite_metric",
    "assert_finite_metrics",
    "is_cuda_oom_error",
    "LLM_GPU",
    "METRICS_FILENAME",
]

# Filename written under ``TrainConfig.checkpoint_dir`` for the exported
# training-metrics JSON (Req 8.5). Kept alongside the checkpoints so a run's
# artifacts live together and the Orchestrator can persist the whole directory.
METRICS_FILENAME = "training_metrics.json"

# GPU 0 is reserved for the Language_Model (vLLM) and is never used for training
# (design.md → GPU Allocation; Req 8.2). The Config loader already rejects a
# training set containing it; the runner re-enforces it at device-selection time
# as a defensive guardrail.
LLM_GPU = 0


def _endpoint_is_loopback(endpoint: str) -> bool:
    """True if the LLM endpoint targets this host (loopback) vs cross-host.

    Mirrors ``src.config._endpoint_is_loopback`` (kept local so this module
    stays import-light). Loopback: localhost / 127.0.0.0/8 / ::1. Unparseable
    endpoints default to loopback (strict: keep the GPU-0 reservation).
    """
    from urllib.parse import urlsplit

    candidate = endpoint if "://" in endpoint else f"//{endpoint}"
    try:
        host = (urlsplit(candidate).hostname or "").lower()
    except (ValueError, TypeError):
        return True
    if not host:
        return True
    return host in {"localhost", "::1"} or host.startswith("127.")


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class TrainConfig:
    """Flat training configuration consumed by :class:`PPORunner`.

    Mirrors the training-relevant slice of the validated
    :class:`src.config.Config` (build one with :meth:`from_config`). Kept as a
    small, torch-free dataclass so the runner's pure logic stays unit-testable
    on the controller host.

    Attributes:
        training_gpus: The GPU indices to train on. Must exclude :data:`LLM_GPU`
            (Req 8.2); the Config loader enforces this and so does
            :func:`select_cuda_visible_devices`.
        num_envs: The number of parallel Environment instances (Req 8.3).
        oom_fallback_envs: Reduced env count used by the OOM retry (Task 10.3,
            Req 15.1). Unused by the Task 10.1 core but carried for that task.
        train_epochs: Default epoch budget when ``train(epochs=...)`` is omitted
            (Req 8.1).
        checkpoint_interval: Write a checkpoint every this-many epochs
            (Req 8.4).
        learning_rate: Default PPO learning rate when ``train`` is called
            without an explicit value.
        checkpoint_dir: Local directory under which checkpoints are written. The
            returned :class:`CheckpointRef` points inside it (Req 8.6).
        env_id: The Isaac Lab task id to train (goal-conditioned H1 flat env).
        llm_gpu: The reserved Language_Model GPU to exclude (defaults to 0).
        seed: Optional RNG seed forwarded to the real trainer.
        extra: Free-form passthrough for the real trainer adapter (RSL-RL
            hyperparameters, logging options) — ignored by the fake trainer.
    """

    training_gpus: list[int]
    num_envs: int
    oom_fallback_envs: int = 0
    train_epochs: int = 1
    num_steps_per_env: int = 24
    checkpoint_interval: int = 1
    learning_rate: float = 1.0e-3
    checkpoint_dir: str = "runs/checkpoints"
    env_id: str = "Isaac-Velocity-Flat-H1-v0"
    llm_gpu: int = LLM_GPU
    seed: int | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_config(cls, config: Any, *, checkpoint_dir: str | None = None,
                    env_id: str | None = None) -> "TrainConfig":
        """Build a :class:`TrainConfig` from a validated :class:`src.config.Config`.

        Reads the training-relevant fields off ``config`` (``training_gpus``,
        ``num_envs``, ``oom_fallback_envs``, ``train_epochs``,
        ``checkpoint_interval``). The env id and checkpoint directory can be
        overridden; otherwise sensible defaults are used.
        """
        kwargs: dict[str, Any] = dict(
            training_gpus=list(getattr(config, "training_gpus")),
            num_envs=int(getattr(config, "num_envs")),
            oom_fallback_envs=int(getattr(config, "oom_fallback_envs", 0)),
            train_epochs=int(getattr(config, "train_epochs", 1)),
            num_steps_per_env=int(getattr(config, "num_steps_per_env", 24)),
            checkpoint_interval=int(getattr(config, "checkpoint_interval", 1)),
            learning_rate=float(getattr(config, "learning_rate", 1.0e-3)),
        )
        # Reserve no GPU for the LLM when it is cross-host (non-loopback
        # endpoint): on a single-GPU Isaac host (g6/L4) the only device is 0 and
        # the vLLM lives on a separate box, so GPU 0 must be trainable. We signal
        # "no reserved GPU" to select_cuda_visible_devices via llm_gpu=-1. For a
        # co-resident (loopback) endpoint the default reservation (GPU 0) holds.
        # Reserve no GPU for the LLM when it is cross-host (non-loopback
        # endpoint) or a managed service (bedrock): on a single-GPU Isaac host
        # (g6/L4) the only device is 0 and the LLM lives off-box, so GPU 0 must
        # be trainable. We signal "no reserved GPU" to select_cuda_visible_devices
        # via llm_gpu=-1. For a co-resident (loopback vLLM) endpoint the default
        # reservation (GPU 0) holds.
        endpoint = str(getattr(config, "llm_endpoint", "") or "")
        provider = str(getattr(config, "llm_provider", "vllm") or "vllm").lower()
        if provider == "bedrock" or (endpoint and not _endpoint_is_loopback(endpoint)):
            kwargs["llm_gpu"] = -1
        # The Isaac Lab task id is sourced from Config (training.task) when present
        # so the registered env name is operator-configurable (Req 8.1); an
        # explicit ``env_id=`` override still wins for tests/tools.
        config_env_id = getattr(config, "env_id", None)
        if config_env_id:
            kwargs["env_id"] = str(config_env_id)
        if checkpoint_dir is not None:
            kwargs["checkpoint_dir"] = checkpoint_dir
        if env_id is not None:
            kwargs["env_id"] = env_id
        return cls(**kwargs)


@dataclass
class TrainResult:
    """Outcome of a training run (design.md → PPO_Runner ``train`` contract).

    Attributes:
        checkpoint: Reference to the **final** policy checkpoint (Req 8.6).
        checkpoint_paths: Every checkpoint written during the run, in epoch
            order — the interval checkpoints (Req 8.4) plus the final one.
        epochs_completed: Number of epochs actually trained (Req 8.1).
        num_envs_used: The parallel env count training actually ran with (Req
            8.3; may be the OOM fallback once Task 10.3 lands).
        device: The ``CUDA_VISIBLE_DEVICES`` string training was restricted to
            (Req 8.2).
        epoch_metrics: Per-epoch metrics dicts returned by the trainer; the
            source for the JSON export (Req 8.5).
        metrics_path: Local path of the exported training-metrics JSON written
            on completion (Req 8.5). ``None`` only when export is disabled.
    """

    checkpoint: CheckpointRef
    checkpoint_paths: list[str] = field(default_factory=list)
    epochs_completed: int = 0
    num_envs_used: int = 0
    device: str = ""
    epoch_metrics: list[dict[str, Any]] = field(default_factory=list)
    metrics_path: str | None = None
    # Live train+eval reuse (Isaac Sim allows one SimulationApp per process):
    # the trainer's already-live rsl_rl env and inference policy, surfaced so the
    # Evaluator can roll out against them instead of building a SECOND env (which
    # would relaunch a second SimulationApp and stall). ``None`` for the fake
    # trainer / when the backend exposes no live handles.
    live_env: Any = None
    live_policy: Any = None


# --------------------------------------------------------------------------- #
# Injected trainer contract
# --------------------------------------------------------------------------- #
@runtime_checkable
class Trainer(Protocol):
    """The narrow training backend the :class:`PPORunner` drives.

    A real implementation adapts RSL-RL's ``OnPolicyRunner`` over the Isaac Lab
    H1 env (see :func:`build_isaaclab_trainer`); tests inject an in-memory fake.
    The runner owns the epoch loop and checkpoint *scheduling* so that logic is
    testable without a real trainer — the backend only has to advance one epoch
    and persist a checkpoint on demand.
    """

    def run_epoch(self, epoch: int) -> dict[str, Any]:
        """Advance training by a single epoch.

        Args:
            epoch: 1-based epoch index currently being trained.

        Returns:
            A metrics dict for the epoch (e.g. mean reward, losses). May be
            empty. The runner accumulates these for the Task 10.2 JSON export
            and for the Task 10.3 divergence check.
        """
        ...

    def save_checkpoint(self, path: str, epoch: int) -> str:
        """Persist the current policy to ``path`` and return the written path.

        Args:
            path: Destination local path the runner computed for this epoch.
            epoch: The epoch the checkpoint corresponds to.

        Returns:
            The path actually written (normally ``path``).
        """
        ...


# A factory builds a configured :class:`Trainer` for one training run, given the
# reward terms, env count, learning rate, goal, and the selected device string.
TrainerFactory = Callable[..., Trainer]

# A capture hook is invoked during training to emit a Training_Capture (Task
# 10.4, Req 20). Defined here so ``train`` can accept it now; the invocation
# cadence is implemented in Task 10.4.
CaptureHook = Callable[..., Any]


# --------------------------------------------------------------------------- #
# Pure logic: device selection (Req 8.2) and checkpoint scheduling (Req 8.4)
# --------------------------------------------------------------------------- #
def select_cuda_visible_devices(
    training_gpus: Sequence[int], *, llm_gpu: int = LLM_GPU
) -> str:
    """Map the configured training GPU set to a ``CUDA_VISIBLE_DEVICES`` string.

    The reserved Language_Model GPU (``llm_gpu``, default :data:`LLM_GPU` = 0) is
    excluded from the result. Pass ``llm_gpu=-1`` to reserve **no** GPU (the
    cross-host LLM case: a single-GPU Isaac box reaching a separate vLLM over the
    VPC, where GPU 0 is free to train on). The Config loader performs the
    matching cross-field check — it only enforces the GPU-0 reservation for
    loopback LLM endpoints. Order is preserved and duplicates are dropped.

    Args:
        training_gpus: The configured training GPU indices.
        llm_gpu: The GPU reserved for the Language_Model to exclude (default 0).

    Returns:
        A comma-joined device string, e.g. ``"1,2,3"``.

    Raises:
        ValueError: If ``training_gpus`` is empty, contains a negative index, or
            reduces to an empty set once the reserved GPU is excluded (which
            would mean there are no GPUs left to train on).
    """
    if not training_gpus:
        raise ValueError("training_gpus must contain at least one GPU index")

    selected: list[int] = []
    seen: set[int] = set()
    for gpu in training_gpus:
        if int(gpu) < 0:
            raise ValueError(f"invalid GPU index {gpu!r}: must be non-negative")
        gpu = int(gpu)
        # llm_gpu < 0 means "no reserved GPU" (cross-host LLM): exclude nothing.
        if llm_gpu >= 0 and gpu == llm_gpu:
            # Defensive: the Config loader already rejects this for loopback
            # LLM endpoints (Req 18.5); the runner re-enforces the boundary.
            continue
        if gpu in seen:
            continue
        seen.add(gpu)
        selected.append(gpu)

    if not selected:
        raise ValueError(
            f"no training GPUs remain after excluding the reserved "
            f"language-model GPU {llm_gpu}: {list(training_gpus)}"
        )
    return ",".join(str(gpu) for gpu in selected)


def checkpoint_epochs(epochs: int, interval: int) -> list[int]:
    """Return the epochs at which a checkpoint is written (Req 8.4 / Property 25).

    A checkpoint is written exactly at every epoch that is a positive multiple
    of ``interval`` within the inclusive budget ``1..epochs``. The final epoch
    always yields a returned checkpoint regardless of alignment, but that final
    checkpoint is handled by :meth:`PPORunner.train` (Req 8.6) and is NOT part of
    this interval schedule — keeping this helper a clean "multiples of k within
    E" function.

    Args:
        epochs: Total number of epochs to be trained (``E``). Must be >= 0.
        interval: Checkpoint interval in epochs (``k``). Must be >= 1.

    Returns:
        Sorted list of 1-based epoch indices that are multiples of ``interval``
        and ``<= epochs``.

    Raises:
        ValueError: If ``interval`` < 1 or ``epochs`` < 0.
    """
    if interval < 1:
        raise ValueError(f"checkpoint interval must be >= 1, got {interval}")
    if epochs < 0:
        raise ValueError(f"epochs must be >= 0, got {epochs}")
    return [e for e in range(interval, epochs + 1, interval)]


# --------------------------------------------------------------------------- #
# Divergence detection (Req 14.1) and CUDA-OOM classification (Req 15.1)
# --------------------------------------------------------------------------- #
def _iter_numeric_leaves(value: Any) -> Any:
    """Yield the scalar numeric leaves of ``value`` for finiteness checking.

    Walks nested mappings and sequences (the shape trainers emit per-epoch:
    flat ``{"loss": ...}`` dicts, but also grouped metrics like
    ``{"losses": {"value": ...}}`` or ``{"rewards": [...]}``). Strings and bytes
    are treated as opaque (never iterated into characters). Booleans are skipped
    (``bool`` is an ``int`` subclass but is always finite and not a metric).
    """
    if isinstance(value, bool):
        return
    if isinstance(value, numbers.Real):
        yield value
        return
    if isinstance(value, (str, bytes)):
        return
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _iter_numeric_leaves(item)
        return
    if isinstance(value, Sequence):
        for item in value:
            yield from _iter_numeric_leaves(item)
        return
    # Unknown / non-numeric leaf (e.g. None, objects): nothing to check.
    return


def find_non_finite_metric(metrics: Mapping[str, Any]) -> str | None:
    """Return the name of the first metric holding a non-finite value, else ``None``.

    A value is non-finite if any of its scalar numeric leaves is NaN or infinity
    (Req 14.1). Nested mappings/sequences are searched so grouped trainer metrics
    are covered. The returned key is the **top-level** metric name, which the
    caller embeds in the :class:`DivergenceError` message.

    Args:
        metrics: A single epoch's metrics dict (as returned by ``Trainer.run_epoch``).

    Returns:
        The offending top-level metric key, or ``None`` if every value is finite.
    """
    for key, value in metrics.items():
        for leaf in _iter_numeric_leaves(value):
            if not math.isfinite(float(leaf)):
                return key
    return None


def assert_finite_metrics(metrics: Mapping[str, Any], *, epoch: int) -> None:
    """Raise :class:`DivergenceError` if ``metrics`` contains a non-finite value.

    This is the divergence trigger the PPO_Runner applies after every epoch
    (Req 14.1): a NaN/inf loss or reward means training has diverged and the
    Orchestrator must revert to the last good reward and reduce the learning
    rate. A no-op when all metrics are finite.

    Args:
        metrics: The epoch's metrics dict.
        epoch: 1-based epoch index, included in the error message for context.

    Raises:
        DivergenceError: If any metric value is non-finite (NaN or infinity).
    """
    offending = find_non_finite_metric(metrics)
    if offending is not None:
        raise DivergenceError(
            f"non-finite training metric {offending!r} at epoch {epoch}: "
            f"{metrics[offending]!r}"
        )


def is_cuda_oom_error(exc: BaseException) -> bool:
    """Return ``True`` if ``exc`` represents a CUDA out-of-memory condition.

    Detected **duck-typed** rather than by importing torch, so the OOM path is
    testable with a fake trainer on a host without CUDA (design.md → testing
    notes; Req 15.1):

    * the project :class:`~src.exceptions.OutOfMemoryError`;
    * ``torch.cuda.OutOfMemoryError`` (matched by class name, no torch import);
    * any :class:`RuntimeError` whose message indicates an OOM (``"out of
      memory"`` / ``"CUDA out of memory"``), which is what real PyTorch raises.

    Args:
        exc: The caught exception.

    Returns:
        Whether the exception should trigger the env-count fallback retry.
    """
    if isinstance(exc, OutOfMemoryError):
        return True
    # Match torch's CUDA OOM by class name without importing torch.
    if type(exc).__name__ == "OutOfMemoryError":
        return True
    message = str(exc).lower()
    if isinstance(exc, (RuntimeError, MemoryError)) and "out of memory" in message:
        return True
    return False


# --------------------------------------------------------------------------- #
# Training-metrics JSON export (Req 8.5)
# --------------------------------------------------------------------------- #
def training_metrics_payload(
    epoch_metrics: Sequence[dict[str, Any]],
    *,
    epochs_completed: int,
    num_envs_used: int,
    device: str,
    checkpoint_paths: Sequence[str],
    final_checkpoint: str,
) -> dict[str, Any]:
    """Build the JSON-serializable training-metrics document (Req 8.5).

    Combines a small run summary with the per-epoch metrics the trainer emitted
    so the exported artifact is self-describing for the Orchestrator and the
    Blog. The shape is plain JSON types only (no torch, no dataclasses) so it
    round-trips exactly through :func:`json.dumps`/:func:`json.loads`
    (design.md → Property 27).

    Args:
        epoch_metrics: Per-epoch metrics dicts, in epoch order.
        epochs_completed: Number of epochs actually trained (Req 8.1).
        num_envs_used: Parallel env count training ran with (Req 8.3).
        device: The ``CUDA_VISIBLE_DEVICES`` string used (Req 8.2).
        checkpoint_paths: Every checkpoint written, in epoch order (Req 8.4).
        final_checkpoint: Path of the final checkpoint (Req 8.6).

    Returns:
        A JSON-ready ``dict`` with a ``summary`` block and an ``epochs`` list of
        ``{"epoch": <1-based index>, "metrics": {...}}`` entries.
    """
    return {
        "summary": {
            "epochs_completed": int(epochs_completed),
            "num_envs_used": int(num_envs_used),
            "device": str(device),
            "checkpoint_paths": [str(p) for p in checkpoint_paths],
            "final_checkpoint": str(final_checkpoint),
        },
        "epochs": [
            {"epoch": index, "metrics": dict(metrics)}
            for index, metrics in enumerate(epoch_metrics, start=1)
        ],
    }


def export_training_metrics(payload: Mapping[str, Any], path: str) -> str:
    """Serialize ``payload`` to ``path`` as JSON and return the written path.

    The parent directory is created if missing (the checkpoint directory may not
    exist yet on the first run). ``sort_keys`` is enabled for deterministic,
    diffable output, matching the Evaluator's ``EvalMetrics.to_json`` convention.

    Args:
        payload: The metrics document (see :func:`training_metrics_payload`).
        path: Destination local file path.

    Returns:
        The path actually written (normally ``path``).
    """
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, sort_keys=True, indent=2)
    return path


# --------------------------------------------------------------------------- #
# Lazy real-trainer factory (Isaac Lab / RSL-RL) — built only when training runs
# --------------------------------------------------------------------------- #
# Process-level handle to the most recently built live trainer. A single process
# may build more than one trainer in sequence — the OOM-fallback retry and the
# divergence-recovery retry both reconstruct one — but Isaac allows only one live
# env/SimulationContext at a time. We track the latest so the next build can
# CLOSE it (release the RTX scene + vec-env) before clearing the context and
# building afresh. Without the close, the prior env's scene lingers and the new
# gym.make either collides ("Simulation context already exists") or hangs during
# the second scene setup.
_LIVE_TRAINER: Any = None


def _close_prior_trainer() -> None:  # pragma: no cover - requires Isaac runtime
    """Close the previously-built live trainer's env, if any (best-effort)."""
    global _LIVE_TRAINER  # noqa: PLW0603 - process singleton
    prior = _LIVE_TRAINER
    _LIVE_TRAINER = None
    if prior is None:
        return
    closer = getattr(prior, "close", None)
    if callable(closer):
        try:
            closer()
        except Exception as exc:  # noqa: BLE001 - teardown is best-effort
            print(f"[trainer] WARNING: could not close prior trainer env: {exc}")


def _clear_existing_sim_context() -> None:  # pragma: no cover - requires Isaac runtime
    """Tear down any live Isaac ``SimulationContext`` so a new env can be built.

    Isaac Sim permits exactly one ``SimulationContext`` per process, and
    ``ManagerBasedEnv.__init__`` raises ``RuntimeError("Simulation context
    already exists. Cannot create a new one.")`` when one is still live at
    ``gym.make`` time. A single training call can legitimately need a SECOND env
    in the same process: the OOM-fallback retry rebuilds the trainer at
    ``oom_fallback_envs`` and the divergence-recovery retry rebuilds it at a
    reduced LR. In both cases the first attempt's env + context linger and the
    rebuild collides or hangs. We first CLOSE the prior trainer's env (releasing
    its RTX scene + vec-env), then clear the ``SimulationContext`` singleton, so
    the rebuild starts from a clean slate.

    Best-effort and import-guarded: a no-op when no context exists (the common
    first-build case) or when Isaac is not importable (controller host / tests).
    The process-level ``SimulationApp`` itself is NOT touched — only the prior
    env and the per-env ``SimulationContext`` singleton are reset, which are the
    objects the env recreates.
    """
    # (a) Close the prior trainer's env first — clearing the context alone leaves
    # the old RTX scene/physics views live, which is what makes the SECOND
    # gym.make hang during scene setup.
    _close_prior_trainer()
    # (b) Reset the SimulationContext singleton so ManagerBasedEnv can recreate it.
    try:
        from isaacsim.core.api.simulation_context import (  # noqa: PLC0415
            SimulationContext,
        )
    except Exception:  # noqa: BLE001 - older layout or not in the Isaac runtime
        try:
            from omni.isaac.core.simulation_context import (  # noqa: PLC0415
                SimulationContext,
            )
        except Exception:  # noqa: BLE001
            return
    try:
        if SimulationContext.instance() is not None:
            SimulationContext.clear_instance()
    except Exception as exc:  # noqa: BLE001 - never let teardown abort a run
        print(f"[trainer] WARNING: could not clear SimulationContext: {exc}")


# --------------------------------------------------------------------------- #
def _train_device_from_visible(device: str) -> str:
    """Map a ``CUDA_VISIBLE_DEVICES`` string to the trainer's torch device.

    :meth:`PPORunner.train` restricts the process to the configured GPUs by
    setting ``CUDA_VISIBLE_DEVICES`` (Req 8.2), which **remaps** those physical
    GPUs to a fresh ``0..N-1`` ordinal space. So the in-process trainer always
    targets ``cuda:0`` (the first visible device) — never the original physical
    index. When the selection is empty, training falls back to CPU (only useful
    for a smoke test; real H1 training needs a GPU).

    Multi-GPU RSL-RL uses ``torchrun`` distributed launch where each rank owns
    one visible device, so a single-process runner targeting ``cuda:0`` is the
    correct per-process device in both the single- and multi-GPU cases.
    """
    return "cuda:0" if device else "cpu"


class _IsaacLabTrainer:  # pragma: no cover - requires the Isaac Sim runtime
    """RSL-RL ``OnPolicyRunner`` adapter over the goal-conditioned H1 env.

    Implements the :class:`Trainer` Protocol (:meth:`run_epoch` /
    :meth:`save_checkpoint`) that :class:`PPORunner` drives. The whole class body
    is ``# pragma: no cover`` because it can only execute inside the Isaac Sim
    runtime (``isaaclab`` + ``rsl_rl`` present, a GPU available); the CPU-host
    unit tests drive a fake :class:`Trainer` instead and only assert that
    :func:`build_isaaclab_trainer` raises a clear :class:`RuntimeError` here.

    Construction (in :meth:`__init__`, all imports lazy/local) follows the
    standard Isaac Lab RSL-RL training-script order:

    1. Launch the simulator headless via ``isaaclab.app.AppLauncher`` **before**
       importing any ``isaaclab_tasks`` module (the task registry only imports
       once the SimulationApp exists).
    2. Parse the env cfg for ``train_config.env_id`` at ``num_envs`` via
       ``isaaclab_tasks.utils.parse_env_cfg``.
    3. :func:`attach_goal_conditioning` to append the Goal_Observation term and
       zero the stock velocity-tracking rewards (Req 6.2, 8.1).
    4. ``gym.make`` the task, then broadcast the per-env :class:`GoalBuffer`
       (:func:`make_goal_buffer`) and publish it on the live env at
       :data:`GOAL_BUFFER_ENV_ATTR` (Req 6.2).
    5. Register the already-wrapped ``reward_terms`` on the env's
       ``RewardManager`` via :class:`LiveRewardBinding` (Req 6.1).
    6. Wrap the env for RSL-RL and build an ``OnPolicyRunner`` on the selected
       device with ``learning_rate`` from the factory arg.

    Version assumptions a GPU shakeout must confirm (the code degrades
    gracefully via ``getattr``/``try`` where it can):

    * ``isaaclab.app.AppLauncher`` (Isaac Lab >= 1.0; pre-rename was
      ``omni.isaac.lab.app``). The RSL-RL vec-env wrapper is tried from
      ``isaaclab_rl.rsl_rl`` then the legacy ``omni.isaac.lab_rl.rsl_rl``.
    * ``isaaclab_tasks.utils.parse_env_cfg(task, device=, num_envs=)`` keyword
      shape (older builds used ``use_gpu=``); both call shapes are attempted.
    * ``rsl_rl.runners.OnPolicyRunner(env, train_cfg: dict, log_dir, device)``
      and ``runner.learn(num_learning_iterations=, init_at_random_ep_len=)`` /
      ``runner.save(path)`` — stable across rsl_rl 1.x–2.x, but the train-cfg
      dict schema (policy/algorithm sub-dicts) should be confirmed against the
      installed rsl_rl version.
    """

    def __init__(
        self,
        train_config: TrainConfig,
        *,
        reward_terms: Any,
        num_envs: int,
        learning_rate: float,
        goal: Any,
        device: str,
        executor: Any = None,
        init_checkpoint: str | None = None,
    ) -> None:
        # (1) Obtain the PROCESS-LEVEL SimulationApp (launched once, reused by
        # every iteration + eval + recorder). Isaac Sim allows one SimulationApp
        # per process; building a fresh trainer each iteration must NOT relaunch
        # it. enable_cameras is required for any RTX camera rendering (Req 10
        # demo video / Req 20 capture); default on, overridable via
        # HUMANOID_ENABLE_CAMERAS=0 for a no-camera perf run.
        import os as _os  # noqa: PLC0415

        from src.sim_app import get_sim_app  # noqa: PLC0415

        _enable_cameras = _os.environ.get("HUMANOID_ENABLE_CAMERAS", "1") != "0"
        self._sim_app = get_sim_app(enable_cameras=_enable_cameras)
        # This trainer does not own the app; it must not close it (see close()).
        self._owns_sim_app = False

        # (1b) Tear down any LEFTOVER SimulationContext before building a new env.
        # Isaac Sim allows one SimulationContext per process and ManagerBasedEnv
        # raises "Simulation context already exists. Cannot create a new one." if
        # one is already live when gym.make runs. A single iteration can legitimately
        # build a second env in-process — the OOM-fallback retry (at oom_fallback_envs)
        # and the divergence-recovery retry (at reduced LR) both reconstruct the
        # trainer — and the prior attempt's context lingers. Clearing the singleton
        # here makes those in-process rebuilds clean instead of colliding. No-op on
        # the very first build (instance() is None) and best-effort across Isaac
        # versions (the API is stable as SimulationContext.clear_instance()).
        _clear_existing_sim_context()

        # Imports that are only valid once the SimulationApp exists.
        import gymnasium as gym  # noqa: PLC0415
        import isaaclab_tasks  # noqa: F401, PLC0415 - registers the task ids
        from isaaclab_tasks.utils import parse_env_cfg  # noqa: PLC0415

        from src.envs.goal_env import (  # noqa: PLC0415
            GOAL_BUFFER_ENV_ATTR,
            attach_goal_conditioning,
            make_goal_buffer,
        )
        from src.rewards.reward_executor import (  # noqa: PLC0415
            LiveRewardBinding,
            RewardExecutor,
        )

        self._train_config = train_config
        self._num_envs = int(num_envs)
        self._device = _train_device_from_visible(device)
        self._epoch = 0

        # (2) Parse the env cfg at the requested env count, tolerant of the
        # parse_env_cfg signature drift across Isaac Lab versions.
        try:
            env_cfg = parse_env_cfg(
                train_config.env_id, device=self._device, num_envs=self._num_envs
            )
        except TypeError:
            env_cfg = parse_env_cfg(
                train_config.env_id, use_gpu=True, num_envs=self._num_envs
            )

        # (3) Reframe as goal-reaching: append Goal_Observation, zero stock
        # velocity-tracking rewards (Req 6.2, 8.1).
        attach_goal_conditioning(env_cfg, goal, self._num_envs)

        # (4) Build the live env and publish the per-env Goal buffer (Req 6.2).
        self._env = gym.make(train_config.env_id, cfg=env_cfg, render_mode=None)
        goal_buffer = make_goal_buffer(goal, self._num_envs, device=self._device)
        setattr(self._env.unwrapped, GOAL_BUFFER_ENV_ATTR, goal_buffer)

        # (5) Register the already-wrapped reward terms on the RewardManager
        # (Req 6.1). The orchestrator passes the output of executor.wrap(...),
        # i.e. a list of WrappedReward callables, in as ``reward_terms``.
        self._executor = executor if executor is not None else RewardExecutor()
        terms = list(reward_terms) if reward_terms is not None else []
        self._reward_binding = LiveRewardBinding(terms, executor=self._executor)
        self._reward_binding.register(self._env)

        # (6) Wrap for RSL-RL and build the OnPolicyRunner on the selected device.
        RslRlVecEnvWrapper = self._import_rsl_rl_wrapper()
        self._rsl_env = RslRlVecEnvWrapper(self._env)

        from rsl_rl.runners import OnPolicyRunner  # noqa: PLC0415

        train_cfg = self._build_train_cfg(learning_rate)
        self._runner = OnPolicyRunner(
            self._rsl_env,
            train_cfg,
            log_dir=train_config.checkpoint_dir,
            device=self._device,
        )

        # (6b) WARM-START (continuous learning): load the previous iteration's
        # trained weights so each Eureka iteration continues learning from where
        # the last one left off, rather than restarting PPO from scratch. RL is
        # continuous — on a fixed goal with successively-refined rewards, carrying
        # the actor/critic forward compounds progress across iterations instead
        # of re-climbing the ~15M-step curve each time. ``OnPolicyRunner.load``
        # restores the actor-critic (and optimizer/normalizer when present); the
        # critic re-calibrates quickly to the new reward's scale. Best-effort: a
        # missing/incompatible checkpoint logs a warning and proceeds cold rather
        # than aborting the run.
        if init_checkpoint:
            try:
                import os as _os2  # noqa: PLC0415

                if _os2.path.exists(init_checkpoint):
                    self._runner.load(init_checkpoint)
                    print(f"[trainer] warm-started from checkpoint: {init_checkpoint}")
                else:
                    print(
                        f"[trainer] WARNING: warm-start checkpoint not found, "
                        f"training cold: {init_checkpoint}"
                    )
            except Exception as exc:  # noqa: BLE001 - warm-start is best-effort
                print(
                    f"[trainer] WARNING: could not warm-start from "
                    f"{init_checkpoint} ({exc}); training cold."
                )

        # Register as the process's live trainer so a subsequent in-process
        # rebuild (OOM-fallback / divergence-recovery retry) closes THIS env
        # before building the next one (see _clear_existing_sim_context).
        global _LIVE_TRAINER  # noqa: PLW0603 - process singleton
        _LIVE_TRAINER = self

    # ------------------------------------------------------------------ #
    # Trainer Protocol
    # ------------------------------------------------------------------ #
    def run_epoch(self, epoch: int) -> dict[str, Any]:
        """Advance training by a single RSL-RL learning iteration (Req 8.1).

        Drives ``OnPolicyRunner.learn`` for exactly one iteration. The runner's
        ``current_learning_iteration`` accumulates across calls, so invoking this
        once per epoch yields a continuous training run. Metric extraction is
        best-effort and version-tolerant: whatever mean reward / losses can be
        read off the runner are returned, otherwise an empty dict (which the
        PPO_Runner treats as "no divergence signal this epoch").
        """
        self._epoch = int(epoch)
        init_at_random_ep_len = epoch == 1
        try:
            self._runner.learn(
                num_learning_iterations=1,
                init_at_random_ep_len=init_at_random_ep_len,
            )
        except TypeError:
            # Older rsl_rl signatures take the count positionally / lack the flag.
            self._runner.learn(1)
        return self._collect_metrics()

    def save_checkpoint(self, path: str, epoch: int) -> str:
        """Persist the current policy to ``path`` and return it (Req 8.4, 8.6)."""
        parent = os.path.dirname(path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self._runner.save(path)
        return path

    def close(self) -> None:
        """Tear down the per-iteration env (best-effort); keep the SimulationApp.

        The ``SimulationApp`` is a PROCESS-LEVEL singleton owned by
        :mod:`src.sim_app` and shared across every loop iteration, eval, and the
        recorder. A per-iteration trainer must release its env (so the next
        iteration's env builds cleanly) but must NOT close the app — closing it
        would make every subsequent iteration's sim work fail, since Isaac Sim
        cannot relaunch a SimulationApp in the same process. The process app is
        torn down once at process exit, not here.
        """
        env = getattr(self, "_env", None)
        if env is not None:
            try:
                env.close()
            except Exception:  # noqa: BLE001 - shutdown is best-effort
                pass
        # Deregister from the process live-trainer slot if we are the current one
        # (avoids a double-close from _close_prior_trainer later).
        global _LIVE_TRAINER  # noqa: PLW0603 - process singleton
        if _LIVE_TRAINER is self:
            _LIVE_TRAINER = None
        # Intentionally do NOT close self._sim_app: it is the shared process app.

    def live_handles(self) -> tuple[Any, Any]:
        """Return the trainer's live ``(rsl_env, inference_policy)`` for eval reuse.

        Isaac Sim allows only one ``SimulationApp`` per process, so the Evaluator
        must roll out against the trainer's already-live env rather than build a
        second one. Returns the rsl_rl-wrapped vec env and the runner's inference
        policy (duck-typed across rsl_rl versions); either may be ``None`` if the
        backend has not built them. Best-effort: never raises.
        """
        rsl_env = getattr(self, "_rsl_env", None)
        policy = None
        runner = getattr(self, "_runner", None)
        if runner is not None:
            getter = getattr(runner, "get_inference_policy", None)
            if callable(getter):
                try:  # pragma: no cover - requires the live rsl_rl runner
                    policy = getter(self._device)
                except Exception:  # noqa: BLE001 - policy is best-effort
                    policy = None
        return rsl_env, policy

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    @staticmethod
    def _import_rsl_rl_wrapper():
        """Return the RSL-RL vec-env wrapper class across Isaac Lab versions."""
        try:
            from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

            return RslRlVecEnvWrapper
        except ImportError:
            # Legacy module path (Isaac Lab < 1.0 / pre-rename).
            from omni.isaac.lab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: PLC0415

            return RslRlVecEnvWrapper

    def _build_train_cfg(self, learning_rate: float) -> dict[str, Any]:
        """Build a reasonable RSL-RL ``OnPolicyRunner`` train cfg dict.

        Uses the ``learning_rate`` from the factory arg and otherwise sensible
        PPO defaults for the H1 flat task. ``train_config.extra`` (the
        free-form passthrough) is shallow-merged last so a caller can override
        any hyperparameter without changing this adapter. The exact dict schema
        should be confirmed against the installed rsl_rl version on shakeout.
        """
        train_cfg: dict[str, Any] = {
            "num_steps_per_env": int(getattr(self._train_config, "num_steps_per_env", 24)),
            "max_iterations": self._train_config.train_epochs,
            "empirical_normalization": False,
            "seed": self._train_config.seed or 0,
            "policy": {
                "class_name": "ActorCritic",
                "init_noise_std": 1.0,
                "actor_hidden_dims": [256, 256, 256],
                "critic_hidden_dims": [256, 256, 256],
                "activation": "elu",
            },
            "algorithm": {
                "class_name": "PPO",
                "value_loss_coef": 1.0,
                "use_clipped_value_loss": True,
                "clip_param": 0.2,
                "entropy_coef": 0.005,
                "num_learning_epochs": 5,
                "num_mini_batches": 4,
                "learning_rate": float(learning_rate),
                "schedule": "adaptive",
                "gamma": 0.99,
                "lam": 0.95,
                "desired_kl": 0.01,
                "max_grad_norm": 1.0,
            },
            "save_interval": max(1, self._train_config.checkpoint_interval),
            "experiment_name": "h1_goal_reaching",
            "run_name": "",
            "logger": "tensorboard",
            "device": self._device,
        }
        extra = getattr(self._train_config, "extra", None)
        if isinstance(extra, dict):
            train_cfg.update(extra)
        return train_cfg

    def _collect_metrics(self) -> dict[str, Any]:
        """Best-effort, version-tolerant per-epoch metric extraction.

        Returns whatever mean reward / loss values can be read off the RSL-RL
        runner and its algorithm without assuming a specific rsl_rl version.
        Falls back to an empty dict when nothing is readable — the PPO_Runner
        accepts an empty metrics dict (no divergence signal) and the JSON export
        simply records it as such.
        """
        metrics: dict[str, Any] = {}
        runner = getattr(self, "_runner", None)
        if runner is None:
            return metrics

        iteration = getattr(runner, "current_learning_iteration", None)
        if iteration is not None:
            metrics["iteration"] = int(iteration)

        # Mean episode reward: rsl_rl keeps a rolling reward buffer on the runner
        # in most versions; read it defensively if present.
        for attr in ("rewbuffer", "_rewbuffer"):
            buffer = getattr(runner, attr, None)
            if buffer:
                try:
                    values = list(buffer)
                    metrics["mean_reward"] = float(sum(values) / len(values))
                    break
                except (TypeError, ZeroDivisionError):
                    pass

        # Last PPO losses, if the algorithm exposes them.
        alg = getattr(runner, "alg", None)
        if alg is not None:
            for key in (
                "mean_value_loss",
                "mean_surrogate_loss",
                "mean_entropy_loss",
            ):
                value = getattr(alg, key, None)
                if isinstance(value, (int, float)):
                    metrics[key] = float(value)

        return metrics


def build_isaaclab_trainer(
    train_config: TrainConfig,
    *,
    reward_terms: Any,
    num_envs: int,
    learning_rate: float,
    goal: Any,
    device: str,
    executor: Any = None,
    init_checkpoint: str | None = None,
) -> Trainer:
    """Construct a real RSL-RL/Isaac Lab trainer for one run (lazy import).

    Isaac Lab and RSL-RL only import inside the Isaac Sim runtime, so they are
    imported **here** (and inside :class:`_IsaacLabTrainer`), at call time, never
    at module load. On the controller/dev host (where they are absent) this
    raises a descriptive :class:`RuntimeError`; injecting a :class:`Trainer` (or
    a ``trainer_factory`` into :class:`PPORunner`) bypasses this entirely, which
    is how the unit tests run with no Isaac Lab present.

    The returned :class:`_IsaacLabTrainer` registers the already-wrapped
    ``reward_terms`` (the output of :meth:`RewardExecutor.wrap`, a list of
    ``WrappedReward`` callables) on the goal-conditioned env's ``RewardManager``,
    builds the RSL-RL ``OnPolicyRunner`` at ``num_envs`` on the selected device,
    and drives learning one iteration per :meth:`~_IsaacLabTrainer.run_epoch`.

    Args:
        train_config: The run's :class:`TrainConfig` (env id, epoch budget,
            checkpoint dir/interval, seed, and the ``extra`` hyperparameter
            passthrough).
        reward_terms: The wrapped ``RewTerm`` callable(s) to register on the env.
        num_envs: Parallel env count to build the vec-env at (Req 8.3).
        learning_rate: PPO learning rate for the ``OnPolicyRunner`` (Req 8.1).
        goal: The :class:`~src.data_models.Goal` broadcast across all envs.
        device: The ``CUDA_VISIBLE_DEVICES`` string ``train`` selected (Req 8.2);
            mapped to ``cuda:0`` in-process (see :func:`_train_device_from_visible`).
        executor: Optional :class:`~src.rewards.reward_executor.RewardExecutor`
            backing the :class:`LiveRewardBinding`; one is constructed if omitted.

    Raises:
        RuntimeError: When Isaac Lab / RSL-RL are not importable in this
            environment (the controller/dev host).
    """
    try:  # pragma: no cover - exercised only inside the Isaac Sim runtime
        # Imported lazily and intentionally inside the function body so importing
        # ``src.train.ppo_runner`` never requires Isaac Lab / RSL-RL.
        import isaaclab  # noqa: F401  (presence check)
        from rsl_rl.runners import OnPolicyRunner  # noqa: F401
    except ImportError as exc:  # pragma: no cover - dev host path
        raise RuntimeError(
            "Isaac Lab / RSL-RL are not importable in this environment. Provide "
            "a trainer via PPORunner(trainer_factory=...) or an explicit trainer "
            "to train(trainer=...) to run without the Isaac Sim runtime."
        ) from exc

    return _IsaacLabTrainer(  # pragma: no cover - requires the Isaac Sim runtime
        train_config,
        reward_terms=reward_terms,
        num_envs=num_envs,
        learning_rate=learning_rate,
        goal=goal,
        device=device,
        executor=executor,
        init_checkpoint=init_checkpoint,
    )


# --------------------------------------------------------------------------- #
# PPO_Runner
# --------------------------------------------------------------------------- #
class PPORunner:
    """Executes RSL-RL PPO training on the goal-conditioned H1 environment.

    The runner owns GPU restriction (Req 8.2), the epoch loop and parallel env
    count (Req 8.1, 8.3), checkpoint-interval scheduling (Req 8.4), and returning
    the final checkpoint (Req 8.6). The heavy training backend is an injected
    :class:`Trainer` (or built lazily by :func:`build_isaaclab_trainer`), so all
    of this logic is unit-testable with an in-memory fake trainer.

    Parameters:
        config: The :class:`TrainConfig` for this runner.
        trainer_factory: Optional factory that builds a :class:`Trainer` for a
            run. When omitted, a real Isaac Lab trainer is built lazily at
            train time (which requires the Isaac Sim runtime). Tests inject a
            fake factory here.
        set_environment: Whether ``train`` should set ``CUDA_VISIBLE_DEVICES`` in
            ``os.environ`` (Req 8.2). Defaults to ``True``; tests may disable it
            to avoid mutating the process environment while still asserting on
            the returned device string.
        write_metrics: Whether ``train`` should export the training-metrics JSON
            on completion (Req 8.5). Defaults to ``True``; may be disabled to run
            the loop without touching the filesystem.
    """

    def __init__(
        self,
        config: TrainConfig,
        *,
        trainer_factory: TrainerFactory | None = None,
        set_environment: bool = True,
        write_metrics: bool = True,
    ) -> None:
        self.config = config
        self._trainer_factory = trainer_factory
        self._set_environment = set_environment
        self._write_metrics = write_metrics

    def train(
        self,
        reward_terms: Any,
        epochs: int | None = None,
        learning_rate: float | None = None,
        goal: Any = None,
        num_envs: int | None = None,
        capture_hook: CaptureHook | None = None,
        *,
        trainer: Trainer | None = None,
        init_checkpoint: str | None = None,
    ) -> TrainResult:
        """Train a PPO policy and return a reference to the final checkpoint.

        Implements the Task 10.1 core:

        * Restrict training to the configured GPUs via ``CUDA_VISIBLE_DEVICES``,
          excluding GPU 0 (Req 8.2).
        * Train for ``epochs`` epochs (default ``config.train_epochs``) on the
          goal-conditioned env (Req 8.1) at the configured parallel env count
          (Req 8.3).
        * Write a checkpoint at every multiple of ``config.checkpoint_interval``
          (Req 8.4), and always write a final checkpoint at the last epoch.
        * Export the training-metrics JSON on completion (Req 8.5) and record its
          path on :attr:`TrainResult.metrics_path`.
        * Return a :class:`TrainResult` whose ``checkpoint`` is the final
          checkpoint reference (Req 8.6).

        Args:
            reward_terms: The wrapped ``RewTerm`` callable(s) from the
                Reward_Executor to register on the env (forwarded to the trainer
                factory).
            epochs: Epoch budget; defaults to ``config.train_epochs``.
            learning_rate: PPO learning rate; defaults to
                ``config.learning_rate``.
            goal: The :class:`Goal` to inject into the env (forwarded to the
                trainer factory).
            num_envs: Parallel env count; defaults to ``config.num_envs``
                (Req 8.3).
            capture_hook: Optional training-capture hook (Task 10.4, Req 20).
                Invoked once per epoch with the 1-based training epoch; the hook
                self-gates on the configured Capture_Interval and owns the
                capture env-subset bound, so it is safe to call every epoch and
                never triggers a full-resolution all-env render (Req 20.1, 20.3,
                20.5). The Orchestrator binds the iteration index when wiring it.
            trainer: Optional explicit :class:`Trainer` to use for this run,
                bypassing both the injected factory and the lazy Isaac Lab
                builder (used by tests).

        Returns:
            A :class:`TrainResult` with the final checkpoint reference (Req 8.6).

        Raises:
            ValueError: If ``epochs``/``num_envs`` are invalid, or device
                selection fails (Req 8.2).
        """
        resolved_epochs = self.config.train_epochs if epochs is None else int(epochs)
        resolved_lr = self.config.learning_rate if learning_rate is None else float(learning_rate)
        resolved_envs = self.config.num_envs if num_envs is None else int(num_envs)

        if resolved_epochs < 1:
            raise ValueError(f"epochs must be >= 1, got {resolved_epochs}")
        if resolved_envs < 1:
            raise ValueError(f"num_envs must be >= 1, got {resolved_envs}")

        # --- Req 8.2: restrict devices, excluding the reserved model GPU. ---
        device = select_cuda_visible_devices(
            self.config.training_gpus, llm_gpu=self.config.llm_gpu
        )
        if self._set_environment:
            os.environ["CUDA_VISIBLE_DEVICES"] = device

        # --- Build the training backend (injected/lazy). ---
        backend = self._resolve_trainer(
            trainer=trainer,
            reward_terms=reward_terms,
            num_envs=resolved_envs,
            learning_rate=resolved_lr,
            goal=goal,
            device=device,
            init_checkpoint=init_checkpoint,
        )

        # --- Drive the epoch loop; retry once at the OOM fallback env count. ---
        try:
            return self._run_training_loop(
                backend,
                epochs=resolved_epochs,
                num_envs=resolved_envs,
                device=device,
                capture_hook=capture_hook,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised below unless OOM
            if not is_cuda_oom_error(exc):
                raise
            # --- Req 15.1: CUDA OOM → reduce env count to the configured
            # fallback and retry the run exactly once. ---
            fallback_envs = self.config.oom_fallback_envs
            if fallback_envs < 1 or fallback_envs >= resolved_envs:
                # No usable, strictly-smaller fallback is configured; surface the
                # OOM as the project error rather than retrying uselessly.
                raise OutOfMemoryError(
                    f"CUDA out of memory at num_envs={resolved_envs} and no valid "
                    f"smaller oom_fallback_envs is configured "
                    f"(got {fallback_envs})"
                ) from exc

            # A fresh backend is required: the OOM'd trainer/vec-env is not
            # reusable, and the fallback run must build at the smaller env count.
            fallback_backend = self._resolve_trainer(
                trainer=trainer,
                reward_terms=reward_terms,
                num_envs=fallback_envs,
                learning_rate=resolved_lr,
                goal=goal,
                device=device,
                init_checkpoint=init_checkpoint,
            )
            try:
                return self._run_training_loop(
                    fallback_backend,
                    epochs=resolved_epochs,
                    num_envs=fallback_envs,
                    device=device,
                    capture_hook=capture_hook,
                )
            except BaseException as retry_exc:  # noqa: BLE001
                if is_cuda_oom_error(retry_exc):
                    # Retried once at the fallback and still OOM'd — give up.
                    raise OutOfMemoryError(
                        f"CUDA out of memory again at fallback num_envs="
                        f"{fallback_envs}; training cannot continue"
                    ) from retry_exc
                raise

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _resolve_trainer(
        self,
        *,
        trainer: Trainer | None,
        reward_terms: Any,
        num_envs: int,
        learning_rate: float,
        goal: Any,
        device: str,
        init_checkpoint: str | None = None,
    ) -> Trainer:
        """Return the trainer to use: explicit > injected factory > lazy real."""
        if trainer is not None:
            return trainer
        factory = self._trainer_factory or build_isaaclab_trainer
        kwargs: dict[str, Any] = dict(
            reward_terms=reward_terms,
            num_envs=num_envs,
            learning_rate=learning_rate,
            goal=goal,
            device=device,
        )
        # Only forward init_checkpoint when set, so injected test factories that
        # don't accept the kwarg keep working unchanged (warm-start is opt-in).
        if init_checkpoint:
            kwargs["init_checkpoint"] = init_checkpoint
        return factory(self.config, **kwargs)

    def _run_training_loop(
        self,
        backend: Trainer,
        *,
        epochs: int,
        num_envs: int,
        device: str,
        capture_hook: CaptureHook | None,
    ) -> TrainResult:
        """Run ``epochs`` epochs, scheduling checkpoints, and return the result.

        This is the single seam the OOM retry wraps (:meth:`train` re-runs it at
        ``config.oom_fallback_envs`` after a caught CUDA OOM; Req 15.1) and where
        the divergence check inspects per-epoch metrics (Req 14.1); the Task 10.4
        capture hook fires from inside this loop, once per epoch with the 1-based
        training epoch (the hook self-gates on the Capture_Interval and bounds the
        capture env-subset, so it never forces a full-resolution all-env render —
        Req 20.1, 20.3, 20.5). It advances epochs, raises
        :class:`DivergenceError` on a non-finite loss/reward (Req 14.1), writes
        interval checkpoints (Req 8.4), exports the training-metrics JSON (Req
        8.5), and returns the final checkpoint (Req 8.6).
        """
        scheduled = set(checkpoint_epochs(epochs, self.config.checkpoint_interval))
        checkpoint_paths: list[str] = []
        epoch_metrics: list[dict[str, Any]] = []
        final_path: str | None = None

        for epoch in range(1, epochs + 1):
            metrics = backend.run_epoch(epoch) or {}
            # --- Req 14.1: a non-finite loss/reward means training diverged. ---
            assert_finite_metrics(metrics, epoch=epoch)
            epoch_metrics.append(dict(metrics))

            # --- Req 20.1, 20.3, 20.5: training-capture hook. ---
            # Invoke the hook once per epoch, passing the 1-based training epoch.
            # The hook self-gates on the configured Capture_Interval (so calling
            # every epoch still only produces a capture at the configured
            # cadence) and owns the capture env-subset bound, so this invocation
            # never triggers a full-resolution all-env render.
            if capture_hook is not None:
                capture_hook(epoch)

            is_final = epoch == epochs
            # Write at scheduled interval epochs (Req 8.4) and always at the
            # final epoch so a final checkpoint reference exists (Req 8.6).
            if epoch in scheduled or is_final:
                written = backend.save_checkpoint(
                    self._checkpoint_path(epoch, is_final=is_final), epoch
                )
                checkpoint_paths.append(written)
                if is_final:
                    final_path = written

        # ``epochs >= 1`` is enforced in ``train``; the final epoch always writes.
        assert final_path is not None  # noqa: S101 - invariant guard

        # --- Req 8.5: export the training-metrics JSON on completion. ---
        metrics_path: str | None = None
        if self._write_metrics:
            payload = training_metrics_payload(
                epoch_metrics,
                epochs_completed=epochs,
                num_envs_used=num_envs,
                device=device,
                checkpoint_paths=checkpoint_paths,
                final_checkpoint=final_path,
            )
            metrics_path = export_training_metrics(
                payload, os.path.join(self.config.checkpoint_dir, METRICS_FILENAME)
            )

        # Surface the trainer's live env + inference policy so the Evaluator can
        # reuse them (single SimulationApp per process). Best-effort: a backend
        # without ``live_handles`` (the fake trainer) leaves these ``None``.
        live_env = None
        live_policy = None
        handles = getattr(backend, "live_handles", None)
        if callable(handles):
            try:
                live_env, live_policy = handles()
            except Exception:  # noqa: BLE001 - live handles are best-effort
                live_env, live_policy = None, None

        return TrainResult(
            checkpoint=CheckpointRef(path=final_path),
            checkpoint_paths=checkpoint_paths,
            epochs_completed=epochs,
            num_envs_used=num_envs,
            device=device,
            epoch_metrics=epoch_metrics,
            metrics_path=metrics_path,  # exported JSON path (Req 8.5)
            live_env=live_env,
            live_policy=live_policy,
        )

    def _checkpoint_path(self, epoch: int, *, is_final: bool) -> str:
        """Compute the local checkpoint path for an epoch (Req 8.4, 8.6)."""
        name = "model_final.pt" if is_final else f"model_{epoch}.pt"
        return os.path.join(self.config.checkpoint_dir, name)
