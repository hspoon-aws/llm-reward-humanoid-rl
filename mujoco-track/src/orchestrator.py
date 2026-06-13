"""Orchestrator for the YOUR_REPO Eureka-style loop.

This module owns the *pure-data* models the Eureka-style Orchestrator loop
carries between iterations (Task 13.1) **and** the loop control flow itself
(Task 13.2: :class:`Orchestrator`, :meth:`Orchestrator.run`,
:meth:`Orchestrator.run_iteration`, and :class:`RunResult`).

The Orchestrator wires together the already-implemented components — the
Qwen_Client, Reward_Executor, PPO_Runner, Evaluator, and S3_Store — behind
narrow constructor-injected interfaces, so the control flow is fully
unit-testable on the controller host with in-memory fakes and **no Isaac Sim /
Isaac Lab / torch dependency**.

Scope of Task 13.2 (this change)
--------------------------------
Implements the happy-path loop spine (design.md → Orchestrator behavior):

  - Per iteration: generate reward → validate/inject → train → evaluate →
    record video (Req 7.1).
  - Append Eval_Metrics + behavior description to Metrics_History (Req 7.2).
  - Pass Metrics_History to the client on refinement iterations (Req 7.3).
  - Stop after ``config.max_iterations``, counting completed and skipped
    iterations alike (Req 7.4).
  - Resume from a persisted ``loop_checkpoint`` when present (Req 16.2 resume
    mechanics; the wait-for-endpoint behavior itself is Task 13.6).

Task 13.4 (this change) adds immediate-fall balance guidance on top of the 13.3
recovery layer:

  - After an iteration completes evaluation, if the Evaluator's ``upright_time_s``
    is below ``config.fall_threshold_s`` (an immediate fall), the *next*
    iteration's reward request sets ``balance_priority`` so the Qwen_Client
    injects guidance to prioritize balance and uprightness and de-emphasize
    goal-directed speed (Req 13.1 -> Req 2.2 wiring). The detection
    (:meth:`Orchestrator._immediate_fall_observed`) reads the most recent
    completed iteration's duck-typed Eval_Metrics and is fail-soft (no history,
    absent/non-numeric fields, or an unreadable threshold -> a normal request).

Task 13.5 (this change) adds divergence revert + learning-rate reduction on top
of the 13.4 balance-guidance layer:

  - When the PPO_Runner raises ``DivergenceError`` (a non-finite loss/reward;
    Task 10.3), the loop reverts to ``last_good_reward`` — the most recent
    Reward_Function that produced a valid policy (Req 14.1) — and retries the
    training run once at a reduced learning rate (``base *
    config.lr_reduction_factor``; Req 14.2). The reduction is centralized in
    :meth:`Orchestrator._current_learning_rate` (``reduced=True``) and the
    recovery itself in :meth:`Orchestrator._train_with_divergence_recovery`. When
    the very first iteration diverges (no prior last-good reward) the divergent
    reward is re-used as the revert target so the reduced-LR retry still occurs.

Task 13.6 (this change) adds service-unavailable wait-and-resume on top of the
13.5 divergence-recovery layer:

  - When the Qwen_Client raises ``ServiceUnavailableError`` (the Language_Model
    endpoint is unreachable; classified in Task 3.4 / Req 16.1) — distinct from
    the ``RequestError`` generation-failure *skip* of Task 13.3 — the loop does
    NOT skip the iteration or terminate. Instead it **waits** for the endpoint to
    become reachable again and then **resumes** the same iteration, re-attempting
    the work that raised (Req 16.2). The wait mechanism is injectable for tests:
    :class:`Orchestrator` accepts a ``health_check`` callable (does the endpoint
    answer?) and a ``sleep`` callable (how to back off between polls), both
    defaulting to real implementations (a TCP connect probe against the
    configured ``llm_endpoint`` and :func:`time.sleep`). Resume does not lose
    loop state: the in-memory loop state (history, best-policy pointer, revert
    target, next-iteration index) is untouched while waiting, and the persisted
    ``loop_checkpoint`` (Task 13.2 / Req 16.2) already lets a hard restart resume
    from the last completed iteration.

Task 13.7 (this change) adds per-iteration S3 persistence on top of the 13.6
service-resume layer:

  - When an iteration completes (generate -> validate -> train -> evaluate), the
    Orchestrator persists that iteration's artifacts to the S3_Store under the
    iteration-identifying path (Req 7.5, 11.1, 11.2), *in addition to* the loop
    checkpoint already saved each iteration by Task 13.2. The artifacts are
    bundled by :meth:`Orchestrator._persist_iteration_artifacts`:

      * ``put_iteration_artifacts`` — the reward code (written to a local
        ``reward.py``), the Eval_Metrics JSON (written from the duck-typed
        metrics' ``to_json``), the exported training-metrics JSON (the
        PPO_Runner's ``TrainResult.metrics_path``), the policy checkpoint, and
        any demo-video files surfaced on the metrics/train result.
      * ``put_training_capture`` — one call per Training_Capture the train result
        exposes (duck-typed ``captures``), each landing under the iteration path
        (Req 20.4).

  - Persistence is **fail-soft** (Req 11.3): the S3_Store retains the local copy
    and returns a :class:`~src.storage.s3_store.PersistResult` failure rather
    than raising, and this layer additionally tolerates a store missing the
    ``put_*`` methods (e.g. a test double) and any local-file write error, so a
    persistence failure never aborts the loop. Local artifact files are written
    under a per-run ``local_artifact_dir`` (injectable; defaults to a temp dir)
    so the retained-local copies survive the iteration.

Deliberately structured for, but NOT implementing, the recovery behaviors that
belong to later tasks: training-media/blog assembly (14/15). The seams for these
(``put_blog`` calls) are present so the later tasks slot in without reshaping the
loop.

Task 13.8 (this change) completes the best-policy contract on top of the 13.7
per-iteration persistence layer:

  - Best-policy *tracking* (the in-loop argmax in
    :meth:`Orchestrator._update_best_policy`) compares each completed iteration's
    configured selection-metric score against the current Best_Policy and, on a
    strict improvement, designates that iteration's checkpoint as the new
    Best_Policy and records which iteration produced it (Req 19.1, 19.2, 19.4).
    The tie-break is strict improvement, so the *earliest* iteration to reach a
    given top score wins ties.
  - Best-policy *export* (:meth:`Orchestrator._export_best_policy`) runs once when
    the loop terminates and persists the tracked Best_Policy checkpoint to the
    S3_Store's stable, well-known ``best_policy_path`` via
    ``store.put_best_policy(checkpoint)`` (Req 19.3) — a fixed,
    iteration-independent location, so the final result can be retrieved
    regardless of which iteration produced it. The export is fail-soft and a
    no-op when no iteration completed evaluation.

Task 13.3 (this change) adds the in-iteration recovery for bad generations on top
of the 13.2 spine:

  - On Reward_Executor validation failure, re-prompt the Qwen_Client with the
    validation error text included, up to 3 retries per iteration (Req 12.1, 12.2);
    on exhaustion, record a ``skipped_invalid`` :class:`IterationRecord` (no metrics)
    and proceed to a new iteration (Req 12.3).
  - On reward-generation failure after the client exhausts its own retries
    (``RequestError``), skip the candidate and record a ``skipped_gen_failure``
    iteration without terminating the loop (Req 7.6).

Skipped iterations carry no metrics and are not appended to Metrics_History, but
they still count toward ``max_iterations`` (Req 7.4; handled by :meth:`run`).

Design references:
  - design.md -> Data Models -> ``IterationRecord``, ``MetricsHistory``,
    ``LoopCheckpoint``, ``BestPolicyRef``
  - design.md -> Components and Interfaces -> Orchestrator (loop state:
    ``metrics_history``, ``best_policy``, ``loop_checkpoint``,
    ``last_good_reward``)
  - requirements.md -> Requirement 7.2 (append Eval_Metrics + behavior
    description to Metrics_History), Requirement 16.2 (resume from the last
    completed checkpoint), Requirement 19.4 (record which Iteration produced
    the Best_Policy)

Coupling notes
--------------
``CheckpointRef`` is the canonical reference type produced by the PPO_Runner and
defined alongside the S3_Store (``src/storage/s3_store.py``); it imports only the
standard library, so reusing it here keeps a single source of truth without
pulling in any simulation dependency.

``EvalMetrics`` (design.md -> Data Models) is owned by the Evaluator and metric
models (Task 11.1) and is not implemented yet. To avoid a hard dependency on
unwritten code, ``IterationRecord.metrics`` is duck-typed: any object exposing
the documented ``EvalMetrics`` fields (and optionally ``to_json`` / ``from_json``)
is accepted, and :meth:`MetricsHistory.render_for_prompt` reads those fields
defensively. Serialization round-trips through ``EvalMetrics`` automatically once
that module lands.
"""

from __future__ import annotations

import dataclasses
import os
import socket
import tempfile
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional
from urllib.parse import urlsplit

from .data_models import Goal, GoalRef
from .exceptions import (
    DivergenceError,
    ExecutionError,
    RequestError,
    ServiceUnavailableError,
    ValidationError,
)
from .storage.s3_store import CheckpointRef, IterationArtifacts

if TYPE_CHECKING:  # pragma: no cover - typing only; module may not exist yet
    from .config import Config
    from .eval.metrics import EvalMetrics

__all__ = [
    "IterationStatus",
    "IterationRecord",
    "MetricsHistory",
    "BestPolicyRef",
    "LoopCheckpoint",
    "RunResult",
    "Orchestrator",
    "DEFAULT_TASK_DESCRIPTION",
    "DEFAULT_OBS_SPACE_DESCRIPTION",
    "DEFAULT_ENDPOINT_POLL_INTERVAL_S",
    "DEFAULT_ENDPOINT_PROBE_TIMEOUT_S",
    "make_endpoint_health_check",
]


# --------------------------------------------------------------------------- #
# Service-unavailable wait-and-resume defaults (Task 13.6 / Req 16.2)
# --------------------------------------------------------------------------- #
# How long to sleep between endpoint-reachability polls while the Language_Model
# is down, and the per-probe TCP connect timeout. Both are plain module
# constants (not Config fields) so the wait cadence is explicit and uniform;
# tests inject their own ``sleep``/``health_check`` so neither real value is hit.
DEFAULT_ENDPOINT_POLL_INTERVAL_S = 5.0
DEFAULT_ENDPOINT_PROBE_TIMEOUT_S = 2.0


def _endpoint_host_port(endpoint: str) -> Optional[tuple[str, int]]:
    """Parse ``host``/``port`` from an OpenAI-compatible endpoint URL.

    Returns ``None`` when the endpoint cannot be parsed into a host (so the
    default health check degrades to "assume reachable" rather than blocking
    forever on an unparseable address). Defaults the port from the URL scheme
    (``https`` -> 443, otherwise 80) when none is given.
    """
    candidate = endpoint if "://" in endpoint else f"//{endpoint}"
    try:
        parts = urlsplit(candidate)
        host = parts.hostname
        if not host:
            return None
        port = parts.port
        if port is None:
            port = 443 if parts.scheme == "https" else 80
        return (host, int(port))
    except (ValueError, TypeError):  # pragma: no cover - defensive
        return None


def make_endpoint_health_check(
    endpoint: str, *, timeout_s: float = DEFAULT_ENDPOINT_PROBE_TIMEOUT_S
) -> Callable[[], bool]:
    """Build the default endpoint-reachability probe for ``endpoint`` (Req 16.2).

    Returns a zero-argument callable that performs a short TCP connect to the
    endpoint's host/port and reports whether it succeeded. This is intentionally
    a *transport-level* liveness probe (does the socket accept a connection?) and
    not an HTTP round-trip, so it is cheap and dependency-free. When the endpoint
    cannot be parsed into a host the probe returns ``True`` (assume reachable) so
    a malformed endpoint never wedges the loop in an infinite wait.

    The Orchestrator injects this by default; tests pass their own callable so no
    real socket is opened.
    """
    host_port = _endpoint_host_port(endpoint)

    def _probe() -> bool:
        if host_port is None:
            return True
        try:
            with socket.create_connection(host_port, timeout=timeout_s):
                return True
        except OSError:
            return False

    return _probe


# --------------------------------------------------------------------------- #
# Iteration status vocabulary
# --------------------------------------------------------------------------- #
class IterationStatus:
    """The finite set of statuses an :class:`IterationRecord` may carry.

    Mirrors the design's ``status`` comment on ``IterationRecord``:
    ``"completed" | "skipped_invalid" | "skipped_gen_failure"``. Provided as
    string constants (not an enum) so records serialize to plain JSON strings.
    """

    COMPLETED = "completed"
    SKIPPED_INVALID = "skipped_invalid"
    SKIPPED_GEN_FAILURE = "skipped_gen_failure"
    # A candidate that passed generation + validation + the jit-check but failed
    # at train/eval time (e.g. a tracer leak under the full PPO graph, a backend
    # OOM that survived the fallback, or a divergence that recurred on the
    # reduced-LR retry). Distinct from SKIPPED_GEN_FAILURE so a run log makes it
    # obvious the LANGUAGE MODEL produced a candidate fine and the failure was
    # downstream in training -- this distinction would have surfaced the
    # ``_n_calls`` trainer bug immediately (see §7E).
    SKIPPED_TRAIN_FAILURE = "skipped_train_failure"

    ALL = frozenset(
        {COMPLETED, SKIPPED_INVALID, SKIPPED_GEN_FAILURE, SKIPPED_TRAIN_FAILURE}
    )


# --------------------------------------------------------------------------- #
# Per-iteration record
# --------------------------------------------------------------------------- #
@dataclass
class IterationRecord:
    """A single Orchestrator iteration's outcome (design.md -> Data Models).

    ``metrics`` is ``None`` for skipped iterations (invalid code or a generation
    failure after retries); for completed iterations it is an ``EvalMetrics``-shaped
    object. ``checkpoint`` is the policy checkpoint produced by training, or ``None``
    when the iteration was skipped before training.
    """

    index: int
    reward_code: str
    metrics: Optional["EvalMetrics"]
    behavior_description: str
    status: str
    checkpoint: Optional[CheckpointRef] = None

    @property
    def completed(self) -> bool:
        """True when this iteration ran end-to-end and produced metrics."""
        return self.status == IterationStatus.COMPLETED

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict (supports loop-checkpoint persistence)."""
        return {
            "index": self.index,
            "reward_code": self.reward_code,
            "metrics": _metrics_to_jsonable(self.metrics),
            "behavior_description": self.behavior_description,
            "status": self.status,
            "checkpoint": _checkpoint_to_jsonable(self.checkpoint),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "IterationRecord":
        """Reconstruct from :meth:`to_dict` output (Req 16.2 resume)."""
        return cls(
            index=int(data["index"]),
            reward_code=data.get("reward_code", ""),
            metrics=_metrics_from_jsonable(data.get("metrics")),
            behavior_description=data.get("behavior_description", ""),
            status=data.get("status", IterationStatus.COMPLETED),
            checkpoint=_checkpoint_from_jsonable(data.get("checkpoint")),
        )


# --------------------------------------------------------------------------- #
# Metrics history (fed back to the model)
# --------------------------------------------------------------------------- #
@dataclass
class MetricsHistory:
    """Accumulated per-iteration records rendered as text for the Language_Model.

    The Orchestrator appends one record per completed iteration (Req 7.2) and
    passes the rendered text to the Qwen_Client on refinement iterations
    (Req 1.2, 7.3). :meth:`render_for_prompt` fills the ``{metrics_history}``
    placeholder in ``prompts/refine_reward.txt``.
    """

    records: list[IterationRecord] = field(default_factory=list)

    def append(self, record: IterationRecord) -> None:
        """Append a single iteration record (one per completion; Req 7.2)."""
        self.records.append(record)

    @property
    def completed_records(self) -> list[IterationRecord]:
        """Only the iterations that completed evaluation (carry metrics)."""
        return [r for r in self.records if r.completed]

    def render_for_prompt(self) -> str:
        """Render the history as plain text for the refinement prompt (Req 1.2, 7.3).

        Produces one block per iteration, most recent last, listing the iteration
        index, status, the key goal-reaching metrics (when present), the staged
        capability gates, and the behavior description. Returns a clear sentinel
        when no iterations have been recorded yet so the prompt never contains an
        empty section.
        """
        if not self.records:
            return "(no completed iterations yet)"

        blocks: list[str] = []
        for record in self.records:
            header = f"Iteration {record.index} [{record.status}]"
            metrics_line = _render_metrics(record.metrics)
            if metrics_line:
                header = f"{header}: {metrics_line}"
            lines = [header]
            gates_line = _render_gates(record.metrics)
            if gates_line:
                lines.append(f"  gates: {gates_line}")
            behavior = (record.behavior_description or "").strip()
            if behavior:
                lines.append(f"  behavior: {behavior}")
            blocks.append("\n".join(lines))
        return "\n".join(blocks)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {"records": [r.to_dict() for r in self.records]}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "MetricsHistory":
        """Reconstruct from :meth:`to_dict` output (Req 16.2 resume)."""
        raw_records = data.get("records") or []
        return cls(records=[IterationRecord.from_dict(r) for r in raw_records])


# --------------------------------------------------------------------------- #
# Best-policy pointer
# --------------------------------------------------------------------------- #
@dataclass
class BestPolicyRef:
    """Pointer to the highest-scoring iteration's policy (design.md -> Data Models).

    Records which iteration produced the Best_Policy (Req 19.4) along with the
    checkpoint reference and the value of the configured selection metric used to
    rank it (Req 19.1).
    """

    iteration_index: int
    checkpoint: CheckpointRef
    score: float

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict."""
        return {
            "iteration_index": self.iteration_index,
            "checkpoint": _checkpoint_to_jsonable(self.checkpoint),
            "score": self.score,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BestPolicyRef":
        """Reconstruct from :meth:`to_dict` output."""
        checkpoint = _checkpoint_from_jsonable(data.get("checkpoint"))
        if checkpoint is None:
            raise ValueError("BestPolicyRef requires a checkpoint reference")
        return cls(
            iteration_index=int(data["iteration_index"]),
            checkpoint=checkpoint,
            score=float(data["score"]),
        )


# --------------------------------------------------------------------------- #
# Loop checkpoint (resume after restart / model outage)
# --------------------------------------------------------------------------- #
@dataclass
class LoopCheckpoint:
    """Persisted loop state enabling resume from the last completed checkpoint.

    Holds the next iteration index to run, the accumulated Metrics_History, the
    current Best_Policy pointer, and the most recent reward source that produced a
    valid policy (the divergence-revert target, Req 14.1). The Orchestrator
    reconstructs this on startup to resume after a model outage or process restart
    (Req 16.2).
    """

    next_iteration: int
    history: MetricsHistory
    best_policy: Optional[BestPolicyRef] = None
    last_good_reward: Optional[str] = None
    best_reward_code: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a JSON-safe dict for ``S3Store.save_loop_checkpoint``."""
        return {
            "next_iteration": self.next_iteration,
            "history": self.history.to_dict(),
            "best_policy": (
                self.best_policy.to_dict() if self.best_policy is not None else None
            ),
            "last_good_reward": self.last_good_reward,
            "best_reward_code": self.best_reward_code,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LoopCheckpoint":
        """Reconstruct from :meth:`to_dict` output (Req 16.2 resume)."""
        raw_best = data.get("best_policy")
        return cls(
            next_iteration=int(data["next_iteration"]),
            history=MetricsHistory.from_dict(data.get("history") or {}),
            best_policy=BestPolicyRef.from_dict(raw_best) if raw_best else None,
            last_good_reward=data.get("last_good_reward"),
            best_reward_code=data.get("best_reward_code"),
        )


# --------------------------------------------------------------------------- #
# Rendering helpers (pure; tolerant of duck-typed EvalMetrics)
# --------------------------------------------------------------------------- #
# Goal-reaching metric fields surfaced in the prompt, in display order, paired
# with a compact formatter. Mirrors design.md -> Data Models -> EvalMetrics.
_METRIC_FIELDS: tuple[tuple[str, str], ...] = (
    ("success_rate", "{:.3f}"),
    ("distance_to_goal_m", "{:.3f}"),
    ("time_to_goal_s", "{:.3f}"),
    ("path_efficiency", "{:.3f}"),
    ("upright_time_s", "{:.3f}"),
    ("fall_rate", "{:.3f}"),
    ("avg_forward_speed_mps", "{:.3f}"),
    ("energy_efficiency", "{:.4f}"),
    ("gait_smoothness", "{:.4f}"),
    ("symmetry_score", "{:.3f}"),
)

_GATE_FIELDS: tuple[str, ...] = ("makes_progress", "reaches_goal", "efficient_goal")


def _format_value(value: Any, fmt: str) -> str:
    """Format a numeric metric value, tolerating ints/bools/non-numerics."""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        try:
            return fmt.format(float(value))
        except (ValueError, TypeError):  # pragma: no cover - defensive
            return str(value)
    return str(value)


def _render_metrics(metrics: Any) -> str:
    """Render the curated metric fields present on a duck-typed EvalMetrics."""
    if metrics is None:
        return ""
    parts: list[str] = []
    for name, fmt in _METRIC_FIELDS:
        if not hasattr(metrics, name):
            continue
        value = getattr(metrics, name)
        if value is None:
            continue
        parts.append(f"{name}={_format_value(value, fmt)}")
    return ", ".join(parts)


def _render_gates(metrics: Any) -> str:
    """Render the staged capability gates if the metrics object exposes them."""
    if metrics is None:
        return ""
    gates = getattr(metrics, "gates", None)
    if gates is None:
        return ""
    parts: list[str] = []
    for name in _GATE_FIELDS:
        if hasattr(gates, name):
            parts.append(f"{name}={bool(getattr(gates, name))}")
    return ", ".join(parts)


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _checkpoint_to_jsonable(checkpoint: Any) -> Optional[dict[str, Any]]:
    """Serialize a ``CheckpointRef`` (or compatible) to a plain dict."""
    if checkpoint is None:
        return None
    if dataclasses.is_dataclass(checkpoint) and not isinstance(checkpoint, type):
        return dataclasses.asdict(checkpoint)
    if isinstance(checkpoint, dict):
        return dict(checkpoint)
    # Duck-typed fallback: expose a ``path``/``iteration_index`` shape.
    return {
        "path": getattr(checkpoint, "path", None),
        "iteration_index": getattr(checkpoint, "iteration_index", None),
    }


def _checkpoint_from_jsonable(data: Any) -> Optional[CheckpointRef]:
    """Reconstruct a ``CheckpointRef`` from :func:`_checkpoint_to_jsonable` output."""
    if data is None:
        return None
    if isinstance(data, CheckpointRef):
        return data
    if isinstance(data, dict):
        return CheckpointRef(
            path=data.get("path", ""),
            iteration_index=data.get("iteration_index"),
        )
    # Already a duck-typed checkpoint object.
    return data


def _metrics_to_jsonable(metrics: Any) -> Any:
    """Serialize a duck-typed ``EvalMetrics`` to a JSON-safe structure.

    Prefers the documented ``to_json`` (parsed back to a dict for embedding),
    then ``to_dict``, then ``dataclasses.asdict``; returns ``None`` for absent
    metrics. Unknown shapes are returned unchanged so callers never lose data.
    """
    if metrics is None:
        return None
    to_json = getattr(metrics, "to_json", None)
    if callable(to_json):
        try:
            return {"__eval_metrics_json__": to_json()}
        except Exception:  # pragma: no cover - defensive against partial impls
            pass
    to_dict = getattr(metrics, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if dataclasses.is_dataclass(metrics) and not isinstance(metrics, type):
        return dataclasses.asdict(metrics)
    return metrics


def _metrics_from_jsonable(data: Any) -> Any:
    """Reconstruct metrics from :func:`_metrics_to_jsonable` output.

    When the value was produced via ``EvalMetrics.to_json`` and that module is
    available, round-trips it back into an ``EvalMetrics`` instance; otherwise the
    stored structure is returned as-is so resume still works before Task 11.1
    lands.
    """
    if data is None:
        return None
    if isinstance(data, dict) and "__eval_metrics_json__" in data:
        payload = data["__eval_metrics_json__"]
        try:  # pragma: no cover - exercised once EvalMetrics exists
            from .eval.metrics import EvalMetrics

            return EvalMetrics.from_json(payload)
        except Exception:
            return payload
    return data


# --------------------------------------------------------------------------- #
# Loop-level prompt context (Req 7.1 / Req 1.1, 1.7)
# --------------------------------------------------------------------------- #
# The task and observation-space descriptions the Orchestrator passes to the
# Qwen_Client on every generation. They frame the point-to-point goal-reaching
# task and state explicitly that there is NO image/camera modality (Req 1.7);
# the concrete Goal position + Success_Radius are filled per-run from Config.
DEFAULT_TASK_DESCRIPTION = (
    "Teach a Unitree H1 humanoid to walk from its start position (point A) to a "
    "configurable target position (point B) on flat terrain. The robot must "
    "arrive within the success radius of the Goal while staying upright and "
    "moving efficiently. This is a point-to-point goal-reaching task, not a "
    "velocity-tracking task."
)

DEFAULT_OBS_SPACE_DESCRIPTION = (
    "The policy observation is proprioceptive and goal-conditioned only (no "
    "image or camera modality): base linear velocity, base angular velocity, "
    "projected gravity / orientation, joint positions, joint velocities, and "
    "last action, augmented with a robot-frame Goal_Observation (vector-to-goal, "
    "distance, heading). The exact proprio layout is the MuJoCo Playground "
    "H1JoystickGaitTracking observation. The reward operates on MJX physics "
    "state (base pose/velocity via qpos/qvel, joint positions/velocities, "
    "actuator forces, contact forces via cfrc_ext, and the Goal); it never "
    "reads pixels."
)


# --------------------------------------------------------------------------- #
# Run result
# --------------------------------------------------------------------------- #
@dataclass
class RunResult:
    """The outcome of a full Orchestrator run (design.md -> Orchestrator.run).

    Attributes:
        iterations: One :class:`IterationRecord` per attempted iteration, in
            order (completed and skipped alike; Req 7.4).
        history: The accumulated :class:`MetricsHistory` (Req 7.2).
        best_policy: The highest-scoring iteration's policy pointer, or ``None``
            when no iteration completed evaluation (Req 19). Tracked in-loop by
            the selection-metric argmax (Req 19.1, 19.2, 19.4) and exported to a
            stable, well-known S3 path on termination (Task 13.8 / Req 19.3).
        iterations_run: The number of iterations attempted this run (Req 7.4).
    """

    iterations: list[IterationRecord] = field(default_factory=list)
    history: MetricsHistory = field(default_factory=lambda: MetricsHistory())
    best_policy: Optional[BestPolicyRef] = None
    iterations_run: int = 0


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
class Orchestrator:
    """Drives the Eureka-style generate -> train -> evaluate -> feedback loop.

    Collaborators are injected (design.md -> Orchestrator signature) so the
    control flow is unit-testable with fakes and carries no Isaac Sim / torch
    dependency of its own:

        Orchestrator(config, qwen, executor, runner, evaluator, store)

    Each collaborator is duck-typed to the narrow surface the loop uses:

    * ``qwen.generate_reward(task_description, obs_space, goal_description,
      metrics_history=..., balance_priority=...)`` -> reward code string.
    * ``executor.validate(code)`` -> ``ValidationResult`` (``.ok`` / ``.error``);
      ``executor.wrap(code, goal_ref)`` -> wrapped reward terms.
    * ``runner.train(reward_terms, epochs=..., learning_rate=..., goal=...,
      num_envs=..., capture_hook=...)`` -> a result exposing ``.checkpoint``.
    * ``evaluator.evaluate(checkpoint, goal, num_episodes)`` -> an
      ``EvalMetrics``-shaped object.
    * ``store.save_loop_checkpoint(state)`` / ``store.load_loop_checkpoint()``
      and (Task 13.7) the per-iteration ``put_*`` methods.

    Parameters:
        config: The validated :class:`src.config.Config` for the run.
        qwen: The Qwen_Client.
        executor: The Reward_Executor.
        runner: The PPO_Runner.
        evaluator: The Evaluator.
        store: The S3_Store.
        eval_episodes: Number of evaluation episodes per iteration (forwarded to
            ``evaluator.evaluate``).
        health_check: Zero-argument callable returning ``True`` when the
            Language_Model endpoint is reachable (Task 13.6 / Req 16.2). Injected
            for testability; defaults to a TCP connect probe against
            ``config.llm_endpoint`` built by :func:`make_endpoint_health_check`.
        sleep: One-argument callable used to back off between endpoint polls while
            waiting for the model service to recover. Injected for testability;
            defaults to :func:`time.sleep`.
        endpoint_poll_interval_s: Seconds to wait between reachability polls
            (passed to ``sleep``); defaults to
            :data:`DEFAULT_ENDPOINT_POLL_INTERVAL_S`.
        local_artifact_dir: Local directory under which per-iteration artifact
            files (the reward code and metrics JSON written for upload) are
            staged before persistence (Task 13.7). The S3_Store retains these
            local copies on an upload failure (Req 11.3). Injected for
            testability; defaults to a per-process temp directory.
    """

    # Maximum invalid-code re-prompts per iteration (Req 12.2). The design fixes
    # this at 3; it is a class constant rather than a Config field so the bound
    # is explicit and uniform across runs.
    MAX_INVALID_REPROMPTS = 3

    def __init__(
        self,
        config: "Config",
        qwen: Any,
        executor: Any,
        runner: Any,
        evaluator: Any,
        store: Any,
        *,
        eval_episodes: int = 16,
        health_check: Optional[Callable[[], bool]] = None,
        sleep: Optional[Callable[[float], None]] = None,
        endpoint_poll_interval_s: float = DEFAULT_ENDPOINT_POLL_INTERVAL_S,
        local_artifact_dir: Optional[str] = None,
    ) -> None:
        self.config = config
        self.qwen = qwen
        self.executor = executor
        self.runner = runner
        self.evaluator = evaluator
        self.store = store
        self.eval_episodes = eval_episodes

        # --- Service-unavailable wait-and-resume wiring (Req 16.2). ------- #
        # Both collaborators are injectable so the wait-and-resume path is
        # unit-testable with no real sleeping and no real endpoint. They default
        # to a TCP connect probe against the configured endpoint and real
        # ``time.sleep``.
        self._health_check: Callable[[], bool] = (
            health_check
            if health_check is not None
            else make_endpoint_health_check(getattr(config, "llm_endpoint", ""))
        )
        self._sleep: Callable[[float], None] = sleep if sleep is not None else time.sleep
        self._endpoint_poll_interval_s = float(endpoint_poll_interval_s)

        # --- Per-iteration artifact persistence wiring (Task 13.7). ------- #
        # Local directory under which per-iteration artifact files (reward code,
        # metrics JSON) are written before upload. The S3_Store's fail-soft
        # contract retains these local copies on an upload failure (Req 11.3), so
        # they must live on disk. Defaults to a per-process temp dir so a real
        # run has a stable location and tests can inject a tmp_path.
        self._local_artifact_dir = (
            local_artifact_dir
            if local_artifact_dir is not None
            else tempfile.mkdtemp(prefix="humanoid-artifacts-")
        )

        # The Goal (point B + Success_Radius) is built once from Config and
        # passed to both training and evaluation (design.md -> Goal wiring).
        self.goal = Goal(
            position_xy=config.goal_position,
            success_radius_m=config.success_radius_m,
        )
        self.goal_ref = GoalRef(goal=self.goal)

        # --- Loop state (design.md -> Orchestrator state). --------------- #
        self.history = MetricsHistory()
        self.best_policy: Optional[BestPolicyRef] = None
        # Most recent reward source that produced a valid policy; the divergence
        # revert target (Req 14.1). Threaded here so Task 13.5 can use it.
        self.last_good_reward: Optional[str] = None
        # Source code of the reward that produced the BEST policy so far. Fed back
        # into the refine prompt as the explicit incremental-improvement starting
        # point so the model refines a known-good reward instead of rewriting
        # blind each iteration (the "can't continuously improve" fix).
        self.best_reward_code: Optional[str] = None
        # Next iteration index to run (0-based); advanced by resume (Req 16.2).
        self._next_iteration = 0
        # Transient handle to the most recent completed iteration's training
        # result (PPO_Runner ``TrainResult``-shaped) and demo-video result, set
        # by :meth:`run_iteration` and consumed by
        # :meth:`_persist_iteration_artifacts` (Task 13.7). Kept off the
        # serialized :class:`IterationRecord` so the loop checkpoint stays
        # pure-data; single-threaded, so there is no staleness between produce
        # and persist within one ``run`` step.
        self._pending_train_result: Any = None
        self._pending_demo_result: Any = None

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def run(self) -> RunResult:
        """Run up to ``config.max_iterations`` iterations and return a summary.

        Resumes from a persisted ``loop_checkpoint`` when one is present
        (Req 16.2 resume mechanics): the run continues at the checkpoint's next
        iteration index without losing prior history or the best-policy pointer.
        Stops after ``max_iterations`` total iterations have been attempted,
        counting both completed and skipped iterations (Req 7.4).

        Returns:
            A :class:`RunResult` carrying every attempted iteration record, the
            accumulated Metrics_History, and the best-policy pointer.
        """
        self._maybe_resume()

        attempted: list[IterationRecord] = []
        index = self._next_iteration
        while index < self.config.max_iterations:
            record = self.run_iteration(index)
            attempted.append(record)

            # Live per-iteration logging so an unattended run surfaces WHY an
            # iteration was skipped at the moment it happens (the reason lives in
            # behavior_description; without this it stays hidden until the run
            # ends). Completed iterations print their status; skipped ones also
            # print the reason.
            ckpt_path = getattr(record.checkpoint, "path", None)
            print(
                f"[iter] {record.index}: status={record.status} checkpoint={ckpt_path}",
                flush=True,
            )
            if not record.completed and record.behavior_description:
                print(f"[iter] {record.index}: reason: {record.behavior_description}", flush=True)

            # Append completed iterations to the history fed back to the model
            # (Req 7.2); skipped iterations carry no metrics so they are not
            # appended, but they still count toward max_iterations (Req 7.4).
            if record.completed:
                self.history.append(record)
                self._update_best_policy(record)
                # Persist this iteration's artifacts under its iteration path
                # (Req 7.5, 11.1, 11.2). Fail-soft: a persistence failure retains
                # the local copy and never aborts the loop (Req 11.3).
                self._persist_iteration_artifacts(record)

            # Advance and persist the loop checkpoint so a restart resumes at
            # the next iteration without redoing this one (Req 16.2).
            self._next_iteration = index + 1
            self._save_checkpoint()
            index = self._next_iteration

        # The loop has terminated (``max_iterations`` attempted): export the
        # tracked Best_Policy to its stable, well-known S3 path (Task 13.8 /
        # Req 19.3). Fail-soft and a no-op when no iteration completed.
        self._export_best_policy()

        return RunResult(
            iterations=attempted,
            history=self.history,
            best_policy=self.best_policy,
            iterations_run=len(attempted),
        )

    def run_iteration(self, index: int) -> IterationRecord:
        """Run a single iteration: generate -> validate/inject -> train ->
        evaluate -> record video (Req 7.1).

        The first iteration (no completed history) uses the initial prompt; once
        the history carries completed iterations the rendered history is passed
        to the client so it refines (Req 7.3).

        Returns:
            An :class:`IterationRecord` for the iteration. On the happy path the
            status is ``COMPLETED`` and ``metrics`` is the Evaluator's output.
            When generation fails after the client exhausts its retries the
            status is ``SKIPPED_GEN_FAILURE`` (Req 7.6); when re-prompting with
            the validation error fails to produce valid code within the bound the
            status is ``SKIPPED_INVALID`` (Req 12.3). Skipped records carry no
            metrics or checkpoint.
        """
        # --- GENERATE + VALIDATE with bounded re-prompting (Req 7.1, 12.1,
        # 12.2). The first attempt uses the normal prompt; each subsequent
        # attempt re-prompts the Qwen_Client with the prior validation error
        # included (Req 12.1), up to ``MAX_INVALID_REPROMPTS`` re-prompts
        # (Req 12.2), for at most ``MAX_INVALID_REPROMPTS + 1`` generations. --- #
        is_refinement = bool(self.history.completed_records)
        # Req 13.1: when the most recent completed iteration fell immediately
        # (upright time below the configured fall threshold), request the next
        # reward with balance-priority guidance. Computed once per iteration so
        # every re-prompt in this iteration carries the same flag.
        balance_priority = self._immediate_fall_observed()
        validation_error: Optional[str] = None
        reward_code = ""
        reward_terms = None
        for _attempt in range(self.MAX_INVALID_REPROMPTS + 1):
            try:
                reward_code = self._generate_reward(
                    is_refinement=is_refinement,
                    validation_error=validation_error,
                    balance_priority=balance_priority,
                )
            except RequestError as exc:
                # Req 7.6: generation failed after the client exhausted its own
                # retries. Skip this candidate and let the loop start a fresh
                # generation on the next iteration without terminating.
                return self._skipped_record(
                    index,
                    IterationStatus.SKIPPED_GEN_FAILURE,
                    reward_code="",
                    reason=f"reward generation failed after retries: {exc}",
                )

            result = self.executor.validate(reward_code)
            if not getattr(result, "ok", False):
                # Req 12.1: keep the validation error so the next re-prompt
                # includes it; fall through to re-prompt (or skip below).
                validation_error = (
                    getattr(result, "error", None) or "generated code failed validation"
                )
                continue

            # --- INJECT (Req 7.1): wrap into the JAX reward callable. -------- #
            # wrap() also runs the jax.jit trace-check (Req 5.5): a syntactically
            # valid but NON-jittable reward (value-dependent control flow) raises
            # ExecutionError here, not in validate(). Treat that exactly like a
            # validation failure so it re-prompts with the trace error included
            # (Req 12.1), rather than escaping the recovery loop and aborting.
            try:
                reward_terms = self.executor.wrap(reward_code, self.goal_ref)
            except (ExecutionError, ValidationError) as exc:
                validation_error = f"reward is not jax.jit-traceable: {exc}"
                reward_terms = None
                continue
            break
        else:
            # Req 12.3: the re-prompt bound was reached without valid, wrappable
            # code. Record the failure and proceed to a new iteration.
            return self._skipped_record(
                index,
                IterationStatus.SKIPPED_INVALID,
                reward_code=reward_code,
                reason=validation_error or "generated code failed validation",
            )

        # --- TRAIN (Req 7.1) with divergence recovery (Req 14.1, 14.2). ---- #
        # On a non-finite loss/reward (``DivergenceError`` from the PPO_Runner,
        # Task 10.3) the run reverts to the most recent reward that produced a
        # valid policy (``last_good_reward``; Req 14.1) and retries once at a
        # reduced learning rate (Req 14.2). On the happy path this trains the
        # freshly-generated reward at the base learning rate.
        #
        # Generated reward code is untrusted and varies every iteration; beyond
        # the validate + jit-check gates, a candidate can still fail at
        # train/eval time in ways the gates can't predict (e.g. an output leaf
        # Brax's scan cannot carry, a backend OOM that survives the fallback, a
        # shape mismatch only realized under the full PPO graph). Such a failure
        # must SKIP the candidate and let the loop continue (Req 7.6 spirit), not
        # abort the unattended run. ``DivergenceError`` keeps its dedicated
        # revert-and-reduce-LR path inside ``_train_with_divergence_recovery``.
        try:
            train_result, reward_code, reward_terms = self._train_with_divergence_recovery(
                reward_code, reward_terms
            )
            checkpoint = self._checkpoint_of(train_result, index)
            # This reward produced a valid policy; remember it as the revert
            # target for divergence recovery (Req 14.1).
            self.last_good_reward = reward_code

            # --- EVALUATE + RECORD video (Req 7.1). ---------------------- #
            metrics = self.evaluator.evaluate(
                checkpoint, self.goal, self.eval_episodes
            )
        except DivergenceError:
            # Divergence on the reduced-LR retry too: skip rather than abort.
            return self._skipped_record(
                index,
                IterationStatus.SKIPPED_TRAIN_FAILURE,
                reward_code=reward_code,
                reason="training diverged after reduced-LR retry",
            )
        except Exception as exc:  # noqa: BLE001 - any train/eval failure -> skip, not abort
            return self._skipped_record(
                index,
                IterationStatus.SKIPPED_TRAIN_FAILURE,
                reward_code=reward_code,
                reason=f"training/eval failed for this candidate: {type(exc).__name__}: {exc}",
            )

        # Stash the training result (and any demo-video result the evaluator
        # surfaced) so the per-iteration persistence step in ``run`` can bundle
        # the training-metrics JSON, checkpoint, captures, and demo videos under
        # the iteration path (Task 13.7 / Req 7.5, 11.1). Kept transient and off
        # the IterationRecord so the loop checkpoint stays pure-data.
        self._pending_train_result = train_result
        self._pending_demo_result = getattr(metrics, "demo_video", None)

        behavior_description = self._describe_behavior(metrics)
        return IterationRecord(
            index=index,
            reward_code=reward_code,
            metrics=metrics,
            behavior_description=behavior_description,
            status=IterationStatus.COMPLETED,
            checkpoint=checkpoint,
        )

    # ------------------------------------------------------------------ #
    # Internals (each a seam for a later recovery task)
    # ------------------------------------------------------------------ #
    def _generate_reward(
        self,
        *,
        is_refinement: bool,
        validation_error: Optional[str] = None,
        balance_priority: bool = False,
    ) -> str:
        """Request a reward function from the Qwen_Client (Req 7.1, 7.3, 12.1, 13.1).

        Passes the rendered Metrics_History only on refinement iterations so the
        model refines against prior results (Req 7.3); the first iteration uses
        the initial prompt (no history). When ``validation_error`` is supplied
        (a re-prompt after a rejected candidate), it is forwarded to the client
        so the request includes the error text (Req 12.1). When
        ``balance_priority`` is set the client injects balance-priority guidance
        (Req 13.1 -> Req 2.2 wiring); the immediate-fall detection that decides
        the flag lives in :meth:`_immediate_fall_observed`.

        Propagates :class:`RequestError` (raised by the client once it exhausts
        its own retries) so :meth:`run_iteration` can apply the generation-failure
        skip (Req 7.6).
        """
        history_text = self.history.render_for_prompt() if is_refinement else None
        best_code = self.best_reward_code or self.last_good_reward
        return self._call_with_service_resume(
            lambda: self.qwen.generate_reward(
                DEFAULT_TASK_DESCRIPTION,
                DEFAULT_OBS_SPACE_DESCRIPTION,
                self._goal_description(),
                metrics_history=history_text,
                validation_error=validation_error,
                balance_priority=balance_priority,
                best_reward_code=best_code,
            )
        )

    def _call_with_service_resume(self, call: Callable[[], Any]) -> Any:
        """Invoke a Qwen_Client call, waiting out a model-service outage (Req 16.2).

        Wraps a single Language_Model interaction so that a
        :class:`ServiceUnavailableError` (the endpoint is unreachable; Req 16.1)
        does NOT skip the iteration or terminate the loop — that is the distinct
        ``RequestError`` generation-failure behavior of Task 13.3. Instead, this
        waits for the endpoint to become reachable again
        (:meth:`_wait_for_endpoint`) and then re-attempts the same call,
        resuming the work that raised. ``RequestError`` and every other
        exception propagate unchanged so the existing recovery edges (skip,
        divergence revert) keep their semantics.

        Loop state (history, best-policy pointer, revert target, next-iteration
        index) is untouched across the wait, so the resume loses nothing; a hard
        process restart additionally resumes from the persisted loop checkpoint
        (Task 13.2 / Req 16.2).
        """
        while True:
            try:
                return call()
            except ServiceUnavailableError:
                # Req 16.2: the endpoint is down. Block until it answers again,
                # then retry the same call (resume, not skip/terminate).
                self._wait_for_endpoint()

    def _wait_for_endpoint(self) -> None:
        """Block until the Language_Model endpoint is reachable again (Req 16.2).

        Polls the injected ``health_check`` and sleeps the injected ``sleep``
        between polls until the probe reports the endpoint is reachable. Both
        collaborators are injected (defaulting to a TCP connect probe and
        :func:`time.sleep`) so the wait is fully unit-testable without real
        sleeping or a real endpoint. A health check that raises is treated as
        "still unreachable" so a flaky probe cannot abort the loop.
        """
        while True:
            try:
                if self._health_check():
                    return
            except Exception:  # noqa: BLE001 - a failing probe == still down
                pass
            self._sleep(self._endpoint_poll_interval_s)

    def _immediate_fall_observed(self) -> bool:
        """Whether the most recent completed iteration fell immediately (Req 13.1).

        Reads the latest completed iteration's Eval_Metrics (duck-typed) and
        reports ``True`` when its ``upright_time_s`` is below the configured
        ``fall_threshold_s``, signaling the next reward request should prioritize
        balance and uprightness over goal-directed speed (Req 13.1; wired to the
        Qwen_Client's ``balance_priority`` guidance, Req 2.2).

        Defensive by design: with no completed history, no metrics, a missing or
        non-numeric ``upright_time_s``, or an unreadable threshold, it returns
        ``False`` so a normal (non-balance) request is made.
        """
        completed = self.history.completed_records
        if not completed:
            return False
        metrics = completed[-1].metrics
        if metrics is None:
            return False
        upright = getattr(metrics, "upright_time_s", None)
        if upright is None or isinstance(upright, bool):
            return False
        threshold = getattr(self.config, "fall_threshold_s", None)
        if threshold is None:
            return False
        try:
            return float(upright) < float(threshold)
        except (TypeError, ValueError):
            return False

    def _skipped_record(
        self,
        index: int,
        status: str,
        *,
        reward_code: str,
        reason: str,
    ) -> IterationRecord:
        """Build a skipped :class:`IterationRecord` (Req 7.6, 12.3).

        Skipped iterations never trained or evaluated, so they carry no metrics
        and no checkpoint; the reason is preserved in ``behavior_description``
        for the run log. They are not appended to Metrics_History but still count
        toward ``max_iterations`` (handled in :meth:`run`).
        """
        return IterationRecord(
            index=index,
            reward_code=reward_code,
            metrics=None,
            behavior_description=reason,
            status=status,
            checkpoint=None,
        )

    def _goal_description(self) -> str:
        """Render the Goal (point B + Success_Radius) for the prompt (Req 1.1)."""
        x, y = self.goal.position_xy
        return (
            f"Reach the Goal at (x={x:g}, y={y:g}) on the ground plane and stop "
            f"within {self.goal.success_radius_m:g} m of it without falling."
        )

    def _train_with_divergence_recovery(
        self, reward_code: str, reward_terms: Any
    ) -> tuple[Any, str, Any]:
        """Train, reverting to the last-good reward at a reduced LR on divergence.

        Runs ``runner.train`` for the freshly-generated reward at the base
        learning rate. If the PPO_Runner raises :class:`DivergenceError` (a
        non-finite loss/reward; Task 10.3), the run reverts to the most recent
        Reward_Function that produced a valid policy (``self.last_good_reward``;
        Req 14.1) and retries the training run once at a reduced learning rate
        (``base * config.lr_reduction_factor``; Req 14.2).

        The revert target is re-wrapped through ``executor.wrap`` so the retry
        trains the reverted reward's terms. When no last-good reward is available
        yet (the very first iteration diverged), the divergent reward itself is
        re-used as the revert target so the reduced-LR retry still happens.

        Returns:
            A ``(train_result, reward_code, reward_terms)`` triple. On the happy
            path these are the original arguments and the result; after a revert
            they are the reverted reward's code, its re-wrapped terms, and the
            retry's training result. A :class:`DivergenceError` on the retry
            propagates (a single retry is attempted, per Req 14.2).
        """
        try:
            train_result = self._train(reward_terms, self._current_learning_rate())
            return train_result, reward_code, reward_terms
        except DivergenceError:
            # Req 14.1: revert to the most recent reward that produced a valid
            # policy. Before any iteration has succeeded there is no prior
            # last-good reward, so fall back to the divergent reward itself so a
            # reduced-LR retry still occurs.
            revert_code = self.last_good_reward or reward_code
            revert_terms = self.executor.wrap(revert_code, self.goal_ref)
            # Req 14.2: retry once at the reduced learning rate.
            train_result = self._train(
                revert_terms, self._current_learning_rate(reduced=True)
            )
            return train_result, revert_code, revert_terms

    def _train(self, reward_terms: Any, learning_rate: float) -> Any:
        """Invoke the PPO_Runner for one training run at ``learning_rate`` (Req 7.1).

        When ``config.warm_start`` is enabled, the training is seeded from the
        best policy found so far (``self.best_policy.checkpoint``) instead of
        random init, so each iteration FINE-TUNES an improving policy rather than
        relearning balance from scratch. The very first iteration (no best policy
        yet) trains from scratch as before. The warm-start path is passed through
        to the PPO_Runner, which loads the saved params into Brax's
        ``restore_params``; a missing/unreadable checkpoint falls back to random
        init (fail-soft) so a warm-start problem never aborts the loop."""
        warm_start_checkpoint = None
        if getattr(self.config, "warm_start", False) and self.best_policy is not None:
            warm_start_checkpoint = getattr(self.best_policy.checkpoint, "path", None)
        return self.runner.train(
            reward_terms,
            epochs=self.config.train_epochs,
            learning_rate=learning_rate,
            goal=self.goal,
            num_envs=self.config.num_envs,
            warm_start_checkpoint=warm_start_checkpoint,
        )

    def _current_learning_rate(self, *, reduced: bool = False) -> float:
        """The learning rate for the next training run (Req 14.2).

        Returns the config-derived base learning rate on the happy path. When
        ``reduced`` is set (a divergence retry; Task 13.5) the base rate is
        scaled by ``config.lr_reduction_factor`` so the retry runs at a strictly
        smaller learning rate (the loader constrains the factor to ``(0, 1)``,
        Req 14.2).
        """
        base = float(getattr(self.config, "learning_rate", 1.0e-3))
        if reduced:
            factor = float(getattr(self.config, "lr_reduction_factor", 0.5))
            return base * factor
        return base

    def _describe_behavior(self, metrics: Any) -> str:
        """Build a short behavior description from Eval_Metrics (Req 7.2).

        This text is appended to the Metrics_History alongside the metrics and
        fed back to the model. It is intentionally terse and derived only from
        the metric fields the Evaluator exposes (duck-typed so it works before
        and after the EvalMetrics module lands).
        """
        if metrics is None:
            return ""
        parts: list[str] = []
        success_rate = getattr(metrics, "success_rate", None)
        if success_rate is not None:
            parts.append(f"success_rate={float(success_rate):.2f}")
        distance = getattr(metrics, "distance_to_goal_m", None)
        if distance is not None:
            parts.append(f"final_distance_to_goal={float(distance):.2f}m")
        upright = getattr(metrics, "upright_time_s", None)
        if upright is not None:
            parts.append(f"upright_time={float(upright):.2f}s")
        fall_rate = getattr(metrics, "fall_rate", None)
        if fall_rate is not None:
            parts.append(f"fall_rate={float(fall_rate):.2f}")
        if not parts:
            return ""
        return "Goal-reaching behavior: " + ", ".join(parts) + "."

    def _checkpoint_of(self, train_result: Any, index: int) -> Optional[CheckpointRef]:
        """Extract the final-checkpoint reference from a training result.

        Accepts either a ``TrainResult``-shaped object exposing ``.checkpoint``
        or a bare :class:`CheckpointRef`; stamps the iteration index when the
        checkpoint does not already carry one.
        """
        checkpoint = getattr(train_result, "checkpoint", train_result)
        if isinstance(checkpoint, CheckpointRef) and checkpoint.iteration_index is None:
            checkpoint = dataclasses.replace(checkpoint, iteration_index=index)
        return checkpoint

    def _selection_score(self, metrics: Any) -> Optional[float]:
        """Composite selection score: the configured metric plus small shaped
        tie-breakers (Req 19.1, with a continuous-improvement fix).

        The configured ``selection_metric`` (default ``success_rate``) is the
        dominant term. But for a hard task it can be identically 0.0 across every
        early iteration, which makes the in-loop argmax a permanent tie on the
        first iteration — so the Best_Policy never updates and warm-start always
        re-seeds from iteration 0 even when a later iteration balances longer.
        To give the loop a usable gradient before any success, we ADD tiny
        tie-breaker terms (upright time, then closeness to the goal) scaled so
        they can only ever break ties in the primary metric, never override a
        genuine difference in it.

        Returns ``None`` only when the primary metric is entirely absent.
        """
        name = getattr(self.config, "selection_metric", None)
        if not name:
            return None
        primary = getattr(metrics, name, None)
        if primary is None or isinstance(primary, bool):
            return None
        try:
            score = float(primary)
        except (TypeError, ValueError):
            return None

        # --- shaped tie-breakers (only meaningful when primary values tie) --- #
        # Upright time: more standing is strictly better for a fall-prone
        # humanoid. Scaled to < 1 unit of the primary metric (success_rate is in
        # [0,1], so a 1e-3 * seconds bonus, capped, can't cross a 1.0 success gap).
        upright = getattr(metrics, "upright_time_s", None)
        if isinstance(upright, (int, float)) and not isinstance(upright, bool):
            try:
                score += 1e-3 * min(float(upright), 100.0)
            except (TypeError, ValueError):
                pass
        # Closeness to goal: smaller distance is better. Even tinier weight so it
        # only breaks ties left after upright time.
        dist = getattr(metrics, "distance_to_goal_m", None)
        if isinstance(dist, (int, float)) and not isinstance(dist, bool):
            try:
                score += 1e-5 * (100.0 - min(float(dist), 100.0))
            except (TypeError, ValueError):
                pass
        return score

    def _update_best_policy(self, record: IterationRecord) -> None:
        """Track the highest-scoring completed iteration (Req 19.1, 19.2, 19.4).

        Task 13.8 owns the full best-policy contract (tie-break policy and the
        stable-path export on termination); this hook performs the in-loop
        argmax so the pointer is available in the :class:`RunResult`. It is a
        strict-improvement comparison (first iteration wins ties by arriving
        first), and it is skipped when the metric or checkpoint is unavailable.
        """
        score = self._selection_score(record.metrics)
        if score is None or record.checkpoint is None:
            return
        if self.best_policy is None or score > self.best_policy.score:
            self.best_policy = BestPolicyRef(
                iteration_index=record.index,
                checkpoint=record.checkpoint,
                score=score,
            )
            # Remember the reward SOURCE that produced this best policy so the
            # refine prompt can start the next candidate from it (incremental
            # improvement instead of blind rewrite).
            self.best_reward_code = record.reward_code or self.best_reward_code

    def _export_best_policy(self) -> None:
        """Export the tracked Best_Policy to a stable, well-known path (Req 19.3).

        Called once when the loop terminates (after ``max_iterations`` have been
        attempted). The in-loop argmax of :meth:`_update_best_policy` has already
        designated the highest-scoring completed iteration's checkpoint as
        ``self.best_policy`` (Req 19.1, 19.2) and recorded which iteration
        produced it (Req 19.4); this step persists that checkpoint to the
        S3_Store's iteration-independent ``best_policy_path`` via
        ``store.put_best_policy(checkpoint)`` so the final demo/export can be
        retrieved from one fixed location regardless of which iteration won.

        Fail-soft contract (mirrors the rest of the loop): a no-op when no
        iteration completed evaluation (``best_policy is None``), tolerant of a
        store/test-double that does not implement ``put_best_policy``, and the
        S3_Store's own ``put_best_policy`` already retains the local copy and
        returns a :class:`~src.storage.s3_store.PersistResult` failure rather
        than raising — so an export failure never turns a finished run into a
        crash.
        """
        if self.best_policy is None:
            return
        put_best = getattr(self.store, "put_best_policy", None)
        if not callable(put_best):
            return
        try:
            put_best(self.best_policy.checkpoint)
        except Exception:  # noqa: BLE001 - fail-soft: export never crashes a finished run
            return

    # ------------------------------------------------------------------ #
    # Per-iteration S3 persistence (Task 13.7 / Req 7.5, 11.1, 11.2, 11.3)
    # ------------------------------------------------------------------ #
    def _persist_iteration_artifacts(self, record: IterationRecord) -> None:
        """Persist a completed iteration's artifacts to the S3_Store (Req 7.5, 11.1).

        Called once per *completed* iteration (after the loop checkpoint is in
        flight), this bundles the iteration's artifacts under its
        iteration-identifying path (Req 11.2) via the injected S3_Store:

          * :meth:`store.put_iteration_artifacts` with a ``files`` mapping of
            ``{name: local_path}`` covering the reward code (``reward.py``,
            written from ``record.reward_code``), the Eval_Metrics JSON
            (``metrics.json``, written from the duck-typed metrics' ``to_json``),
            the exported training-metrics JSON (the PPO_Runner's
            ``TrainResult.metrics_path``), the policy checkpoint
            (``record.checkpoint.path``), and any best/worst demo-video files the
            Evaluator surfaced (Req 11.1).
          * :meth:`store.put_training_capture` once per Training_Capture the
            training result exposes (duck-typed ``captures``), each landing under
            the same iteration path (Req 20.4).

        Fail-soft contract (Req 11.3): the S3_Store already retains the local copy
        and returns a :class:`~src.storage.s3_store.PersistResult` failure rather
        than raising. This layer additionally tolerates a store that does not
        implement the ``put_*`` methods (e.g. a lightweight test double) and any
        local-file write error, so a persistence failure never aborts the loop.
        The transient ``_pending_*`` handles set by :meth:`run_iteration` are
        consumed and cleared here.
        """
        train_result = self._pending_train_result
        demo_result = self._pending_demo_result
        # Clear the transient handles immediately so a later iteration can never
        # persist stale results even if this method returns early.
        self._pending_train_result = None
        self._pending_demo_result = None

        try:
            files = self._iteration_artifact_files(record, train_result, demo_result)
            put_artifacts = getattr(self.store, "put_iteration_artifacts", None)
            if files and callable(put_artifacts):
                put_artifacts(record.index, IterationArtifacts(files=files))

            self._persist_training_captures(record.index, train_result)
        except Exception:  # noqa: BLE001 - fail-soft: persistence never aborts the loop
            return

    def _iteration_artifact_files(
        self, record: IterationRecord, train_result: Any, demo_result: Any
    ) -> dict[str, str]:
        """Build the ``{name: local_path}`` artifact mapping for an iteration.

        Writes the text artifacts (reward code, Eval_Metrics JSON) to local files
        under the per-iteration staging directory so the S3_Store has on-disk
        copies to upload and to retain on failure (Req 11.3). References the
        already-on-disk artifacts (training-metrics JSON, checkpoint, demo videos)
        by their existing paths. Every step is individually fail-soft so one
        unwritable artifact never drops the others.
        """
        files: dict[str, str] = {}

        # reward.py — the generated reward source (Req 11.1).
        reward_path = self._write_iteration_file(
            record.index, "reward.py", record.reward_code or ""
        )
        if reward_path is not None:
            files["reward.py"] = reward_path

        # metrics.json — the Eval_Metrics JSON (Req 11.1). Prefer the duck-typed
        # ``to_json`` (EvalMetrics, Task 11.1); skip silently when unavailable.
        metrics_json = self._metrics_json(record.metrics)
        if metrics_json is not None:
            metrics_path = self._write_iteration_file(
                record.index, "metrics.json", metrics_json
            )
            if metrics_path is not None:
                files["metrics.json"] = metrics_path

        # training_metrics.json — exported by the PPO_Runner (Req 8.5); already on
        # disk, referenced by path.
        train_metrics_path = getattr(train_result, "metrics_path", None)
        if train_metrics_path:
            files["training_metrics.json"] = str(train_metrics_path)

        # The policy checkpoint produced by training (Req 11.1).
        checkpoint_path = getattr(record.checkpoint, "path", None)
        if checkpoint_path:
            files[os.path.basename(str(checkpoint_path))] = str(checkpoint_path)

        # Best/worst demo-video files surfaced by the Evaluator (Req 11.1).
        for name, path in self._demo_video_files(demo_result).items():
            files[name] = path

        return files

    def _persist_training_captures(self, index: int, train_result: Any) -> None:
        """Persist each Training_Capture the training result exposes (Req 20.4).

        The training result may surface a sequence of Training_Capture-shaped
        objects on a duck-typed ``captures`` attribute; each is persisted under
        the iteration path via ``store.put_training_capture``. Fail-soft and
        tolerant of a store/test-double without the method or a result without
        captures.
        """
        captures = getattr(train_result, "captures", None)
        if not captures:
            return
        put_capture = getattr(self.store, "put_training_capture", None)
        if not callable(put_capture):
            return
        for capture in captures:
            put_capture(index, capture)

    def _demo_video_files(self, demo_result: Any) -> dict[str, str]:
        """Collect best/worst demo-video file paths from an Evaluator result.

        Accepts the Evaluator's ``DemoVideoResult`` shape (``best``/``worst``,
        each a ``DemoVideo`` exposing ``video_paths``) and returns a
        ``{name: local_path}`` mapping with collision-free, role-prefixed names.
        Tolerant of ``None`` and partially-populated results.
        """
        files: dict[str, str] = {}
        if demo_result is None:
            return files
        for role in ("best", "worst"):
            demo = getattr(demo_result, role, None)
            if demo is None:
                continue
            paths = getattr(demo, "video_paths", None) or ()
            for position, path in enumerate(paths):
                if not path:
                    continue
                name = f"{role}-{position}-{os.path.basename(str(path))}"
                files[name] = str(path)
        return files

    @staticmethod
    def _metrics_json(metrics: Any) -> Optional[str]:
        """Serialize duck-typed Eval_Metrics to a JSON string, or ``None``.

        Prefers the documented ``to_json`` (EvalMetrics); returns ``None`` when
        metrics are absent or expose no usable serializer so the caller simply
        omits the file rather than failing.
        """
        if metrics is None:
            return None
        to_json = getattr(metrics, "to_json", None)
        if callable(to_json):
            try:
                return to_json()
            except Exception:  # noqa: BLE001 - defensive against partial impls
                return None
        return None

    def _write_iteration_file(
        self, index: int, name: str, text: str
    ) -> Optional[str]:
        """Write ``text`` to ``<local_artifact_dir>/iteration-NN/<name>``.

        Returns the absolute local path on success or ``None`` on any write
        error so persistence stays fail-soft (Req 11.3): an unwritable local
        staging file is skipped rather than aborting the iteration.
        """
        try:
            iter_dir = os.path.join(
                self._local_artifact_dir, f"iteration-{int(index):02d}"
            )
            os.makedirs(iter_dir, exist_ok=True)
            path = os.path.join(iter_dir, name)
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(text)
            return path
        except OSError:
            return None

    # ------------------------------------------------------------------ #
    # Resume / checkpoint persistence (Req 16.2 resume mechanics)
    # ------------------------------------------------------------------ #
    def _maybe_resume(self) -> None:
        """Restore loop state from a persisted checkpoint when present (Req 16.2).

        ``store.load_loop_checkpoint`` returns the parsed JSON object (or
        ``None``). When present, the run continues at the stored next-iteration
        index with the prior history, best-policy pointer, and revert target
        intact. A missing or unreadable checkpoint degrades to a fresh start.
        """
        loader = getattr(self.store, "load_loop_checkpoint", None)
        if not callable(loader):
            return
        try:
            raw = loader()
        except Exception:  # noqa: BLE001 - absent/unreadable -> fresh start
            return
        if not raw:
            return
        try:
            checkpoint = LoopCheckpoint.from_dict(raw)
        except Exception:  # noqa: BLE001 - malformed checkpoint -> fresh start
            return
        self._next_iteration = checkpoint.next_iteration
        self.history = checkpoint.history
        self.best_policy = checkpoint.best_policy
        self.last_good_reward = checkpoint.last_good_reward
        self.best_reward_code = checkpoint.best_reward_code or self.best_reward_code

    def _current_checkpoint(self) -> LoopCheckpoint:
        """Snapshot the current loop state for persistence (Req 16.2)."""
        return LoopCheckpoint(
            next_iteration=self._next_iteration,
            history=self.history,
            best_policy=self.best_policy,
            last_good_reward=self.last_good_reward,
            best_reward_code=self.best_reward_code,
        )

    def _save_checkpoint(self) -> None:
        """Persist the loop checkpoint, fail-soft (Req 16.2).

        The S3_Store's ``save_loop_checkpoint`` is already fail-soft; this guard
        additionally tolerates a store that does not implement it (e.g. a test
        double) so the loop never aborts on checkpoint persistence.
        """
        saver = getattr(self.store, "save_loop_checkpoint", None)
        if not callable(saver):
            return
        try:
            saver(self._current_checkpoint())
        except Exception:  # noqa: BLE001 - fail-soft: never abort the loop
            return
