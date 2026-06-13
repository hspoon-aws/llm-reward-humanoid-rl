"""Training-media capture for the YOUR_REPO system (spec Req 20).

This module produces **Training_Captures** — periodic visual captures (a still
Screenshot and/or a short video clip) of the simulation taken *during* training,
distinct from the evaluation-time best/worst demo videos (Req 10). The design
assigns capture coordination to the Orchestrator, which configures the
PPO_Runner's ``capture_hook`` to emit a Training_Capture at the configured
Capture_Interval, tagged with iteration index, training epoch, and wall-clock
time, limited to the capture env-subset size, and persisted under the iteration
path via the S3_Store.

Design references:
  - design.md -> Components and Interfaces -> Orchestrator (capture coordination)
  - design.md -> Components and Interfaces -> PPO_Runner (``capture_hook``)
  - design.md -> Data Models (``TrainingCapture``)
  - requirements.md -> Requirement 20 (produce + persist training media)

Responsibilities (Req 20):
  - Produce a Training_Capture at the configured Capture_Interval (Req 20.1, 20.5).
  - Tag each capture with the Iteration index and training progress — training
    epoch and wall-clock time — so captures order deterministically for the Blog
    narrative (Req 20.2).
  - Produce at the configured capture resolution and limit each capture to the
    configured capture env-subset size of parallel Environment instances; never a
    full-resolution all-env render (Req 20.3, 20.5).
  - Persist each Training_Capture as an Artifact to the S3_Store under a path that
    identifies its Iteration (Req 20.4) — reusing
    :meth:`src.storage.s3_store.S3Store.put_training_capture` rather than
    duplicating persistence logic.

Rendering dependency (testability)
----------------------------------
Actual frame rendering requires Isaac Sim / Isaac Lab, which is **not available**
on the controller/dev host. The renderer is therefore an **injected** dependency
(:class:`CaptureRenderer`): any object exposing
``render(iteration_index, training_epoch, resolution, env_subset_size) ->
RenderOutput`` is accepted. This lets the production + persistence logic be
unit-tested with an in-memory fake renderer and a fake S3 client, with no GPU,
no Isaac Sim import, and no real AWS calls.

When **no renderer is injected** the producer is a safe no-op: it returns
``None`` instead of producing a capture, so an unattended training run continues
uninterrupted even where rendering is unavailable (the capture concern is
strictly additive and never fatal — consistent with the fail-soft persistence
contract in Req 11.3).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable

from .storage.s3_store import PersistResult, S3Store, TrainingCapture

__all__ = [
    "RenderOutput",
    "CaptureRenderer",
    "CaptureResult",
    "TrainingCaptureProducer",
    "CaptureHook",
]


# A capture hook is the callable the PPO_Runner invokes during training. It
# receives the current iteration index and training epoch and returns the
# produced :class:`TrainingCapture` (or ``None`` when no capture was due / no
# renderer is available). Persistence is performed as a side effect.
CaptureHook = Callable[[int, int], "TrainingCapture | None"]


@dataclass
class RenderOutput:
    """The media produced by a :class:`CaptureRenderer` for one capture.

    At least one of ``screenshot_path`` / ``clip_path`` should be set; a render
    that yields neither is treated as "nothing captured" (Req 20.1 requires each
    Training_Capture to contain a Screenshot and/or a short clip).
    """

    screenshot_path: str | None = None
    clip_path: str | None = None


@runtime_checkable
class CaptureRenderer(Protocol):
    """Renders a small, fixed-resolution view of the running simulation.

    Implementations (Isaac Sim-backed in production) MUST honor ``resolution``
    and render at most ``env_subset_size`` parallel Environment instances — never
    a full-resolution render of all training envs (Req 20.3, 20.5). The producer
    only ever passes the configured resolution and a clamped env-subset size, so
    the scale bound is enforced structurally at the call site.
    """

    def render(
        self,
        *,
        iteration_index: int,
        training_epoch: int,
        resolution: tuple[int, int],
        env_subset_size: int,
    ) -> RenderOutput | None:
        ...


@dataclass
class CaptureResult:
    """Outcome of a produce-and-persist cycle.

    ``capture`` is the produced :class:`TrainingCapture` (``None`` when no capture
    was due or no renderer was available). ``persist`` is the S3_Store
    :class:`~src.storage.s3_store.PersistResult` when persistence was attempted.
    """

    capture: TrainingCapture | None = None
    persist: PersistResult | None = None


class TrainingCaptureProducer:
    """Produces and persists Training_Captures at the configured cadence (Req 20).

    The producer is intentionally free of any Isaac Sim / torch dependency: it
    drives an injected :class:`CaptureRenderer` and an
    :class:`~src.storage.s3_store.S3Store`, so the production + persistence logic
    is unit-testable on a CPU-only host with fakes.

    Parameters
    ----------
    store:
        The :class:`~src.storage.s3_store.S3Store` used to persist each capture
        under its iteration path (Req 20.4).
    capture_interval:
        The Capture_Interval, measured in training epochs (Config
        ``capture_interval``). A capture is produced the first time an epoch falls
        into a new interval bucket — i.e. at epochs ``0``, ``interval``,
        ``2*interval`` ... — so captures occur **only** at the configured cadence
        (Req 20.1, 20.5).
    capture_resolution:
        The ``(width, height)`` render resolution (Config ``capture_resolution``),
        passed verbatim to the renderer (Req 20.3).
    capture_env_subset_size:
        The maximum number of parallel Environment instances to include in a
        capture (Config ``capture_env_subset_size``). Clamped to ``total_envs``
        when known so it never exceeds the live env count (Req 20.3, 20.5).
    renderer:
        Injected :class:`CaptureRenderer`. When ``None`` the producer is a no-op
        (returns ``None``), so a run without rendering support is unaffected.
    total_envs:
        Optional count of parallel training Environment instances; used only to
        clamp the effective env-subset size.
    clock:
        Monotonic-ish time source (seconds) used to stamp wall-clock progress;
        injectable for deterministic tests. Defaults to :func:`time.monotonic`.
    run_start_s:
        The run's start time on the same scale as ``clock``; wall-clock progress
        is ``clock() - run_start_s``. Defaults to ``clock()`` at construction.
    persist:
        When ``True`` (default) produced captures are persisted via the store;
        set ``False`` to produce without persisting (the Orchestrator may batch).
    """

    def __init__(
        self,
        store: S3Store,
        *,
        capture_interval: float,
        capture_resolution: tuple[int, int],
        capture_env_subset_size: int,
        renderer: CaptureRenderer | None = None,
        total_envs: int | None = None,
        clock: Callable[[], float] = time.monotonic,
        run_start_s: float | None = None,
        persist: bool = True,
    ) -> None:
        self._store = store
        self._interval = float(capture_interval)
        self._resolution = (int(capture_resolution[0]), int(capture_resolution[1]))
        self._configured_subset = int(capture_env_subset_size)
        self._renderer = renderer
        self._total_envs = int(total_envs) if total_envs is not None else None
        self._clock = clock
        self._run_start_s = run_start_s if run_start_s is not None else clock()
        self._persist = persist
        # Highest interval bucket already captured; -1 means "nothing captured".
        self._last_bucket = -1

    # ------------------------------------------------------------------ #
    # Configuration-derived, pure helpers (public for testing)
    # ------------------------------------------------------------------ #
    @property
    def resolution(self) -> tuple[int, int]:
        """The configured capture resolution (Req 20.3)."""
        return self._resolution

    @property
    def env_subset_size(self) -> int:
        """The effective capture env-subset size, clamped to the live env count.

        Never exceeds the configured subset size, and never exceeds ``total_envs``
        when known — so a capture never expands to a full all-env render
        (Req 20.3, 20.5).
        """
        if self._total_envs is not None:
            return max(1, min(self._configured_subset, self._total_envs))
        return max(1, self._configured_subset)

    def should_capture(self, training_epoch: int) -> bool:
        """Whether a capture is due at ``training_epoch`` (Req 20.1, 20.5).

        Returns ``True`` the first time an epoch enters a new interval bucket and
        ``False`` otherwise, so captures occur only at the configured cadence and
        a single bucket never yields two captures.
        """
        if self._renderer is None:
            return False
        bucket = self._bucket_for(training_epoch)
        return bucket > self._last_bucket

    # ------------------------------------------------------------------ #
    # Production + persistence
    # ------------------------------------------------------------------ #
    def maybe_capture(
        self, iteration_index: int, training_epoch: int
    ) -> TrainingCapture | None:
        """Produce + persist a capture iff one is due (the ``capture_hook`` body).

        This is the callable wired into the PPO_Runner as ``capture_hook``. It
        enforces the Capture_Interval itself (Req 20.5) so it is safe to invoke
        every epoch: it only renders when a capture is actually due.
        """
        return self.capture_and_persist(iteration_index, training_epoch).capture

    def capture_and_persist(
        self, iteration_index: int, training_epoch: int
    ) -> CaptureResult:
        """Produce a due capture and persist it under the iteration path (Req 20.4).

        Returns a :class:`CaptureResult` describing what (if anything) was produced
        and the persistence outcome. No-op (empty result) when no capture is due
        or no renderer is available.
        """
        if not self.should_capture(training_epoch):
            return CaptureResult()

        capture = self._produce(iteration_index, training_epoch)
        if capture is None:
            # Renderer yielded no media; do not advance the bucket so a later
            # epoch in the same window can retry, and persist nothing.
            return CaptureResult()

        # Mark this interval bucket as captured so we don't duplicate within it.
        self._last_bucket = self._bucket_for(training_epoch)

        if not self._persist:
            return CaptureResult(capture=capture)

        result = self._store.put_training_capture(iteration_index, capture)
        return CaptureResult(capture=capture, persist=result)

    def as_hook(self) -> CaptureHook:
        """Return the bound ``capture_hook`` callable for the PPO_Runner."""
        return self.maybe_capture

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _bucket_for(self, training_epoch: int) -> int:
        """Map a training epoch to its interval bucket index.

        With a non-positive interval (defensive — Config validates ``> 0``) every
        epoch is its own bucket, so capture is attempted on each call.
        """
        if self._interval <= 0:
            return int(training_epoch)
        return int(training_epoch // self._interval)

    def _produce(
        self, iteration_index: int, training_epoch: int
    ) -> TrainingCapture | None:
        """Render media and assemble a tagged :class:`TrainingCapture` (Req 20.2).

        Passes only the configured resolution and the clamped env-subset size to
        the renderer, so capture never triggers a full-resolution all-env render
        (Req 20.3, 20.5). A renderer that raises is treated as "no capture" so the
        additive capture concern never aborts training.
        """
        renderer = self._renderer
        if renderer is None:
            return None

        try:
            output = renderer.render(
                iteration_index=iteration_index,
                training_epoch=training_epoch,
                resolution=self._resolution,
                env_subset_size=self.env_subset_size,
            )
        except Exception:  # noqa: BLE001 - capture is additive; never abort training
            return None

        screenshot_path, clip_path = _media_paths(output)
        if screenshot_path is None and clip_path is None:
            return None

        wall_clock_s = max(0.0, float(self._clock()) - float(self._run_start_s))
        return TrainingCapture(
            iteration_index=int(iteration_index),
            training_epoch=int(training_epoch),
            wall_clock_s=wall_clock_s,
            screenshot_path=screenshot_path,
            clip_path=clip_path,
        )


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _media_paths(output: Any) -> tuple[str | None, str | None]:
    """Extract ``(screenshot_path, clip_path)`` from a renderer's output.

    Accepts a :class:`RenderOutput`, any object exposing the same attributes, or
    a mapping with the same keys — keeping the renderer contract duck-typed for
    testability.
    """
    if output is None:
        return (None, None)
    if isinstance(output, dict):
        screenshot = output.get("screenshot_path")
        clip = output.get("clip_path")
    else:
        screenshot = getattr(output, "screenshot_path", None)
        clip = getattr(output, "clip_path", None)
    screenshot = str(screenshot) if screenshot else None
    clip = str(clip) if clip else None
    return (screenshot, clip)
