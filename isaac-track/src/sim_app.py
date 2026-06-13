"""Process-level Isaac Sim ``SimulationApp`` ownership (single instance per process).

Isaac Sim permits **exactly one** ``SimulationApp`` per Python process. Every
sim-touching component in this system needs one: the per-iteration trainer
(:class:`src.train.ppo_runner._IsaacLabTrainer`), the evaluation rollout, and the
demo-video recorder. If each launches its own, the second launch stalls forever
on the RTX rendering-kit reinit (observed live as the Stage C hang).

This module centralizes that ownership: :func:`get_sim_app` launches the app on
first call and returns the **same** instance on every subsequent call, for the
life of the process. The Eureka loop runs many iterations in one process, so the
app is launched once (iteration 0) and reused by every later iteration, by eval,
and by the recorder — never relaunched, never closed mid-run.

Isaac Lab is a lazy, guarded dependency (it only imports inside the Isaac Sim
runtime), so the ``AppLauncher`` import lives inside :func:`get_sim_app` and this
module stays importable on the controller host. The whole file is therefore
``# pragma: no cover`` for its live body; the pure bookkeeping (idempotency,
camera-flag latch) is what the controller-host tests exercise via a fake launcher.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["get_sim_app", "sim_app_launched", "reset_sim_app_for_tests"]


# Process-global handles. Populated on the first get_sim_app() call and reused
# thereafter; never reassigned (one SimulationApp per process).
_APP: Any = None
_LAUNCHER: Any = None
_ENABLE_CAMERAS: bool | None = None


def sim_app_launched() -> bool:
    """True iff the process-level ``SimulationApp`` has already been launched."""
    return _APP is not None


def get_sim_app(*, enable_cameras: bool | None = None, headless: bool = True) -> Any:
    """Return the process's one ``SimulationApp``, launching it on first call.

    The first call launches a headless ``SimulationApp`` (with ``enable_cameras``
    resolved as below) and caches it. Every subsequent call returns that same
    instance, ignoring the arguments — so the trainer, eval rollout, and recorder
    all share one app across all loop iterations.

    ``enable_cameras`` resolution (first call only):
      * explicit ``enable_cameras=`` argument wins;
      * else the ``HUMANOID_ENABLE_CAMERAS`` env var (``"0"`` -> off, anything
        else -> on);
      * default on. RTX camera rendering (demo video Req 10, training capture
        Req 20) needs it; it is cheap when unused.

    If a later call requests ``enable_cameras=True`` but the app was already
    launched with cameras off, that is a programming error (cameras can only be
    enabled at launch) — it is logged via a ``RuntimeError`` message string but
    NOT raised, because the established app is still usable for non-camera work;
    callers that strictly need cameras should ensure the first launch enables them
    (the trainer does when ``record_demo_video`` is on).
    """
    global _APP, _LAUNCHER, _ENABLE_CAMERAS  # noqa: PLW0603 - process singleton

    if _APP is not None:
        if enable_cameras and not _ENABLE_CAMERAS:
            # Cameras can only be turned on at launch; surface the mismatch
            # without tearing down the running app.
            print(
                "[sim_app] WARNING: get_sim_app(enable_cameras=True) but the "
                "process SimulationApp was already launched with cameras "
                "disabled; camera rendering will be unavailable this run."
            )
        return _APP

    if enable_cameras is None:
        enable_cameras = os.environ.get("HUMANOID_ENABLE_CAMERAS", "1") != "0"

    from isaaclab.app import AppLauncher  # noqa: PLC0415 - lazy, guarded

    _LAUNCHER = AppLauncher(headless=headless, enable_cameras=bool(enable_cameras))
    _APP = _LAUNCHER.app
    _ENABLE_CAMERAS = bool(enable_cameras)
    return _APP


def reset_sim_app_for_tests() -> None:
    """Clear the cached handles WITHOUT closing the app (unit-test hook only).

    Never call this in production: it does not close the SimulationApp (closing +
    relaunching in one process is exactly what is unsupported). It exists so a
    controller-host test can inject a fresh fake launcher between cases.
    """
    global _APP, _LAUNCHER, _ENABLE_CAMERAS  # noqa: PLW0603
    _APP = None
    _LAUNCHER = None
    _ENABLE_CAMERAS = None
