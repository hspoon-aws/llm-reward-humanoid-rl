"""Camera_Config — external world-frame chase and side render cameras.

This module owns the camera configuration for the YOUR_REPO
system (design.md → Components and Interfaces → Camera_Config (`src/sensors/
camera_cfg.py`), Req 10).

What this provides (Task 12.1, Req 10.2, 10.4)
----------------------------------------------
* :class:`CameraConfig` — a small factory exposing :meth:`CameraConfig.chase_camera`
  and :meth:`CameraConfig.side_camera`. Each returns an **external, world-frame**
  camera config whose prim lives under ``/World`` (NOT attached to the robot —
  ``H1_MINIMAL_CFG`` carries no camera). Each camera produces **RGB** frames at the
  configured resolution and frame rate (Req 10.2).
* :class:`CameraSpec` — the pure-data description of one camera (prim path,
  width, height, fps, RGB data type, world-frame pose). It is always importable
  and inspectable on the controller host without Isaac Lab, so the config values
  can be verified independently of the simulator.

Eval/play only (Req 10.4)
-------------------------
These cameras are a guardrail-documented eval-time concern. They are meant to be
instantiated **only** during evaluation / demo-video capture with a small env
count (~50) by the Evaluator (Task 11.4). They are **never** created in the
parallel (4096-env) training run and are **never** provided as policy inputs.
Nothing in this module touches the training env build or the observation space;
the factory functions simply return inert config objects that a caller chooses to
instantiate at play time.

Isaac Lab is an optional, lazy dependency
-----------------------------------------
Isaac Lab's ``CameraCfg`` / ``PinholeCameraCfg`` are only importable inside the
Isaac Sim runtime, which is absent on the controller host (and in CI). Importing
this module must therefore NOT require Isaac Lab. The factory methods return a
:class:`CameraSpec` by default — a torch/Isaac-free dataclass that exposes the
width, height, fps, RGB data type, and ``/World`` prim path. When Isaac Lab IS
importable, :meth:`CameraSpec.to_isaaclab_cfg` builds the real
``isaaclab.sensors.CameraCfg`` from the same values, so the spec is the single
source of truth shared by both code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "RGB_DATA_TYPE",
    "WORLD_PRIM_ROOT",
    "CHASE_CAMERA_PRIM_PATH",
    "SIDE_CAMERA_PRIM_PATH",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_FPS",
    "CameraSpec",
    "CameraConfig",
]


# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# The single RGB modality these demo cameras produce (Req 10.2). Kept as a
# one-tuple so a caller could, in principle, request additional render products
# at play time without changing the contract that RGB is always present.
RGB_DATA_TYPE = "rgb"

# All Camera_Config prims live under the world, never under the robot prim
# (Req 10.4: external, world-frame cameras; H1_MINIMAL_CFG has no camera).
WORLD_PRIM_ROOT = "/World"
CHASE_CAMERA_PRIM_PATH = f"{WORLD_PRIM_ROOT}/chase_camera"
SIDE_CAMERA_PRIM_PATH = f"{WORLD_PRIM_ROOT}/side_camera"

# Documented defaults mirror the ``video`` section of config/run_config.yaml.
DEFAULT_WIDTH = 1920
DEFAULT_HEIGHT = 1080
DEFAULT_FPS = 30.0


# --------------------------------------------------------------------------- #
# Pure-data camera description
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CameraSpec:
    """Immutable, Isaac-free description of one external world-frame camera.

    Holds exactly the values the Camera_Config contract is defined over (Req
    10.2): a ``/World`` prim path, RGB data type, the configured resolution
    (``width`` × ``height``), and the frame rate (``fps``). ``position`` and
    ``look_at`` express the world-frame placement; the chase camera is offset
    behind/above the env origin (point A) looking toward the goal direction,
    and the side camera is offset to the side looking across the path.

    This object is intentionally inert: constructing or inspecting it never
    touches Isaac Lab, so the config values are verifiable on any host. Call
    :meth:`to_isaaclab_cfg` inside the Isaac Sim runtime to obtain the real
    ``CameraCfg`` for instantiation at eval/play time.
    """

    name: str
    prim_path: str
    width: int
    height: int
    fps: float
    position: tuple[float, float, float]
    look_at: tuple[float, float, float]
    data_types: tuple[str, ...] = (RGB_DATA_TYPE,)

    @property
    def update_period(self) -> float:
        """Seconds between rendered frames (``1 / fps``).

        Isaac Lab's ``CameraCfg`` schedules capture by ``update_period`` rather
        than by frames-per-second, so this converts the configured frame rate
        into the value the simulator expects (Req 10.2).
        """
        return 1.0 / self.fps

    def is_world_frame(self) -> bool:
        """True iff this camera's prim lives under ``/World`` (Req 10.4).

        External, world-frame cameras must not be parented under the robot prim.
        """
        return self.prim_path == WORLD_PRIM_ROOT or self.prim_path.startswith(
            WORLD_PRIM_ROOT + "/"
        )

    def to_mujoco_camera(self) -> Any:
        """Build a MuJoCo free-camera positioned/aimed per this spec.

        MuJoCo-track replacement for the Isaac ``to_isaaclab_cfg``. Returns a
        ``mujoco.MjvCamera`` placed in world coordinates at ``position`` and
        aimed at ``look_at`` (free camera; not attached to the robot — the H1
        env carries no on-robot camera). Imported lazily so this module stays
        importable without MuJoCo on the controller host.

        The Evaluator uses this with ``mujoco.Renderer`` (EGL headless on the
        B200, verified) to record best/worst demo episodes (Req 10.2-10.4).

        Raises:
            ImportError: If MuJoCo is not importable (controller host). Use the
                :class:`CameraSpec` attributes directly instead.
        """
        import math as _math  # noqa: PLC0415

        import mujoco  # noqa: PLC0415

        cam = mujoco.MjvCamera()
        cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        px, py, pz = self.position
        lx, ly, lz = self.look_at
        cam.lookat[:] = [lx, ly, lz]
        dx, dy, dz = px - lx, py - ly, pz - lz
        cam.distance = _math.sqrt(dx * dx + dy * dy + dz * dz) or 1.0
        # azimuth/elevation (degrees) of the camera relative to the look-at point
        cam.azimuth = _math.degrees(_math.atan2(dy, dx))
        horiz = _math.sqrt(dx * dx + dy * dy) or 1e-9
        cam.elevation = _math.degrees(_math.atan2(dz, horiz)) * -1.0
        return cam


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
@dataclass
class CameraConfig:
    """Factory for the external, world-frame demo cameras (Req 10.2, 10.4).

    Resolution and frame rate are sourced from the run configuration (the
    ``video`` section of ``config/run_config.yaml``); :meth:`from_config` reads
    them off a loaded config object/mapping when present, otherwise the
    documented defaults (1920×1080 @ 30 fps) apply.

    Both :meth:`chase_camera` and :meth:`side_camera` return a :class:`CameraSpec`
    describing a ``/World`` camera that produces RGB at ``width`` × ``height`` and
    ``fps``. These specs are inert: they are instantiated into live cameras only
    by the Evaluator at eval/play time with a small env count, never in the
    parallel training run and never as policy inputs (Req 10.4).
    """

    width: int = DEFAULT_WIDTH
    height: int = DEFAULT_HEIGHT
    fps: float = DEFAULT_FPS

    def __post_init__(self) -> None:
        # Fail loudly on nonsensical render settings rather than producing a
        # camera that renders nothing or schedules an infinite frame period.
        if not isinstance(self.width, int) or isinstance(self.width, bool) or self.width < 1:
            raise ValueError(f"width must be a positive integer, got {self.width!r}")
        if not isinstance(self.height, int) or isinstance(self.height, bool) or self.height < 1:
            raise ValueError(f"height must be a positive integer, got {self.height!r}")
        fps_ok = isinstance(self.fps, (int, float)) and not isinstance(self.fps, bool)
        if not fps_ok or not self.fps > 0:
            raise ValueError(f"fps must be a positive number, got {self.fps!r}")
        self.fps = float(self.fps)

    @classmethod
    def from_config(cls, config: Any) -> "CameraConfig":
        """Build a :class:`CameraConfig` from a loaded run config.

        Accepts either a mapping with a ``video`` section (the raw YAML layout)
        or an object exposing ``video_width`` / ``video_height`` / ``video_fps``
        attributes. Missing values fall back to the documented defaults so the
        factory always yields a usable camera config (Req 10.2).
        """
        width, height, fps = DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_FPS

        video: Mapping[str, Any] | None = None
        if isinstance(config, Mapping):
            section = config.get("video")
            if isinstance(section, Mapping):
                video = section
        if video is not None:
            width = video.get("width", width)
            height = video.get("height", height)
            fps = video.get("fps", fps)
        else:
            # Attribute-style access for a typed config object that may carry
            # video fields (defensive; current Config does not, so defaults hold).
            width = getattr(config, "video_width", width)
            height = getattr(config, "video_height", height)
            fps = getattr(config, "video_fps", fps)

        return cls(width=int(width), height=int(height), fps=float(fps))

    def chase_camera(self) -> CameraSpec:
        """External world-frame chase camera (Req 10.2, 10.4).

        Placed behind and above the env origin (point A), looking toward the
        goal direction, to follow the robot's progress across the path. The prim
        lives under ``/World`` and renders RGB at the configured resolution/fps.
        """
        return CameraSpec(
            name="chase",
            prim_path=CHASE_CAMERA_PRIM_PATH,
            width=self.width,
            height=self.height,
            fps=self.fps,
            position=(-3.0, 0.0, 2.0),
            look_at=(5.0, 0.0, 0.5),
            data_types=(RGB_DATA_TYPE,),
        )

    def side_camera(self) -> CameraSpec:
        """External world-frame side-view camera (Req 10.2, 10.4).

        A fixed side view offset along ``+y`` looking across the A→goal path, so
        gait and posture are visible in profile. The prim lives under ``/World``
        and renders RGB at the configured resolution/fps.
        """
        return CameraSpec(
            name="side",
            prim_path=SIDE_CAMERA_PRIM_PATH,
            width=self.width,
            height=self.height,
            fps=self.fps,
            position=(2.5, -5.0, 1.5),
            look_at=(2.5, 0.0, 0.5),
            data_types=(RGB_DATA_TYPE,),
        )

    def cameras(self) -> tuple[CameraSpec, CameraSpec]:
        """Both demo cameras as a ``(chase, side)`` tuple for convenience."""
        return (self.chase_camera(), self.side_camera())
