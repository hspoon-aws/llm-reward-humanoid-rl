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
    # Pinhole focal length in mm against Isaac's default ~20.955 mm horizontal
    # aperture. Smaller => WIDER field of view. The demo cameras use a short
    # focal length so the whole A->Goal walk stays in frame as the robot moves,
    # rather than tracking it (these are fixed world-frame cameras). Isaac's
    # default is 24 mm (fairly tele); 12 mm roughly doubles the FOV.
    focal_length: float = 12.0

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

    def _look_at_quat_wxyz(self) -> tuple[float, float, float, float]:
        """Orientation quaternion (w, x, y, z) aiming at ``look_at``, WORLD convention.

        Isaac Lab's ``CameraCfg.OffsetCfg(convention="world")`` defines the camera
        with **forward = +X** and **up = +Z** in the camera's rotated frame (per
        ``camera_cfg.py``). So the rotation we need maps the camera's local axes
        to a basis where local +X points from ``position`` toward ``look_at``,
        local +Z is the world-up-aligned camera up, and local +Y completes a
        right-handed frame. (An earlier version used the OpenGL/ROS -Z-forward
        convention, which is why the robot landed at the frame edge.)

        Pure math (no torch/Isaac): build the basis and convert to a quaternion.
        """
        import math  # noqa: PLC0415

        px, py, pz = self.position
        tx, ty, tz = self.look_at
        # Local +X = forward (camera -> target).
        fx, fy, fz = (tx - px, ty - py, tz - pz)
        fn = math.sqrt(fx * fx + fy * fy + fz * fz) or 1.0
        fx, fy, fz = (fx / fn, fy / fn, fz / fn)

        # World up; if forward is near-vertical, fall back to +X as the up hint.
        wux, wuy, wuz = (0.0, 0.0, 1.0)
        if abs(fx * wux + fy * wuy + fz * wuz) > 0.999:
            wux, wuy, wuz = (1.0, 0.0, 0.0)

        # Local +Y = up x forward  (right-handed: X=fwd, Y=left/up-plane, Z=up).
        # Compute Z (up) first as forward-orthogonalized world up, then Y = Z x X.
        # up_proj = world_up - (world_up . fwd) fwd
        d = wux * fx + wuy * fy + wuz * fz
        uzx, uzy, uzz = (wux - d * fx, wuy - d * fy, wuz - d * fz)
        un = math.sqrt(uzx * uzx + uzy * uzy + uzz * uzz) or 1.0
        uzx, uzy, uzz = (uzx / un, uzy / un, uzz / un)  # local +Z (camera up)
        # local +Y = Z x X (right-handed)
        yx = uzy * fz - uzz * fy
        yy = uzz * fx - uzx * fz
        yz = uzx * fy - uzy * fx

        # Rotation matrix columns are the local axes expressed in world:
        #   col0 (+X) = forward, col1 (+Y) = yx.., col2 (+Z) = up
        m00, m01, m02 = fx, yx, uzx
        m10, m11, m12 = fy, yy, uzy
        m20, m21, m22 = fz, yz, uzz

        trace = m00 + m11 + m22
        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            w = 0.25 * s
            x = (m21 - m12) / s
            y = (m02 - m20) / s
            z = (m10 - m01) / s
        elif m00 > m11 and m00 > m22:
            s = math.sqrt(1.0 + m00 - m11 - m22) * 2.0
            w = (m21 - m12) / s
            x = 0.25 * s
            y = (m01 + m10) / s
            z = (m02 + m20) / s
        elif m11 > m22:
            s = math.sqrt(1.0 + m11 - m00 - m22) * 2.0
            w = (m02 - m20) / s
            x = (m01 + m10) / s
            y = 0.25 * s
            z = (m12 + m21) / s
        else:
            s = math.sqrt(1.0 + m22 - m00 - m11) * 2.0
            w = (m10 - m01) / s
            x = (m02 + m20) / s
            y = (m12 + m21) / s
            z = 0.25 * s
        qn = math.sqrt(w * w + x * x + y * y + z * z) or 1.0
        return (w / qn, x / qn, y / qn, z / qn)

    def to_isaaclab_cfg(self) -> Any:
        """Build the real Isaac Lab ``CameraCfg`` from this spec.

        Imported lazily so this module stays importable without Isaac Sim. The
        resulting camera is a USD prim under ``/World`` (external, world-frame),
        renders only the ``rgb`` product at the configured resolution, and is
        scheduled at ``update_period`` (== ``1 / fps``). The camera is **aimed**
        at ``look_at`` via a computed rotation quaternion (Isaac's OffsetCfg has
        no look_at), so the robot is actually in frame.

        Raises:
            ImportError: If Isaac Lab is not importable (i.e. not running inside
                the Isaac Sim runtime). Callers on the controller host should use
                the :class:`CameraSpec` attributes directly instead.
        """
        # Lazy, guarded imports: only valid inside the Isaac Sim runtime.
        from isaaclab.sensors import CameraCfg  # noqa: PLC0415
        from isaaclab.sim import PinholeCameraCfg  # noqa: PLC0415
        import isaaclab.sim as sim_utils  # noqa: PLC0415

        return CameraCfg(
            prim_path=self.prim_path,
            update_period=self.update_period,
            height=self.height,
            width=self.width,
            data_types=list(self.data_types),
            spawn=PinholeCameraCfg(focal_length=float(self.focal_length)),
            offset=CameraCfg.OffsetCfg(
                pos=self.position,
                # Aim the camera at look_at (world-frame). Without rot the camera
                # keeps its default orientation and never frames the robot.
                rot=self._look_at_quat_wxyz(),
                convention="world",
            ),
        )


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

        Placed well behind and above the start (point A near the origin),
        looking toward the **midpoint** of the A→Goal path at torso height
        (~1 m), with a WIDE field of view (short focal length). The wide lens +
        generous standoff keep the **whole walk** (start → goal at x≈5) in frame
        as the robot moves, since this is a fixed world-frame camera that does
        not track. The prim lives under ``/World`` and renders RGB at the
        configured resolution/fps.
        """
        return CameraSpec(
            name="chase",
            prim_path=CHASE_CAMERA_PRIM_PATH,
            width=self.width,
            height=self.height,
            fps=self.fps,
            position=(-5.0, 0.0, 2.2),
            look_at=(2.5, 0.0, 1.0),
            data_types=(RGB_DATA_TYPE,),
            focal_length=12.0,
        )

    def side_camera(self) -> CameraSpec:
        """External world-frame side-view camera (Req 10.2, 10.4).

        A side view offset along ``-y`` at torso height looking across the
        A→Goal path midpoint, with a WIDE field of view (short focal length), so
        gait and posture are visible in profile and the **entire** A→Goal path
        (x≈0→5) fits in frame as the robot walks across. The standoff (~8 m to
        the side) plus the wide lens keep the robot framed end-to-end without
        tracking. The prim lives under ``/World`` and renders RGB at the
        configured resolution/fps.
        """
        return CameraSpec(
            name="side",
            prim_path=SIDE_CAMERA_PRIM_PATH,
            width=self.width,
            height=self.height,
            fps=self.fps,
            position=(2.5, -8.0, 1.8),
            look_at=(2.5, 0.0, 1.0),
            data_types=(RGB_DATA_TYPE,),
            focal_length=12.0,
        )

    def cameras(self) -> tuple[CameraSpec, CameraSpec]:
        """Both demo cameras as a ``(chase, side)`` tuple for convenience."""
        return (self.chase_camera(), self.side_camera())
