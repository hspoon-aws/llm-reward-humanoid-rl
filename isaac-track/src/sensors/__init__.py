"""External render cameras for demo-video capture.

See ``src/sensors/camera_cfg.py`` (Camera_Config, Task 12, Req 10).

Only the Isaac-free camera factory + spec are re-exported here; importing this
package does NOT pull in Isaac Lab. The factory functions return inert
:class:`CameraSpec` descriptions of external, world-frame (``/World``) RGB
cameras that the Evaluator instantiates ONLY at eval/play time — never in the
parallel training run and never as policy inputs (Req 10.4).
"""

from src.sensors.camera_cfg import (
    CHASE_CAMERA_PRIM_PATH,
    DEFAULT_FPS,
    DEFAULT_HEIGHT,
    DEFAULT_WIDTH,
    RGB_DATA_TYPE,
    SIDE_CAMERA_PRIM_PATH,
    WORLD_PRIM_ROOT,
    CameraConfig,
    CameraSpec,
)

__all__ = [
    "CameraConfig",
    "CameraSpec",
    "RGB_DATA_TYPE",
    "WORLD_PRIM_ROOT",
    "CHASE_CAMERA_PRIM_PATH",
    "SIDE_CAMERA_PRIM_PATH",
    "DEFAULT_WIDTH",
    "DEFAULT_HEIGHT",
    "DEFAULT_FPS",
]
