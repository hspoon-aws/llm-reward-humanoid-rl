"""Per-iteration demo-video recorder — run as a FRESH SUBPROCESS.

Isaac Lab cannot create a second manager-based RL env in a process that already
built one (the in-loop recorder stalls during the second env's scene/sensor
setup, even after closing the first env and sharing the SimulationApp). The
robust fix is to record in a **fresh process**: one SimulationApp, one env — the
exact configuration the standalone validation proved works.

The Evaluator's :class:`SubprocessDemoRecorder` invokes this script via
``isaaclab.sh -p`` after an iteration writes its checkpoint. It loads that
checkpoint, builds a SINGLE-env camera scene (chase + side world-frame cameras
aimed at the A->goal path), replays one episode, and muxes one .mp4 per camera
into ``--output-dir``. It prints ``RECORD_DEMO: PASS <json paths>`` on success so
the parent can collect the paths.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Record a demo video for one checkpoint")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--label", default="best")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=400)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    # (1) Process-level SimulationApp with cameras enabled, before task imports.
    from src.sim_app import get_sim_app  # noqa: PLC0415

    sim_app = get_sim_app(enable_cameras=True)

    rc = 1
    paths: list[str] = []
    try:
        from src.data_models import Goal  # noqa: PLC0415
        from src.eval.evaluator import EpisodeRollout, IsaacLabDemoVideoRecorder  # noqa: PLC0415
        from src.sensors.camera_cfg import CameraConfig  # noqa: PLC0415

        goal = Goal(position_xy=(args.goal_x, args.goal_y), success_radius_m=args.success_radius)
        cameras = CameraConfig(width=args.width, height=args.height, fps=args.fps).cameras()

        recorder = IsaacLabDemoVideoRecorder(
            goal=goal,
            task_id=args.task,
            eval_env_count=args.num_envs,
            max_episode_steps=args.max_steps,
        )
        rollout = EpisodeRollout(
            trajectory=None,
            replay={"env_index": 0, "goal": goal, "checkpoint": args.checkpoint},
        )
        paths = list(
            recorder.record(
                rollout=rollout,
                cameras=cameras,
                label=args.label,
                output_dir=args.output_dir,
            )
        )
        ok = bool(paths) and all(
            os.path.exists(p) and os.path.getsize(p) > 0 for p in paths
        )
        rc = 0 if ok else 1
        print(f"RECORD_DEMO: {'PASS' if rc == 0 else 'FAIL'} {json.dumps(paths)}")
    except Exception:  # noqa: BLE001 - report any failure to the parent
        traceback.print_exc()
        print("RECORD_DEMO: FAIL []")
        rc = 1
    finally:
        try:
            sim_app.close()
        except Exception:  # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
