"""Standalone validation of the live demo-video recorder render loop.

Exercises ``IsaacLabDemoVideoRecorder.record`` end-to-end in its OWN process
(one SimulationApp, so no train/eval co-existence to worry about): it builds a
camera-instrumented goal-conditioned H1 env, loads a trained checkpoint, replays
one episode capturing the chase + side cameras' RGB, and muxes an .mp4 per
camera. This isolates the recorder's `# pragma: no cover` render path (camera
spawn -> data.output["rgb"] read -> imageio mux) so it can be shaken out
independently of the full Eureka loop.

Usage (inside the Isaac Lab container, --network host):
  isaaclab.sh -p scripts/validate_recorder.py \
      --checkpoint runs/checkpoints/model_final.pt \
      --output-dir runs/demo_videos \
      --num-envs 4 --max-steps 200

Prints VALIDATE_RECORDER: PASS/FAIL and the written video paths.
"""

from __future__ import annotations

import argparse
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Validate the demo-video recorder render loop")
    parser.add_argument("--checkpoint", default="runs/checkpoints/model_final.pt")
    parser.add_argument("--output-dir", default="runs/demo_videos")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args(argv)

    # (1) Launch the process-level SimulationApp with cameras enabled (required
    # for RTX RGB rendering), before importing any task module. Uses the shared
    # owner so this matches how the live loop obtains its app.
    from src.sim_app import get_sim_app  # noqa: PLC0415

    sim_app = get_sim_app(enable_cameras=True)

    rc = 1
    try:
        from src.data_models import Goal  # noqa: PLC0415
        from src.eval.evaluator import EpisodeRollout, IsaacLabDemoVideoRecorder  # noqa: PLC0415
        from src.sensors.camera_cfg import CameraConfig  # noqa: PLC0415

        goal = Goal(
            position_xy=(args.goal_x, args.goal_y),
            success_radius_m=args.success_radius,
        )
        cameras = CameraConfig(width=args.width, height=args.height, fps=args.fps).cameras()
        print(f"[validate] cameras: {[c.name for c in cameras]}")

        recorder = IsaacLabDemoVideoRecorder(
            goal=goal,
            task_id=args.task,
            eval_env_count=args.num_envs,
            max_episode_steps=args.max_steps,
        )

        # The producer normally supplies a rollout with a replay handle naming
        # the env index + goal + checkpoint to replay. Build one directly.
        rollout = EpisodeRollout(
            trajectory=None,
            replay={"env_index": 0, "goal": goal, "checkpoint": args.checkpoint},
        )

        paths = recorder.record(
            rollout=rollout,
            cameras=cameras,
            label="validate_best",
            output_dir=args.output_dir,
        )
        paths = list(paths)
        print(f"[validate] recorder returned {len(paths)} path(s): {paths}")

        # Verify the files exist and are non-empty.
        ok = bool(paths)
        for p in paths:
            size = os.path.getsize(p) if os.path.exists(p) else 0
            print(f"[validate]   {p} -> {size} bytes")
            if size <= 0:
                ok = False
        rc = 0 if ok else 1
        print(f"VALIDATE_RECORDER: {'PASS' if rc == 0 else 'FAIL'}")
    except Exception:  # noqa: BLE001 - report any render-loop failure verbatim
        traceback.print_exc()
        print("VALIDATE_RECORDER: FAIL")
        rc = 1
    finally:
        try:
            sim_app.close()
        except Exception:  # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
