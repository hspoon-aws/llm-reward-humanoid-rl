"""Per-iteration eval + demo-video recording — run as a FRESH SUBPROCESS.

Isaac Lab cannot create a second manager-based RL env in a process that already
built one. The training process builds the trainer's env; any in-process eval or
recorder that builds a second env stalls. The robust design (mirroring Isaac's
standard train.py/play.py split) is to run the ENTIRE eval+record phase in a
fresh process from the checkpoint.

Critically, this does a SINGLE camera-equipped env rollout that produces BOTH:
  * the per-step trajectory (base xy + upright) -> EvalMetrics, and
  * the per-step camera RGB -> demo videos,
so there is only ONE env in the process (no second-env stall).

Emits one JSON line: ``EVAL_RESULT: {"metrics": {...}, "videos": {...}}`` that the
parent :class:`SubprocessEvaluator` parses. Fail-soft: any error prints
``EVAL_RESULT: {}`` and a traceback.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Subprocess eval + demo recording for one checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--task", default="Isaac-Velocity-Flat-H1-v0")
    p.add_argument("--goal-x", type=float, default=5.0)
    p.add_argument("--goal-y", type=float, default=0.0)
    p.add_argument("--success-radius", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--record", type=int, default=1)  # 1 = also write videos
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=float, default=30.0)
    # gate thresholds (so metrics gates match the run config)
    p.add_argument("--min-progress-distance-m", type=float, default=1.0)
    p.add_argument("--time-to-goal-threshold-s", type=float, default=12.0)
    p.add_argument("--path-efficiency-threshold", type=float, default=0.6)
    p.add_argument("--success-radius-gate", type=float, default=None)
    args = p.parse_args(argv)

    # (1) Fresh process-level SimulationApp with cameras enabled.
    from src.sim_app import get_sim_app  # noqa: PLC0415

    sim_app = get_sim_app(enable_cameras=True)

    result: dict = {}
    try:
        from src.data_models import Goal  # noqa: PLC0415
        from src.eval.evaluator import (  # noqa: PLC0415
            GateThresholds,
            IsaacLabDemoVideoRecorder,
            compute_eval_metrics,
        )
        from src.sensors.camera_cfg import CameraConfig  # noqa: PLC0415

        goal = Goal(position_xy=(args.goal_x, args.goal_y), success_radius_m=args.success_radius)
        cameras = CameraConfig(width=args.width, height=args.height, fps=args.fps).cameras()

        recorder = IsaacLabDemoVideoRecorder(
            goal=goal, task_id=args.task, eval_env_count=1, max_episode_steps=args.max_steps,
        )

        # SINGLE camera-equipped env pass -> video frames + trajectory.
        video_paths, trajectory = recorder.record_with_trajectory(
            goal=goal,
            checkpoint=args.checkpoint,
            cameras=cameras,
            label="best",
            output_dir=args.output_dir,
            env_index=0,
        )

        if trajectory is not None:
            thresholds = GateThresholds(
                min_progress_distance_m=args.min_progress_distance_m,
                time_to_goal_threshold_s=args.time_to_goal_threshold_s,
                path_efficiency_threshold=args.path_efficiency_threshold,
            )
            metrics = compute_eval_metrics(
                [trajectory], goal, include_secondary=False, gate_thresholds=thresholds,
            )
            result["metrics"] = json.loads(metrics.to_json())

        if args.record and video_paths:
            # name -> path; the parent maps these into the iteration artifacts.
            result["videos"] = {
                os.path.splitext(os.path.basename(pth))[0]: pth for pth in video_paths
            }

        print("EVAL_RESULT: " + json.dumps(result))
        rc = 0 if "metrics" in result else 1
    except Exception:  # noqa: BLE001
        traceback.print_exc()
        print("EVAL_RESULT: {}")
        rc = 1
    finally:
        try:
            sim_app.close()
        except Exception:  # noqa: BLE001
            pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
