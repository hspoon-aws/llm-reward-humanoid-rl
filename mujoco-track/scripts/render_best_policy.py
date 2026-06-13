#!/usr/bin/env python3
"""End-only demo render: roll out a TRAINED best-policy checkpoint and record
best/worst demo videos via MuJoCo's headless EGL Renderer.

Designed to run AFTER a run finishes (the loop itself records no per-iteration
video — see lesson §6/§10). It reuses the exact eval rollout + recorder the
Evaluator would use, but drives them directly from a checkpoint path so it can be
pointed at any finished run's best policy without disturbing a live loop.

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/render_best_policy.py \
        --checkpoint /data/mujoco/runs/run-1gpu-v4/iter_03_policy.pkl \
        --out-dir   /data/mujoco/runs/run-1gpu-v4/demo \
        --episodes 8

The checkpoint must have its ``<path>.netcfg.json`` sidecar next to it (written
by _MjxTrainer.save_checkpoint); without it the policy can't be reconstructed and
the rollout falls back to a zero-action policy.
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass
class _EvalCfg:
    env_name: str = "H1JoystickGaitTracking"
    fall_threshold_s: float = 3.0
    episode_length: int = 1000
    include_secondary: bool = True
    selection_metric: str = "success_rate"


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Render best/worst demo video for a trained checkpoint")
    parser.add_argument("--checkpoint", required=True, help="path to <iter>_policy.pkl (needs .netcfg.json sidecar)")
    parser.add_argument("--out-dir", required=True, help="directory to write the demo .mp4 files")
    parser.add_argument("--episodes", type=int, default=8, help="episodes to roll out for best/worst selection")
    parser.add_argument("--env-name", default="H1JoystickGaitTracking")
    parser.add_argument("--goal-x", type=float, default=5.0)
    parser.add_argument("--goal-y", type=float, default=0.0)
    parser.add_argument("--success-radius", type=float, default=0.5)
    parser.add_argument("--selection-metric", default="success_rate")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args(argv)

    sys.path.insert(0, "/data/mujoco")
    from src.data_models import Goal
    from src.eval.evaluator import DemoVideoProducer
    from src.eval.mjx_rollout import MjxPolicyRollout, MjxDemoVideoRecorder
    from src.sensors.camera_cfg import CameraConfig

    netcfg = args.checkpoint + ".netcfg.json"
    if not os.path.exists(args.checkpoint):
        print(f"[render] FAIL: checkpoint not found: {args.checkpoint}", file=sys.stderr)
        return 2
    if not os.path.exists(netcfg):
        print(f"[render] WARN: netcfg sidecar missing ({netcfg}); policy will be zero-action", file=sys.stderr)

    goal = Goal(position_xy=(args.goal_x, args.goal_y), success_radius_m=args.success_radius)
    cfg = _EvalCfg(env_name=args.env_name, selection_metric=args.selection_metric)
    os.makedirs(args.out_dir, exist_ok=True)

    # Reuse the Evaluator's best/worst selection + recorder, but feed it the
    # checkpoint path directly. DemoVideoProducer rolls out `episodes` episodes,
    # scores them by the selection metric, and records the best and worst.
    rollout = MjxPolicyRollout(goal, cfg)
    recorder = MjxDemoVideoRecorder(goal, cfg)
    cams = CameraConfig(width=args.width, height=args.height, fps=args.fps)

    producer = DemoVideoProducer(
        cams,
        rollout=rollout,
        recorder=recorder,
        selection_metric=args.selection_metric,
        num_episodes=args.episodes,
        output_dir=args.out_dir,
    )
    result = producer.record_best_worst(checkpoint=args.checkpoint, goal=goal)

    best = getattr(result, "best", None)
    worst = getattr(result, "worst", None)
    best_paths = list(getattr(best, "video_paths", []) or []) if best else []
    worst_paths = list(getattr(worst, "video_paths", []) or []) if worst else []
    all_paths = best_paths + worst_paths
    print(f"[render] best videos:  {best_paths}")
    print(f"[render] worst videos: {worst_paths}")

    ok = bool(all_paths) and all(os.path.exists(p) and os.path.getsize(p) > 0 for p in all_paths)
    if not ok:
        print("[render] FAIL: no non-empty demo videos were produced", file=sys.stderr)
        return 1
    print(f"[render] OK: {len(all_paths)} video file(s) under {args.out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
