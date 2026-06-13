#!/usr/bin/env python3
"""Evaluator rollout + headless demo-render smoke test (Phase 5).

Runs the MJX policy rollout to produce EpisodeTrajectory data, computes
EvalMetrics + staged gates from it (the framework-agnostic metric math), and
records a short demo video via MuJoCo's headless EGL Renderer. Uses a small
episode budget so it finishes quickly.

Run inside the MJX venv on the B200:
    MUJOCO_GL=egl /data/mjxvenv/bin/python scripts/smoke_test_eval.py
"""
from __future__ import annotations

import sys
from dataclasses import dataclass


@dataclass
class _EvalCfg:
    env_name: str = "H1JoystickGaitTracking"
    fall_threshold_s: float = 3.0
    episode_length: int = 60
    include_secondary: bool = True
    selection_metric: str = "success_rate"


def main() -> int:
    sys.path.insert(0, "/data/mujoco")
    from src.data_models import Goal
    from src.eval.evaluator import compute_eval_metrics
    from src.eval.evaluator import GateThresholds
    from src.eval.mjx_rollout import MjxPolicyRollout, MjxDemoVideoRecorder
    from src.sensors.camera_cfg import CameraConfig

    goal = Goal(position_xy=(5.0, 0.0), success_radius_m=0.5)
    cfg = _EvalCfg()

    # --- rollout -> trajectories ---
    rollout = MjxPolicyRollout(goal, cfg)
    episodes = rollout.rollout(checkpoint=None, goal=goal, num_episodes=4)
    print(f"[eval] rolled out {len(episodes)} episodes")
    trajs = [e.trajectory for e in episodes]
    print(f"[eval] episode 0: {trajs[0].num_steps} steps, "
          f"start={trajs[0].positions_xy[0]}, end={trajs[0].positions_xy[-1]}")

    # --- metric math (framework-agnostic) ---
    metrics = compute_eval_metrics(
        trajs, goal, include_secondary=True,
        gate_thresholds=GateThresholds(
            min_progress_distance_m=1.0,
            time_to_goal_threshold_s=12.0,
            path_efficiency_threshold=0.6,
        ),
    )
    print(f"[eval] success_rate={metrics.success_rate} "
          f"dist_to_goal={metrics.distance_to_goal_m:.3f} "
          f"upright_s={metrics.upright_time_s:.3f} fall_rate={metrics.fall_rate}")
    print(f"[eval] gates: makes_progress={metrics.gates.makes_progress} "
          f"reaches_goal={metrics.gates.reaches_goal} "
          f"efficient_goal={metrics.gates.efficient_goal}")
    # JSON round-trip (Req 9.7)
    js = metrics.to_json()
    assert len(js) > 0
    print(f"[eval] EvalMetrics JSON serialized ({len(js)} chars)")

    # --- headless demo video render (EGL) ---
    cams = CameraConfig(width=320, height=240, fps=20)
    recorder = MjxDemoVideoRecorder(goal, cfg)
    import os
    out_dir = "/data/mujoco/runs/smoke"
    os.makedirs(out_dir, exist_ok=True)
    paths = recorder.record(
        rollout=episodes[0], cameras=cams.cameras(), label="best", output_dir=out_dir
    )
    ok = all(os.path.exists(p) and os.path.getsize(p) > 0 for p in paths) and bool(paths)
    print(f"[eval] demo videos written: {paths}")
    if not ok:
        print("[FAIL] demo video missing/empty")
        return 1

    print("=== EVAL SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
