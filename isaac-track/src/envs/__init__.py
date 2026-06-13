"""Goal-conditioned Isaac Lab environment wiring for YOUR_REPO.

This package reframes the stock manager-based H1 flat task
(``Isaac-Velocity-Flat-H1-v0``) as a point-to-point goal-reaching task
(design.md → "Introducing the Goal and Goal_Observation into the environment").

The Isaac Lab touchpoints are lazy/guarded so the package imports cleanly on the
controller/dev host without ``isaaclab``; the robot-frame Goal_Observation
transform and the per-env Goal buffer logic are pure and unit-testable with
synthetic tensors / fakes.
"""

from __future__ import annotations

from src.envs.goal_env import (
    GOAL_OBS_DIM,
    STOCK_VELOCITY_REWARD_TERMS,
    GoalBuffer,
    compute_goal_observation,
    disable_velocity_tracking_rewards,
    make_goal_buffer,
    yaw_from_quat,
)

__all__ = [
    "GOAL_OBS_DIM",
    "STOCK_VELOCITY_REWARD_TERMS",
    "GoalBuffer",
    "compute_goal_observation",
    "disable_velocity_tracking_rewards",
    "make_goal_buffer",
    "yaw_from_quat",
]
