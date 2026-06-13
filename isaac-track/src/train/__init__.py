"""Training subpackage for the YOUR_REPO system.

Hosts the PPO_Runner (`src/train/ppo_runner.py`), which executes RSL-RL PPO
training inside Isaac Lab on the goal-conditioned H1 environment (design.md →
Components → PPO_Runner, Req 8, 14, 15, 20).
"""

from __future__ import annotations

from .ppo_runner import (
    PPORunner,
    TrainConfig,
    TrainResult,
    Trainer,
    checkpoint_epochs,
    select_cuda_visible_devices,
)

__all__ = [
    "PPORunner",
    "TrainConfig",
    "TrainResult",
    "Trainer",
    "checkpoint_epochs",
    "select_cuda_visible_devices",
]
