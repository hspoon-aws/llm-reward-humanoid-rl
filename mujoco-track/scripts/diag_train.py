"""Direct train/eval diagnostic: run one tiny PPO train with a trivial reward
and print the FULL traceback so we can locate the tracer-leak InvalidInputException."""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import load_config
from src.rewards.reward_executor import RewardExecutor, SandboxConfig
from src.train.mjx_trainer import PPORunner, TrainConfig
from src.orchestrator import GoalRef, Goal

SIMPLE_REWARD = '''
def compute_reward(obs, action, next_obs, goal):
    import jax.numpy as jnp
    # distance-to-goal shaped reward; all jittable, no python control flow
    pos = next_obs[..., 0:2]
    gx = goal["goal_x"]
    gy = goal["goal_y"]
    dx = pos[..., 0] - gx
    dy = pos[..., 1] - gy
    dist = jnp.sqrt(dx * dx + dy * dy + 1e-8)
    alive = jnp.ones_like(dist)
    return alive - 0.1 * dist
'''


def main() -> int:
    cfg = load_config("config/run_config.yaml")
    import dataclasses
    cfg = dataclasses.replace(cfg, train_epochs=2, num_envs=256, checkpoint_interval=2)

    goal = Goal(position_xy=cfg.goal_position, success_radius_m=cfg.success_radius_m)
    goal_ref = GoalRef(goal=goal)

    ex = RewardExecutor(SandboxConfig(time_limit_s=cfg.sandbox_time_limit_s))
    res = ex.validate(SIMPLE_REWARD)
    print("validate ok:", getattr(res, "ok", None), getattr(res, "error", None), flush=True)
    terms = ex.wrap(SIMPLE_REWARD, goal_ref)
    print("wrap ok (jit-check passed)", flush=True)

    runner = PPORunner(TrainConfig.from_config(cfg, checkpoint_dir="/data/mujoco/runs/diag"))
    try:
        result = runner.train(
            terms,
            epochs=cfg.train_epochs,
            learning_rate=cfg.learning_rate,
            goal=goal,
            num_envs=cfg.num_envs,
        )
        print("TRAIN OK:", type(result).__name__, flush=True)
        ckpt = getattr(result, "checkpoint", None)
        print("checkpoint:", getattr(ckpt, "path", ckpt), flush=True)
        return 0
    except Exception:
        print("TRAIN FAILED ----- full traceback -----", flush=True)
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
