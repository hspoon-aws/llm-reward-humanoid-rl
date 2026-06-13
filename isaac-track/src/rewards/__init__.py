"""Reward handling: validation, sandboxed execution, and Isaac Lab wrapping.

See ``src/rewards/reward_executor.py`` (Reward_Executor, Tasks 4 and 9).

Only the stdlib-only validation surface plus the torch-light sandbox execution
layer (Tasks 4.1/4.2) and the ``wrap`` layer (Task 4.3) are re-exported here;
importing this package does not pull in ``torch`` (torch is imported lazily
inside the sandbox builder, only when a reward body actually needs it). The
``wrap`` layer produces :class:`WrappedReward` Isaac Lab ``RewTerm`` callables
bound to a :class:`GoalRef`; the live ``RewardManager`` registration is Task 9.1.
"""

from src.rewards.reward_executor import (
    ALLOWED_BUILTINS,
    REWARD_FUNCTION_NAME,
    GoalRef,
    RewardExecutor,
    SandboxConfig,
    ValidationResult,
    WrappedReward,
)

__all__ = [
    "RewardExecutor",
    "ValidationResult",
    "SandboxConfig",
    "GoalRef",
    "WrappedReward",
    "REWARD_FUNCTION_NAME",
    "ALLOWED_BUILTINS",
]
