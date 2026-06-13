"""Run entrypoint: wire the real collaborators and drive the Eureka loop (MuJoCo).

MuJoCo-track analog of the Isaac ``run_loop.py``. Constructs the live
collaborators from a validated :class:`src.config.Config` and hands them to the
:class:`src.orchestrator.Orchestrator`. Pure-Python collaborators (Qwen_Client,
Reward_Executor, S3_Store, Evaluator) are built eagerly; the MJX/Brax backends
(PPO_Runner, the Evaluator's rollout/recorder) only touch JAX/MuJoCo lazily at
train/eval time, so importing/constructing here never requires the GPU stack.

Entry points:
  * :func:`build_orchestrator` — wire an Orchestrator from a Config (every
    collaborator overridable for tests).
  * :func:`main` — CLI: load config, build, run (or ``--smoke`` for hour-0).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.config import Config, load_config
from src.eval.evaluator import EvalConfig, Evaluator
from src.llm.qwen_client import QwenClient, QwenClientConfig
from src.orchestrator import Orchestrator, RunResult
from src.rewards.reward_executor import RewardExecutor, SandboxConfig
from src.sensors.camera_cfg import CameraConfig
from src.storage.s3_store import S3Store
from src.train.mjx_trainer import PPORunner, TrainConfig

__all__ = [
    "build_qwen_client",
    "build_reward_executor",
    "build_ppo_runner",
    "build_evaluator",
    "build_s3_store",
    "build_orchestrator",
    "main",
]


def build_qwen_client(config: Config, *, prompts_dir: str = "prompts"):
    # A shared cross-process lock serializes vLLM calls across concurrent loop
    # processes that share one self-hosted endpoint (§7D). Opt-in via
    # QWEN_REQUEST_LOCK (a file path); unset => no locking (single-process runs).
    import os as _os

    lock_path = _os.environ.get("QWEN_REQUEST_LOCK")
    # Generous retries/backoff so a momentarily busy endpoint doesn't turn into a
    # skipped iteration even if the lock isn't used.
    max_retries = max(int(config.qwen_max_retries), 6)

    provider = getattr(config, "llm_provider", "vllm")
    if provider == "bedrock":
        # Amazon Bedrock Claude backend. Reuses every QwenClient behavior
        # (templates, extraction, validation) and only swaps the transport.
        # Bedrock is a managed concurrent endpoint, so the request lock is
        # unnecessary (left wired for parity but typically unset for Bedrock).
        from src.llm.bedrock_client import BedrockClient, BedrockClientConfig

        return BedrockClient(
            BedrockClientConfig(
                model=getattr(config, "llm_model", "us.anthropic.claude-opus-4-8"),
                region=getattr(config, "llm_region", "us-west-2"),
                max_retries=max_retries,
                retry_backoff_s=3.0,
                request_timeout_s=180.0,
                prompts_dir=Path(prompts_dir),
                request_lock_path=lock_path,
            )
        )

    return QwenClient(
        QwenClientConfig(
            endpoint=config.llm_endpoint,
            model=getattr(config, "llm_model", "Qwen3-Coder-30B-A3B-Instruct"),
            max_retries=max_retries,
            retry_backoff_s=3.0,
            request_timeout_s=180.0,
            prompts_dir=Path(prompts_dir),
            request_lock_path=lock_path,
        )
    )


def build_reward_executor(config: Config) -> RewardExecutor:
    return RewardExecutor(SandboxConfig(time_limit_s=config.sandbox_time_limit_s))


def build_ppo_runner(config: Config, *, checkpoint_dir: Optional[str] = None) -> PPORunner:
    return PPORunner(TrainConfig.from_config(config, checkpoint_dir=checkpoint_dir))


def build_evaluator(config: Config) -> Evaluator:
    return Evaluator(
        EvalConfig.from_config(config),
        CameraConfig.from_config(config),
    )


def build_s3_store(config: Config, *, run_id: Optional[str] = None) -> S3Store:
    return S3Store(config.s3_location, run_id=run_id)


def build_orchestrator(
    config: Config,
    *,
    run_id: Optional[str] = None,
    checkpoint_dir: Optional[str] = None,
    prompts_dir: str = "prompts",
    qwen: Any = None,
    executor: Any = None,
    runner: Any = None,
    evaluator: Any = None,
    store: Any = None,
    eval_episodes: Optional[int] = None,
) -> Orchestrator:
    qwen = qwen if qwen is not None else build_qwen_client(config, prompts_dir=prompts_dir)
    executor = executor if executor is not None else build_reward_executor(config)
    runner = runner if runner is not None else build_ppo_runner(config, checkpoint_dir=checkpoint_dir)
    evaluator = evaluator if evaluator is not None else build_evaluator(config)
    store = store if store is not None else build_s3_store(config, run_id=run_id)

    kwargs: dict[str, Any] = {}
    if eval_episodes is not None:
        kwargs["eval_episodes"] = int(eval_episodes)
    return Orchestrator(config, qwen, executor, runner, evaluator, store, **kwargs)


def _smoke_config(config: Config, *, epochs: int, num_envs: int) -> Config:
    import dataclasses

    return dataclasses.replace(
        config,
        max_iterations=1,
        train_epochs=max(1, int(epochs)),
        checkpoint_interval=max(1, int(epochs)),
        num_envs=max(1, int(num_envs)),
        oom_fallback_envs=max(1, min(int(num_envs), config.oom_fallback_envs)),
    )


def _print_result(result: RunResult) -> None:
    print(f"[run] iterations attempted: {result.iterations_run}")
    for record in result.iterations:
        ckpt = getattr(record.checkpoint, "path", None)
        print(f"[run]   iteration {record.index}: status={record.status} checkpoint={ckpt}")
        # Surface the reason for non-completed iterations. For skipped records
        # the reason is preserved in behavior_description (see
        # Orchestrator._skipped_record); without this the run log only shows the
        # ambiguous status and the real cause (generation vs train/eval vs
        # divergence) stays hidden in the checkpoint JSON.
        if not getattr(record, "completed", False):
            reason = getattr(record, "behavior_description", None)
            if reason:
                print(f"[run]       reason: {reason}")
    if result.best_policy is not None:
        print(
            f"[run] best policy: iteration {result.best_policy.iteration_index} "
            f"score={result.best_policy.score} checkpoint={result.best_policy.checkpoint.path}"
        )
    else:
        print("[run] best policy: none (no iteration completed evaluation)")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Humanoid MuJoCo LLM+RL Eureka loop runner")
    parser.add_argument("--config", default="config/run_config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--checkpoint-dir", default="/data/mujoco/runs")
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-epochs", type=int, default=2)
    parser.add_argument("--smoke-num-envs", type=int, default=256)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--llm-endpoint", default=None)
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.llm_endpoint:
        import dataclasses
        config = dataclasses.replace(config, llm_endpoint=args.llm_endpoint)
        print(f"[run] LLM endpoint override: {args.llm_endpoint}")
    if args.smoke:
        config = _smoke_config(config, epochs=args.smoke_epochs, num_envs=args.smoke_num_envs)
        print(
            f"[smoke] hour-0 smoke: 1 iteration, {config.train_epochs} epochs, "
            f"{config.num_envs} envs"
        )

    orchestrator = build_orchestrator(
        config,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        prompts_dir=args.prompts_dir,
        eval_episodes=args.eval_episodes if args.eval_episodes is not None else (2 if args.smoke else None),
    )

    result = orchestrator.run()
    _print_result(result)

    if args.smoke:
        completed = [r for r in result.iterations if r.completed and r.checkpoint]
        ok = bool(completed)
        print("=== HOUR-0 SMOKE: PASS ===" if ok else "=== HOUR-0 SMOKE: FAIL ===")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
