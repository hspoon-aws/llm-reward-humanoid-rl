"""Run entrypoint: wire the real collaborators and drive the Eureka loop.

This module is the single place that constructs the *live* collaborators from a
validated :class:`src.config.Config` and hands them to the
:class:`src.orchestrator.Orchestrator`. It is the bridge between the pure,
unit-tested components and the Isaac Sim runtime on the GPU host.

Design references:
  - design.md -> Components and Interfaces -> Orchestrator (collaborator wiring)
  - tasks.md -> Task 16 (Phase 2 hour-0 smoke) / Task 17 (final checkpoint)

Two entry points:
  * :func:`build_orchestrator` — construct an :class:`Orchestrator` wired with the
    real Qwen_Client, Reward_Executor, PPO_Runner, Evaluator, and S3_Store from a
    Config. Pure-Python collaborators (Qwen_Client, Reward_Executor, S3_Store) are
    built eagerly; the Isaac-Sim-backed PPO_Runner/Evaluator only touch Isaac Lab
    lazily at train/eval time (see :mod:`src.train.ppo_runner` /
    :mod:`src.eval.evaluator`), so importing and constructing here never requires
    the simulator.
  * :func:`main` — CLI: load config, build the orchestrator, run the loop (or a
    short hour-0 smoke with ``--smoke``), and report the outcome.

The heavy backends remain dependency-injected: ``build_orchestrator`` accepts
optional ``runner`` / ``evaluator`` overrides so a smoke test (or a host without
Isaac Lab) can inject fakes, exactly like the unit tests do.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

# Make 'src' importable when run as a script from the repo root.
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
from src.train.ppo_runner import PPORunner, TrainConfig

__all__ = [
    "build_qwen_client",
    "build_reward_executor",
    "build_ppo_runner",
    "build_evaluator",
    "build_s3_store",
    "build_orchestrator",
    "main",
]


# --------------------------------------------------------------------------- #
# Collaborator factories (pure-Python; no Isaac Lab at construction time)
# --------------------------------------------------------------------------- #
def build_qwen_client(config: Config, *, prompts_dir: str = "prompts") -> QwenClient:
    """Build the Language_Model client from the run Config (Req 1, 3, 16).

    Selects the backend via ``config.llm_provider``: the default ``"vllm"`` talks
    to the self-hosted Qwen at ``llm_endpoint``; ``"bedrock"`` calls Amazon
    Bedrock directly (``bedrock_model_id`` in ``bedrock_region``) with no
    self-hosted vLLM required. The prompt templates load eagerly from
    ``prompts_dir`` so a missing template fails fast (Req 3.3).
    """
    return QwenClient(
        QwenClientConfig(
            endpoint=config.llm_endpoint,
            max_retries=config.qwen_max_retries,
            retry_backoff_s=config.qwen_retry_backoff_s,
            request_timeout_s=config.qwen_request_timeout_s,
            prompts_dir=Path(prompts_dir),
            provider=getattr(config, "llm_provider", "vllm"),
            bedrock_model_id=getattr(
                config, "bedrock_model_id", "global.anthropic.claude-opus-4-8"
            ),
            bedrock_region=getattr(config, "bedrock_region", "us-west-2"),
        )
    )


def build_reward_executor(config: Config) -> RewardExecutor:
    """Build the Reward_Executor with the configured sandbox time limit (Req 5.2)."""
    return RewardExecutor(SandboxConfig(time_limit_s=config.sandbox_time_limit_s))


def build_ppo_runner(config: Config, *, checkpoint_dir: Optional[str] = None) -> PPORunner:
    """Build the PPO_Runner from the run Config (Req 8).

    The runner restricts devices to the configured training GPUs (excluding the
    reserved model GPU 0) and builds the real Isaac Lab / RSL-RL trainer lazily at
    ``train`` time, so constructing it here needs no GPU.
    """
    train_config = TrainConfig.from_config(config, checkpoint_dir=checkpoint_dir)
    return PPORunner(train_config)


def build_evaluator(config: Config, *, record_demo_video: bool = True) -> Evaluator:
    """Build the Evaluator from the run Config (Req 9, 10, 17).

    The live policy-rollout + demo-video recorder are built lazily inside the
    Evaluator at ``evaluate`` time, so constructing it here needs no GPU.

    ``record_demo_video=False`` disables best/worst demo recording (Req 10),
    which is strictly additive. On a single-process train+eval host the live
    recorder must build a second Isaac Lab env after training already owns the
    one SimulationApp, which can stall on the RTX rendering-kit reinit; the smoke
    gate (iteration completed + checkpoint) does not need video, so it disables
    recording to keep the gate unblocked.
    """
    import dataclasses

    eval_config = EvalConfig.from_config(config)
    eval_config = dataclasses.replace(eval_config, record_demo_video=record_demo_video)
    return Evaluator(
        eval_config,
        CameraConfig.from_config(config),
    )


def build_s3_store(config: Config, *, run_id: Optional[str] = None) -> S3Store:
    """Build the S3_Store from the run Config (Req 11).

    Uses the configured ``s3_location`` and an optional per-run id so artifacts
    land under ``s3://<bucket>/<prefix>/<run-id>/iteration-NN/`` (Req 11.2).
    """
    return S3Store(config.s3_location, run_id=run_id)


# --------------------------------------------------------------------------- #
# Orchestrator wiring
# --------------------------------------------------------------------------- #
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
    record_demo_video: bool = True,
) -> Orchestrator:
    """Construct an :class:`Orchestrator` wired with the live collaborators.

    Builds the Qwen_Client, Reward_Executor, PPO_Runner, Evaluator, and S3_Store
    from ``config`` and assembles the Orchestrator. Every collaborator can be
    overridden with a keyword argument so a smoke test (or a host without Isaac
    Lab) can inject fakes — the same dependency-injection seam the unit tests use.

    Args:
        config: The validated run :class:`~src.config.Config`.
        run_id: Optional per-run id for the S3 artifact path (Req 11.2).
        checkpoint_dir: Optional local checkpoint directory for the PPO_Runner.
        prompts_dir: Directory holding the prompt templates (Req 3.1).
        qwen / executor / runner / evaluator / store: Optional collaborator
            overrides (injected fakes in tests).
        eval_episodes: Evaluation episodes per iteration; defaults to a small
            value suitable for the unattended loop.

    Returns:
        A ready-to-run :class:`Orchestrator`.
    """
    qwen = qwen if qwen is not None else build_qwen_client(config, prompts_dir=prompts_dir)
    executor = executor if executor is not None else build_reward_executor(config)
    runner = runner if runner is not None else build_ppo_runner(config, checkpoint_dir=checkpoint_dir)
    # In-loop recording is OFF by construction: Isaac Lab cannot build a second
    # manager-based env in the training process, so per-iteration recording
    # stalls. The mandatory demo video (Req 10) is rendered END-ONLY via the
    # final_video_recorder hook below, in a fresh subprocess from the Best_Policy.
    evaluator = evaluator if evaluator is not None else build_evaluator(config, record_demo_video=False)
    store = store if store is not None else build_s3_store(config, run_id=run_id)

    kwargs: dict[str, Any] = {}
    if eval_episodes is not None:
        kwargs["eval_episodes"] = int(eval_episodes)
    # End-of-run + per-iteration Best/worst demo recorders (Req 10), unless
    # recording is disabled. Both spawn a FRESH subprocess per call (separate
    # SimulationApp), which is the only configuration that works on Isaac Lab.
    # Per-iteration matches the MuJoCo track's per-iteration mp4s; end-only adds
    # the final Best_Policy demo.
    if record_demo_video:
        from src.eval.evaluator import build_final_video_recorder  # noqa: PLC0415
        recorder_fn = build_final_video_recorder(config)
        kwargs["final_video_recorder"] = recorder_fn
        kwargs["iteration_video_recorder"] = recorder_fn
    return Orchestrator(config, qwen, executor, runner, evaluator, store, **kwargs)


# --------------------------------------------------------------------------- #
# Smoke-mode config narrowing (Task 16: hour-0 smoke)
# --------------------------------------------------------------------------- #
def _smoke_config(config: Config, *, epochs: int, num_envs: int) -> Config:
    """Return a copy of ``config`` narrowed for the hour-0 smoke (Task 16).

    The smoke proves the full pipeline wires end-to-end — one reward generated,
    a short goal-conditioned PPO run completes and produces a checkpoint — so it
    runs a single iteration with a tiny epoch budget and env count rather than the
    full 24-hour sprint. Validation bounds are preserved (the values stay within
    the Config loader's accepted ranges).
    """
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
    """Print a concise run summary (iterations, best policy, checkpoint)."""
    print(f"[run] iterations attempted: {result.iterations_run}")
    for record in result.iterations:
        ckpt = getattr(record.checkpoint, "path", None)
        print(f"[run]   iteration {record.index}: status={record.status} checkpoint={ckpt}")
    if result.best_policy is not None:
        print(
            f"[run] best policy: iteration {result.best_policy.iteration_index} "
            f"score={result.best_policy.score} checkpoint={result.best_policy.checkpoint.path}"
        )
    else:
        print("[run] best policy: none (no iteration completed evaluation)")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[list[str]] = None) -> int:
    """Load config, build the orchestrator, and run the loop (or hour-0 smoke).

    Usage:
        # full run from the canonical config
        python -m src.run_loop --config config/run_config.yaml

        # hour-0 smoke: one reward -> short PPO run -> checkpoint (Task 16)
        python -m src.run_loop --config config/run_config.yaml --smoke

    Returns a process exit code: 0 on success, non-zero on failure.
    """
    parser = argparse.ArgumentParser(description="Humanoid LLM+RL Eureka loop runner")
    parser.add_argument("--config", default="config/run_config.yaml",
                        help="Path to the run configuration YAML (Req 18.1).")
    parser.add_argument("--run-id", default=None,
                        help="Per-run id for the S3 artifact path (Req 11.2).")
    parser.add_argument("--checkpoint-dir", default=None,
                        help="Local directory for policy checkpoints.")
    parser.add_argument("--prompts-dir", default="prompts",
                        help="Directory holding the prompt templates (Req 3.1).")
    parser.add_argument("--smoke", action="store_true",
                        help="Run the hour-0 smoke: 1 iteration, short PPO run (Task 16).")
    parser.add_argument("--smoke-epochs", type=int, default=5,
                        help="Epoch budget for --smoke (default 5).")
    parser.add_argument("--smoke-num-envs", type=int, default=64,
                        help="Parallel env count for --smoke (default 64).")
    parser.add_argument("--eval-episodes", type=int, default=None,
                        help="Evaluation episodes per iteration (default: loop default).")
    parser.add_argument("--no-demo-video", action="store_true",
                        help="Disable best/worst demo-video recording (Req 10). "
                             "Recording is additive; implied by --smoke unless "
                             "--demo-video is also passed.")
    parser.add_argument("--demo-video", action="store_true",
                        help="Force best/worst demo-video recording ON even under "
                             "--smoke (overrides the smoke default of off). Use to "
                             "validate the in-loop recorder on a small run.")
    parser.add_argument("--llm-endpoint", default=None,
                        help="Override the OpenAI-compatible vLLM endpoint (e.g. a "
                             "cross-host vLLM at http://10.20.20.70:8000/v1). Defaults "
                             "to the config's llm.endpoint.")
    args = parser.parse_args(argv)

    config = load_config(args.config)
    if args.llm_endpoint:
        import dataclasses
        config = dataclasses.replace(config, llm_endpoint=args.llm_endpoint)
        print(f"[run] LLM endpoint override: {args.llm_endpoint}")
    if args.smoke:
        config = _smoke_config(
            config, epochs=args.smoke_epochs, num_envs=args.smoke_num_envs
        )
        print(
            f"[smoke] hour-0 smoke: 1 iteration, {config.train_epochs} epochs, "
            f"{config.num_envs} envs"
        )

    # Demo recording is additive; disabled for the smoke gate by default (which
    # only needs iteration-completed + checkpoint) or when explicitly requested.
    # --demo-video forces it ON even under --smoke (to validate the in-loop
    # recorder on a small run).
    record_demo_video = args.demo_video or not (args.no_demo_video or args.smoke)

    orchestrator = build_orchestrator(
        config,
        run_id=args.run_id,
        checkpoint_dir=args.checkpoint_dir,
        prompts_dir=args.prompts_dir,
        eval_episodes=args.eval_episodes,
        record_demo_video=record_demo_video,
    )

    try:
        result = orchestrator.run()
    finally:
        # Tear down the process-level SimulationApp once, at process exit (it is
        # shared across all iterations/eval/recorder and never closed mid-run).
        try:
            from src.sim_app import sim_app_launched, get_sim_app  # noqa: PLC0415

            if sim_app_launched():
                get_sim_app().close()
        except Exception:  # noqa: BLE001 - teardown is best-effort
            pass
    _print_result(result)

    # The smoke gate passes when the single iteration completed and produced a
    # checkpoint (Task 16 exit criterion).
    if args.smoke:
        completed = [r for r in result.iterations if r.completed and r.checkpoint]
        ok = bool(completed)
        print("=== HOUR-0 SMOKE: PASS ===" if ok else "=== HOUR-0 SMOKE: FAIL ===")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
