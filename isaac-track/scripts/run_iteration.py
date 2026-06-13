"""Run EXACTLY ONE Eureka iteration in this (fresh) process, then exit.

Why this exists
---------------
Isaac Lab cannot build a second manager-based RL env in a process that already
built one: ``gym.make`` on iteration 2+ stalls forever at "Parsing configuration
from ... H1FlatEnvCfg" with the GPU idle (the single-``SimulationApp`` limit —
blocker #9 in docs/lesson-isaac-lab-bringup.md). The full in-process loop
therefore completes iteration 0 and then hangs on iteration 1's env rebuild.

The fix: run ONE iteration per process. Each invocation builds a fresh
SimulationApp, runs a single iteration via ``Orchestrator.run(max_new_iterations=1)``,
persists the loop checkpoint to S3, and exits — tearing down the SimulationApp
cleanly. The outer driver (``scripts/run_full_training_chunked.sh``) calls this
repeatedly; the orchestrator's existing resume mechanism (loop_checkpoint.json)
carries history / best-policy / next-index across processes, so the multi-process
run is behaviorally identical to the single-process loop, minus the stall.

The LAST iteration (the one that reaches ``max_iterations``) additionally runs
the end-of-run finalize inside ``run()``: Best_Policy export + final demo video.

Prints ITERATION_DONE: <rc> <json> so the driver can detect completion and tell
whether the whole run has finished (``"run_complete": true``).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Run one Eureka iteration in a fresh process")
    parser.add_argument("--config", default="config/run_config.g6.yaml")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--llm-endpoint", default=None)
    parser.add_argument("--prompts-dir", default="prompts")
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--no-demo-video", action="store_true",
                        help="Disable per-iteration / final demo video recording.")
    args = parser.parse_args(argv)

    from src.config import load_config  # noqa: PLC0415
    from src.run_loop import build_orchestrator  # noqa: PLC0415

    config = load_config(args.config)
    if args.llm_endpoint:
        import dataclasses  # noqa: PLC0415
        config = dataclasses.replace(config, llm_endpoint=args.llm_endpoint)
        print(f"[iter] LLM endpoint override: {args.llm_endpoint}")

    orchestrator = build_orchestrator(
        config,
        run_id=args.run_id,
        prompts_dir=args.prompts_dir,
        eval_episodes=args.eval_episodes,
        record_demo_video=not args.no_demo_video,
    )

    rc = 1
    payload: dict = {"run_complete": False}
    try:
        # Run exactly one new iteration this process, then exit so the next
        # iteration gets a fresh SimulationApp.
        result = orchestrator.run(max_new_iterations=1)
        next_index = orchestrator._next_iteration  # noqa: SLF001 - intentional read
        run_complete = next_index >= config.max_iterations
        payload = {
            "run_complete": bool(run_complete),
            "next_iteration": int(next_index),
            "max_iterations": int(config.max_iterations),
            "iterations_this_process": int(result.iterations_run),
        }
        if result.iterations:
            rec = result.iterations[-1]
            payload["last_status"] = str(getattr(rec, "status", ""))
            ckpt = getattr(getattr(rec, "checkpoint", None), "path", None)
            payload["last_checkpoint"] = ckpt
        rc = 0
    except Exception:  # noqa: BLE001 - report any failure to the driver
        traceback.print_exc()
        rc = 1

    # NOTE: we deliberately do NOT call SimulationApp.close() here. Its teardown
    # frequently hangs (busy-spins, GPU idle, never returns), which would block
    # the driver forever. We hard-exit below instead; the OS reclaims the app.
    print(f"ITERATION_DONE: {rc} {json.dumps(payload)}")
    # Force a hard process exit. Isaac Sim's SimulationApp teardown frequently
    # HANGS (busy-spins on CPU with the GPU idle and never returns), which would
    # block the chunked driver's `docker run` forever and prevent the next
    # iteration from launching. All artifacts (checkpoint, metrics, videos) and
    # the loop_checkpoint are already persisted to S3 by the time run() returns,
    # so we can safely skip a graceful interpreter shutdown: flush stdio and exit
    # the process immediately with the iteration's return code. os._exit bypasses
    # atexit/finalizers (including the wedged SimulationApp close).
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)


if __name__ == "__main__":
    sys.exit(main())
