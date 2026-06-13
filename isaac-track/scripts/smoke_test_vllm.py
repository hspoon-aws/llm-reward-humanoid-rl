#!/usr/bin/env python3
"""vLLM / Qwen_Client smoke test (deployment gates: "vLLM responds" and
"Run 1 LLM reward generation -> Valid Python returned").

Checks, in order:
  1. the vLLM server is reachable and lists the served model
  2. the Qwen_Client can generate a reward function whose compute_reward
     definition extracts and ast-parses cleanly

Usage:
    python scripts/smoke_test_vllm.py --endpoint http://localhost:8000/v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import requests

# Make 'src' importable when run from the repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.llm import QwenClient, QwenClientConfig, QwenClientError  # noqa: E402

OBS_SPACE = (
    "indices 0-2 base linear velocity, 3-5 base angular velocity, "
    "6-8 projected gravity, 9-11 velocity commands, 12-30 joint positions (19), "
    "31-49 joint velocities (19), 50-68 previous actions (19), "
    "then goal-conditioning: vector-to-goal (x,y) in the robot frame plus "
    "distance and heading to the Goal"
)
TASK = (
    "Make the Unitree H1 humanoid walk from its start position (point A) to a "
    "target position (point B) and arrive within the success radius without "
    "falling, moving efficiently."
)
GOAL = (
    "Goal (point B) is at (5.0, 0.0) m on the ground plane; success radius 0.5 m. "
    "Reward should reduce distance-to-goal, reward arrival within the radius, and "
    "keep the robot upright en route."
)


def check_server(endpoint: str) -> bool:
    url = f"{endpoint.rstrip('/')}/models"
    try:
        resp = requests.get(url, timeout=10)
        resp.raise_for_status()
        models = [m.get("id") for m in resp.json().get("data", [])]
        print(f"[ok] vLLM reachable; served models: {models}")
        return True
    except requests.RequestException as exc:
        print(f"[FAIL] vLLM not reachable at {url}: {exc}")
        return False


def check_generation(endpoint: str, prompts_dir: Path) -> bool:
    client = QwenClient(
        QwenClientConfig(endpoint=endpoint, prompts_dir=prompts_dir)
    )
    try:
        code = client.generate_reward(
            task_description=TASK, obs_space=OBS_SPACE, goal_description=GOAL
        )
    except QwenClientError as exc:
        print(f"[FAIL] reward generation failed: {exc}")
        return False

    print("[ok] generated + validated compute_reward; first lines:")
    for line in code.splitlines()[:8]:
        print(f"      {line}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="vLLM + Qwen client smoke test")
    parser.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    parser.add_argument(
        "--prompts-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "prompts",
    )
    args = parser.parse_args()

    server_ok = check_server(args.endpoint)
    gen_ok = check_generation(args.endpoint, args.prompts_dir) if server_ok else False

    passed = server_ok and gen_ok
    print("=== LLM GATE: PASS ===" if passed else "=== LLM GATE: FAIL ===")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
