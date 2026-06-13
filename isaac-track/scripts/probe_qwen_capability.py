#!/usr/bin/env python3
"""Capability probe for the self-hosted Qwen3-Coder model.

Unlike scripts/smoke_test_vllm.py (a liveness gate), this exercises real
reward-generation capability and auto-grades the output against the hard
constraints the Eureka loop depends on:

  - returns exactly one compute_reward(env, obs, actions)
  - returns (reward, components) with components a dict
  - torch-only (no os/open/eval/numpy/subprocess imports)
  - vectorized (no `for` loop over environments)
  - task-grounded (distance/progress, arrival, upright, alive terms)
  - some NaN/Inf guard
  - ast.parse succeeds

Exit code 0 if all REQUIRED checks pass, else 1.

Usage:
    python3 scripts/probe_qwen_capability.py --endpoint http://127.0.0.1:8000/v1
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import sys
import urllib.request

PROMPT = """You are an expert reinforcement-learning reward engineer working with NVIDIA Isaac Lab.

TASK: A Unitree H1 humanoid (19 DOF) must walk from its start position (point A,
the origin) to a Goal (point B) at (5.0, 0.0) m on flat ground, arriving within a
0.5 m success radius without falling, moving efficiently. There is NO camera —
reward operates on environment state only.

OBSERVATION (obs tensor, shape (num_envs, 69)):
  [0:3]   base linear velocity
  [3:6]   base angular velocity
  [6:9]   projected gravity (upright when ~[0,0,-1])
  [9:12]  velocity commands
  [12:31] joint positions (19)
  [31:50] joint velocities (19)
  [50:69] previous actions (19)
The Goal and full robot pose are also readable from `env`.

WRITE exactly one function with this signature:

    def compute_reward(env, obs, actions):
        # returns (reward, components)
        #   reward:     torch.Tensor shape (num_envs,)
        #   components: dict[str, torch.Tensor], each shape (num_envs,)

HARD RULES:
1. Use only torch operations. Import nothing except torch.
2. No file/network/os/eval/exec access. No camera/image input.
3. Fully vectorized - no Python loops over environments.
4. Every component MUST be in the `components` dict, and `reward` MUST equal the
   sum of all components.
5. Keep tensors on the same device/dtype as `obs`. Guard against NaN/Inf.

Include a goal-distance/progress term, an arrival bonus inside the radius, an
upright/posture term, an alive bonus, and small smoothness penalties.

Respond with a brief explanation, then the function in a single ```python block."""

_CODE_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def call_model(endpoint: str, model: str, temperature: float) -> str:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = json.dumps({
        "model": model,
        "temperature": temperature,
        "max_tokens": 2048,
        "messages": [{"role": "user", "content": PROMPT}],
    }).encode()
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        data = json.loads(r.read().decode())
    return data["choices"][0]["message"]["content"]


def extract_code(text: str) -> str:
    for block in _CODE_RE.findall(text):
        if "def compute_reward" in block:
            return block.strip()
    if "def compute_reward" in text:
        return text[text.index("def compute_reward"):].strip()
    return ""


def grade(code: str) -> tuple[list[tuple[str, bool, bool, str]], bool]:
    """Return [(name, passed, required, detail)] and overall pass (all required)."""
    checks: list[tuple[str, bool, bool, str]] = []

    parses = True
    tree = None
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        parses = False
        detail = str(e)
    else:
        detail = ""
    checks.append(("ast.parse", parses, True, detail))

    funcs = [n for n in ast.walk(tree)] if tree else []
    has_fn = any(isinstance(n, ast.FunctionDef) and n.name == "compute_reward" for n in funcs)
    checks.append(("defines compute_reward", has_fn, True, ""))

    sig_ok = False
    if tree:
        for n in ast.walk(tree):
            if isinstance(n, ast.FunctionDef) and n.name == "compute_reward":
                args = [a.arg for a in n.args.args]
                sig_ok = args[:3] == ["env", "obs", "actions"]
    checks.append(("signature(env,obs,actions)", sig_ok, True, ""))

    # imports: torch only
    bad_imports = []
    if tree:
        for n in ast.walk(tree):
            if isinstance(n, ast.Import):
                bad_imports += [a.name for a in n.names if a.name != "torch"]
            elif isinstance(n, ast.ImportFrom):
                if n.module and n.module != "torch":
                    bad_imports.append(n.module)
    checks.append(("torch-only imports", not bad_imports, True, ",".join(bad_imports)))

    # forbidden names (word-boundary / call-pattern matched to avoid false
    # positives like 'pos', 'position', 'cos', 'loss' tripping a bare 'os')
    forbidden_patterns = {
        "os module": r"\bos\.",
        "open()": r"\bopen\s*\(",
        "eval()": r"\beval\s*\(",
        "exec()": r"\bexec\s*\(",
        "__import__": r"__import__",
        "subprocess": r"\bsubprocess\b",
        "socket": r"\bsocket\b",
        "numpy": r"\bnumpy\b|\bnp\.",
    }
    found_forbidden = [name for name, pat in forbidden_patterns.items() if re.search(pat, code)]
    checks.append(("no forbidden calls", not found_forbidden, True, ",".join(found_forbidden)))

    # returns a tuple (reward, components)
    returns_tuple = bool(re.search(r"return\s+\w+\s*,\s*\w+", code))
    checks.append(("returns (reward, components)", returns_tuple, True, ""))

    # components dict present
    has_components_dict = bool(re.search(r"components\s*=\s*\{", code)) or "components[" in code
    checks.append(("builds components dict", has_components_dict, True, ""))

    # vectorized: no `for ... in range(num_envs)` style loop over envs
    env_loop = bool(re.search(r"for\s+\w+\s+in\s+range\(.*num_envs", code)) or \
        bool(re.search(r"for\s+\w+\s+in\s+range\(.*obs\.shape\[0\]", code))
    checks.append(("vectorized (no env loop)", not env_loop, True, ""))

    # task grounding (advisory)
    lc = code.lower()
    grounding = {
        "goal/distance term": any(k in lc for k in ["goal", "dist", "progress", "target"]),
        "upright/posture term": any(k in lc for k in ["upright", "gravity", "posture", "orient", "[:, 6:9]", "[:,6:9]"]),
        "arrival bonus": any(k in lc for k in ["radius", "arriv", "reach", "0.5", "success"]),
        "alive bonus": "alive" in lc or "alive_bonus" in lc or "is_alive" in lc,
    }
    for name, ok in grounding.items():
        checks.append((name, ok, False, ""))

    # NaN/Inf guard (advisory)
    guard = any(k in code for k in ["nan_to_num", "clamp", "isfinite", "clip"])
    checks.append(("NaN/Inf guard", guard, False, ""))

    overall = all(passed for _, passed, required, _ in checks if required)
    return checks, overall


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--endpoint", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen3-Coder-30B-A3B-Instruct")
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--show-code", action="store_true")
    args = ap.parse_args()

    print(f"[probe] querying {args.endpoint} (model={args.model}, temp={args.temperature})")
    try:
        resp = call_model(args.endpoint, args.model, args.temperature)
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] model call failed: {e}")
        return 1

    code = extract_code(resp)
    if not code:
        print("[FAIL] no compute_reward code block in response")
        print("--- raw response (first 800 chars) ---")
        print(resp[:800])
        return 1

    checks, overall = grade(code)
    print("\n=== capability checks (R=required) ===")
    for name, passed, required, detail in checks:
        tag = "R" if required else " "
        mark = "PASS" if passed else "FAIL"
        extra = f"  <- {detail}" if detail and not passed else ""
        print(f"  [{tag}] {mark}  {name}{extra}")

    if args.show_code:
        print("\n--- generated compute_reward ---")
        print(code)
    else:
        print("\n--- generated compute_reward (first 20 lines) ---")
        for line in code.splitlines()[:20]:
            print(f"   {line}")

    print("\n=== CAPABILITY PROBE: " + ("PASS ===" if overall else "FAIL ==="))
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
