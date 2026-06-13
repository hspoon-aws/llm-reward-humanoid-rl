# humanoid-mujoco-from-scratch

Eureka-style **LLM → Reward → RL** loop teaching a Unitree **H1** humanoid point-to-point
goal-reaching, on **MuJoCo MJX + Brax PPO + JAX**, running on a `p6-b200.48xlarge` (8× B200).

Sibling track to the Isaac Lab project (`../`). Same task, same 22 requirements, same
LLM-driven loop — only the simulator + RL stack differ. See `PROJECT-PLAN.md` for the full
plan, the dual-track (Isaac/RTX vs MuJoCo/B200) rationale, and open decisions.

## Why MuJoCo here

Isaac Sim needs RT cores; the B200 (data-center Blackwell) has none. MJX runs physics as
JAX/XLA on CUDA **compute**, the same path vLLM already uses on this host — so the sim half
maps onto the hardware we already have. The B200 is dedicated to this track; Isaac runs on a
separate RTX box.

## Status

**Scaffolding only.** Skeletons with the DI seam in place; no MJX/JAX/Brax wired yet. The
framework-agnostic core (Qwen_Client, Orchestrator, Config, S3_Store, Evaluator math) is
carried over from the Isaac project; the simulator adapter (`build_mjx_trainer`) and the
JAX reward contract are the new work. See `PROJECT-PLAN.md` → "Phased execution plan".

## Environment

Confirmed against the live MuJoCo Playground locomotion registry:
- **Primary env:** `H1JoystickGaitTracking` (Unitree H1), reframed to goal-reaching.
- **Fallback:** `G1JoystickFlatTerrain` (config-swappable via `training.env_name`).

## Layout

```
src/            framework-agnostic core (carried over) + MJX/Brax adapters (new)
prompts/        JAX-rewritten reward prompt templates
config/         run_config.yaml
scripts/        capability probe + headless MJX/render smoke gates
tests/          off-GPU suite (no MJX/JAX import at module load)
docs/           bring-up + decision lessons
.kiro/specs/humanoid-mujoco-llm-rl/   requirements / design / tasks
```

## Dev

```bash
pip install -e ".[dev]"
pytest            # off-GPU suite; must pass with no GPU and no MJX/JAX installed
```
