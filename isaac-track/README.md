# Humanoid Locomotion from Scratch — LLM + Isaac Lab

Teach a Unitree H1 humanoid to walk/run via an Eureka-style
LLM → Reward → RL loop. A **Qwen3-Coder-30B-A3B** (vLLM) iteratively
writes reward functions; **Isaac Lab** trains a PPO policy with them; metrics
feed back to the model for refinement.

> **Hardware note.** Isaac Sim needs RT cores, so this track runs on an
> **RT-core GPU — a `g6`/`g6e` instance (NVIDIA L4 / L40S, Ada)**, not on a
> data-center card. (A data-center Blackwell box has no RT cores and cannot
> launch Isaac Sim — see the lessons below.) The reward LLM is served by vLLM,
> reached cross-host. See
> `../docs/lessons/lesson-gpu-architecture-purpose-fit.md` and
> `../docs/lessons/lesson-isaac-lab-bringup.md`.

## Status — Pre-Work (Qwen LLM + Isaac Lab setup)

| Item | State |
|------|-------|
| AWS account / region | `<ACCOUNT_ID>`, **us-west-2** (profile `your-aws-profile`) |
| Model weights → S3 | `Qwen/Qwen3-Coder-30B-A3B-Instruct` (~61 GB) syncing to bucket below |
| Prompt templates | `prompts/` (initial / refine / analyze) |
| Qwen client | `src/llm/qwen_client.py` (+ 7 passing offline tests) |
| vLLM launch | `scripts/launch_vllm.sh` (B200 / sm_100 aware) |
| Container | `Dockerfile` (Isaac Lab 2.2 base + vLLM) |
| Run config | `config/run_config.yaml` |
| Gates | `scripts/smoke_test_env.py`, `scripts/smoke_test_vllm.py` |

**Model artifact:** `s3://humanoid-from-scratch-<ACCOUNT_ID>/models/qwen3-coder-30b/`

## Layout

```
src/llm/qwen_client.py     # OpenAI-compatible client: generate/refine/analyze
prompts/                   # editable prompt templates
scripts/
  download_and_sync.sh     # HF -> local staging -> S3 (model weights)
  launch_vllm.sh           # serve Qwen on GPU 0
  smoke_test_env.py        # Isaac Lab H1 env-reset gate
  smoke_test_vllm.py       # vLLM reachable + reward generation gate
config/run_config.yaml     # all tunable run parameters (spec Req 18)
tests/                     # offline unit tests (no GPU needed)
Dockerfile                 # Isaac Lab + vLLM runtime image
```

## Quick checks (controller host, no GPU)

```bash
python -m pytest tests/ -q
```

## On the GPU host (Hour 0)

```bash
# 1. hydrate weights
aws s3 sync s3://humanoid-from-scratch-<ACCOUNT_ID>/models/qwen3-coder-30b/ \
    /data/models/qwen3-coder-30b/

# 2. serve the model on GPU 0
MODEL_DIR=/data/models/qwen3-coder-30b scripts/launch_vllm.sh &

# 3. gates
isaaclab.sh -p scripts/smoke_test_env.py --num-envs 16
python scripts/smoke_test_vllm.py --endpoint http://localhost:8000/v1
```

## GPU allocation

Isaac Lab training runs on the `g6`/`g6e` RT-core GPU (single L4 / L40S). The
reward LLM (Qwen3-Coder-30B-A3B, MoE ~3B active) is served by vLLM and reached
cross-host, so the simulator GPU is dedicated to PPO training (thousands of
parallel humanoids). The original 8-GPU layout that co-located vLLM on GPU 0 and
trained on GPUs 1-7 does not apply here, because Isaac Sim cannot run on that
data-center hardware (no RT cores).
