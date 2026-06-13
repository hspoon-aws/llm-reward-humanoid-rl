#!/usr/bin/env bash
#
# Launch vLLM serving Qwen3-Coder-30B-A3B-Instruct on GPU 0.
#
# Target hardware: EC2 p6-b200.48xlarge (8x NVIDIA B200, sm_100 / Blackwell).
# vLLM must be a Blackwell-aware build running on CUDA 12.8+ for sm_100.
# The model is a 30B MoE with ~3B active params, so a single B200 (180 GB)
# has ample headroom.
#
# Weights are expected to be present locally (hydrated from S3 / EBS) at
# MODEL_DIR. We serve from the local path to avoid any runtime HF download.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/data/models/qwen3-coder-30b}"
SERVED_NAME="${SERVED_NAME:-Qwen3-Coder-30B-A3B-Instruct}"
# Loopback-only bind (Req 22.3). vLLM defaults to 0.0.0.0; pin to 127.0.0.1 so
# the endpoint is reachable on-host only and never exposed via the SG. The
# Orchestrator/PPO loop consume it in-process; operators reach it via SSM
# port-forwarding (Req 22.4), not an inbound rule.
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"

if [[ ! -d "${MODEL_DIR}" ]]; then
    echo "ERROR: model directory not found: ${MODEL_DIR}" >&2
    echo "Hydrate it first, e.g.:" >&2
    echo "  aws s3 sync s3://humanoid-from-scratch-123456789012/models/qwen3-coder-30b/ ${MODEL_DIR}/" >&2
    exit 1
fi

# Pin the language model to GPU 0; GPUs 1-7 are reserved for Isaac Lab PPO.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

echo "=== [$(date)] Launching vLLM ==="
echo "  model dir : ${MODEL_DIR}"
echo "  served as : ${SERVED_NAME}"
echo "  bind      : ${HOST}:${PORT} (loopback-only)"
echo "  GPU       : CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

exec vllm serve "${MODEL_DIR}" \
    --served-model-name "${SERVED_NAME}" \
    --dtype bfloat16 \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --tensor-parallel-size 1 \
    --host "${HOST}" \
    --port "${PORT}" \
    --enable-prefix-caching \
    --trust-remote-code
