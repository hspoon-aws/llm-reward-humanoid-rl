#!/usr/bin/env bash
# Launch the A/B comparison runs on the B200 (run ON the box via SSM).
#   A: 1-GPU  (GPU 1)         -> run-1gpu-v4   config/run_config.yaml      (num_envs 2048)
#   B: 6-GPU  (GPUs 2-7)      -> run-6gpu-v4   config/run_config_6gpu.yaml (num_envs 3072)
# Both share one self-hosted vLLM (GPU 0) and a cross-process request lock so
# their (rare) generation calls take turns (§7D). Wall-clock + final best-policy
# score per run are the comparison metrics.
set -uo pipefail
cd /data/mujoco

# Stale .pyc has bitten us before (a changed-signature ImportError) — always
# clear bytecode after pushing source.
find /data/mujoco -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null || true

LOCK=/data/mujoco/runs/vllm.lock
PY=/data/mjxvenv/bin/python
mkdir -p /data/mujoco/runs

# A — 1 GPU
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl QWEN_REQUEST_LOCK="$LOCK" \
  nohup "$PY" -m src.run_loop \
    --config config/run_config.yaml \
    --run-id run-1gpu-v4 \
    --checkpoint-dir /data/mujoco/runs/run-1gpu-v4 \
    > /data/mujoco/runs/run-1gpu-v4.log 2>&1 &
echo "A (1-GPU) pid=$!"

# B — 6 GPUs (2..7)
CUDA_VISIBLE_DEVICES=2,3,4,5,6,7 MUJOCO_GL=egl QWEN_REQUEST_LOCK="$LOCK" \
  nohup "$PY" -m src.run_loop \
    --config config/run_config_6gpu.yaml \
    --run-id run-6gpu-v4 \
    --checkpoint-dir /data/mujoco/runs/run-6gpu-v4 \
    > /data/mujoco/runs/run-6gpu-v4.log 2>&1 &
echo "B (6-GPU) pid=$!"

sleep 5
echo "=== launched; processes ==="
pgrep -af "run_loop" || echo "NONE (check logs)"
