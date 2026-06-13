#!/usr/bin/env bash
# Full Eureka-style LLM->Reward->RL training run on the single L4 (g6).
# Uses the g6 config (12 iterations, 600 epochs/iter search budget, 1024 envs),
# cross-host vLLM on the B200, and in-loop per-iteration demo-video recording
# (on by default for a non-smoke run). A unique RUN_ID avoids the stale-
# checkpoint gotcha (reusing a run-id skips already-done iterations).
#
# Logs to /var/log/full_train.log with a FULLTRAIN_DONE_<rc> marker.
set -uo pipefail
LOG=/var/log/full_train.log
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://10.20.20.70:8000/v1}"
RUN_ID="${RUN_ID:-full-train-$(date +%s)}"
: > "${LOG}"
{
  echo "=== full training $(date -u) run_id=${RUN_ID} llm=${LLM_ENDPOINT} ==="
  # Ensure the latest camera patch is in the repo for the in-loop recorder.
  aws s3 cp s3://humanoid-from-scratch-123456789012/runs/code/patches/camera_cfg.py \
    /opt/humanoid/src/sensors/camera_cfg.py --region us-west-2 || true

  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    -e HUMANOID_LLM_ENDPOINT="${LLM_ENDPOINT}" \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p -m src.run_loop \
        --config config/run_config.g6.yaml \
        --llm-endpoint ${LLM_ENDPOINT} \
        --demo-video --run-id ${RUN_ID}"
  echo "FULLTRAIN_DONE_$?"
} >> "${LOG}" 2>&1
