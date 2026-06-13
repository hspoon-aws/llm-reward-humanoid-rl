#!/usr/bin/env bash
# Run the standalone demo-video recorder validation inside the Isaac Lab
# container. Logs to /var/log/recorder.log with a RECORDER_DONE_<rc> marker.
set -uo pipefail
LOG=/var/log/recorder.log
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
NUM_ENVS="${NUM_ENVS:-4}"
MAX_STEPS="${MAX_STEPS:-200}"
CKPT="${CKPT:-runs/checkpoints/model_final.pt}"
: > "${LOG}"
{
  echo "=== recorder validation $(date -u) envs=${NUM_ENVS} steps=${MAX_STEPS} ckpt=${CKPT} ==="
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p scripts/validate_recorder.py \
        --checkpoint ${CKPT} --output-dir runs/demo_videos \
        --num-envs ${NUM_ENVS} --max-steps ${MAX_STEPS}"
  echo "RECORDER_DONE_$?"
} >> "${LOG}" 2>&1
