#!/usr/bin/env bash
# Host-side launcher for the Stage A Isaac Lab import/registry check.
# Runs the container (GPU, --network host, repo bind-mounted) and logs to
# /var/log/stageA.log with a STAGEA_DONE_<rc> marker. First run includes the
# Isaac Sim warmup, so this is meant to be backgrounded and polled.
set -uo pipefail
LOG=/var/log/stageA.log
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
: > "${LOG}"
{
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p scripts/stage_a_check.py"
  echo "STAGEA_DONE_$?"
} >> "${LOG}" 2>&1
