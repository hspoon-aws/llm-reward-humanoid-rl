#!/usr/bin/env bash
# Full Eureka loop, ONE ITERATION PER FRESH PROCESS (the blocker-#9 fix).
#
# Isaac Lab cannot build a second manager-based env in a process that already
# built one, so the single-process loop hangs on iteration 1's env rebuild
# ("Parsing configuration..." with the GPU idle). This driver instead runs each
# iteration in its OWN fresh docker container (fresh SimulationApp) by calling
# scripts/run_iteration.py, which runs Orchestrator.run(max_new_iterations=1).
# The orchestrator's S3 loop_checkpoint carries state across iterations, so the
# multi-process run is behaviorally identical to the single-process loop.
#
# Logs to /var/log/full_train.log with a FULLTRAIN_DONE_<rc> marker.
set -uo pipefail
LOG=/var/log/full_train.log
BUCKET=humanoid-from-scratch-123456789012
REGION=us-west-2
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
CONFIG="${CONFIG:-config/run_config.g6.yaml}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://10.20.20.70:8000/v1}"
RUN_ID="${RUN_ID:-full-train-$(date +%s)}"
MAX_PROCS="${MAX_PROCS:-30}"   # hard safety cap on iterations launched
: > "${LOG}"
{
  echo "=== chunked full training $(date -u) run_id=${RUN_ID} config=${CONFIG} llm=${LLM_ENDPOINT} ==="
  # Ensure the latest camera patch is in the repo for the in-loop recorder.
  aws s3 cp "s3://${BUCKET}/runs/code/patches/camera_cfg.py" \
    /opt/humanoid/src/sensors/camera_cfg.py --region "${REGION}" || true

  i=0
  rc=0
  while [ "${i}" -lt "${MAX_PROCS}" ]; do
    echo "--- launching iteration process #${i} $(date -u) ---"
    OUT=$(docker run --rm --gpus all --network host \
      -v /opt/humanoid:/work -w /work \
      -e HUMANOID_LLM_ENDPOINT="${LLM_ENDPOINT}" \
      "${IMAGE}" \
      bash -lc "/workspace/isaaclab/isaaclab.sh -p scripts/run_iteration.py \
          --config ${CONFIG} --run-id ${RUN_ID} --llm-endpoint ${LLM_ENDPOINT}" \
      2>&1 | tee /dev/stderr)
    rc=$?
    echo "iteration process #${i} docker rc=${rc}"

    # Detect completion from the ITERATION_DONE marker the driver prints.
    DONE_LINE=$(echo "${OUT}" | grep -E "^ITERATION_DONE:" | tail -1)
    echo "marker: ${DONE_LINE}"
    if echo "${DONE_LINE}" | grep -q '"run_complete": true'; then
      echo "RUN COMPLETE after process #${i}"
      break
    fi
    # If the process itself failed (rc != 0) AND made no progress marker, stop
    # to avoid an infinite crash loop.
    if [ "${rc}" -ne 0 ] && [ -z "${DONE_LINE}" ]; then
      echo "iteration process #${i} failed with no marker; aborting driver"
      break
    fi
    i=$((i + 1))
  done

  echo "FULLTRAIN_DONE_${rc}"
} >> "${LOG}" 2>&1
