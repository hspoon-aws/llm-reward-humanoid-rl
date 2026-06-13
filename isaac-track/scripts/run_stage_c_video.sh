#!/usr/bin/env bash
# Stage C smoke WITH in-loop demo-video recording enabled (1 GPU).
# Same tiny budget as run_stage_c.sh but passes --demo-video so the recorder
# runs inside the loop (validates the single-SimulationApp in-loop recording
# path). Logs to /var/log/stageC_video.log with a STAGEC_DONE_<rc> marker.
set -uo pipefail
LOG=/var/log/stageC_video.log
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://10.20.20.70:8000/v1}"
SMOKE_ENVS="${SMOKE_ENVS:-64}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
: > "${LOG}"
{
  echo "=== Stage C +video $(date -u) envs=${SMOKE_ENVS} epochs=${SMOKE_EPOCHS} eval=${EVAL_EPISODES} llm=${LLM_ENDPOINT} ==="
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    -e HUMANOID_LLM_ENDPOINT="${LLM_ENDPOINT}" \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p -m src.run_loop \
        --config config/run_config.g6.yaml \
        --smoke --smoke-num-envs ${SMOKE_ENVS} --smoke-epochs ${SMOKE_EPOCHS} \
        --eval-episodes ${EVAL_EPISODES} --llm-endpoint ${LLM_ENDPOINT} \
        --demo-video --run-id stage-c-video"
  echo "STAGEC_DONE_$?"
} >> "${LOG}" 2>&1
