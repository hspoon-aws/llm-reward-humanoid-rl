#!/usr/bin/env bash
# Stage C hour-0 smoke: run the Eureka loop end-to-end at a tiny budget inside
# the Isaac Lab container, using the cross-host B200 vLLM over the VPC.
# Logs to /var/log/stageC.log with a STAGEC_DONE_<rc> marker. Backgroundable.
#
# Env knobs:
#   LLM_ENDPOINT  - vLLM OpenAI endpoint (default: B200 private IP)
#   SMOKE_ENVS    - parallel envs (default 64)
#   SMOKE_EPOCHS  - PPO epochs (default 5)
#   EVAL_EPISODES - eval episodes (default 2)
set -uo pipefail
LOG=/var/log/stageC.log
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
LLM_ENDPOINT="${LLM_ENDPOINT:-http://10.20.20.70:8000/v1}"
SMOKE_ENVS="${SMOKE_ENVS:-64}"
SMOKE_EPOCHS="${SMOKE_EPOCHS:-5}"
EVAL_EPISODES="${EVAL_EPISODES:-2}"
: > "${LOG}"
{
  echo "=== Stage C smoke $(date -u) envs=${SMOKE_ENVS} epochs=${SMOKE_EPOCHS} eval=${EVAL_EPISODES} llm=${LLM_ENDPOINT} ==="
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    -e HUMANOID_LLM_ENDPOINT="${LLM_ENDPOINT}" \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p -m src.run_loop \
        --config config/run_config.g6.yaml \
        --smoke --smoke-num-envs ${SMOKE_ENVS} --smoke-epochs ${SMOKE_EPOCHS} \
        --eval-episodes ${EVAL_EPISODES} --llm-endpoint ${LLM_ENDPOINT} \
        --run-id stage-c-smoke"
  echo "STAGEC_DONE_$?"
} >> "${LOG}" 2>&1
