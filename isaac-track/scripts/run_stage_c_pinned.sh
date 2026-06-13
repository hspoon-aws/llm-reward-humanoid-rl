#!/usr/bin/env bash
# Stage C with a runtime typing_extensions pin (pre-bake validation).
# Pins numpy + typing_extensions in the Isaac python, then runs the smoke.
set -uo pipefail
LOG=/var/log/stageC.log
: > "${LOG}"
{
  echo "=== Stage C (pinned deps) $(date -u) ==="
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    humanoid-from-scratch:latest \
    bash -lc '/workspace/isaaclab/_isaac_sim/python.sh -m pip install -q numpy==1.26.4 typing_extensions==4.12.2 && /workspace/isaaclab/isaaclab.sh -p -m src.run_loop --config config/run_config.g6.yaml --smoke --smoke-num-envs 64 --smoke-epochs 5 --eval-episodes 2 --llm-endpoint http://10.20.20.70:8000/v1 --run-id stage-c-smoke'
  echo "STAGEC_DONE_$?"
} >> "${LOG}" 2>&1
