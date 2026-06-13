#!/usr/bin/env bash
# Pull the patched camera_cfg.py, re-render the chase+side demo videos from the
# final checkpoint with the adjusted (pulled-back, recentered) cameras, and
# upload them to s3://.../runs/inspect/. Logs to /var/log/rerender.log with a
# RERENDER_DONE_<rc> marker.
set -uo pipefail
LOG=/var/log/rerender.log
BUCKET=humanoid-from-scratch-123456789012
REGION=us-west-2
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
NUM_ENVS="${NUM_ENVS:-1}"
MAX_STEPS="${MAX_STEPS:-300}"
CKPT="${CKPT:-runs/checkpoints/model_final.pt}"
OUTDIR=runs/demo_videos_v2
: > "${LOG}"
{
  echo "=== rerender $(date -u) envs=${NUM_ENVS} steps=${MAX_STEPS} ckpt=${CKPT} ==="

  # (1) Pull the patched camera config into the repo on the host.
  aws s3 cp "s3://${BUCKET}/runs/code/patches/camera_cfg.py" \
    /opt/humanoid/src/sensors/camera_cfg.py --region "${REGION}"
  echo "patched camera_cfg.py:"
  grep -nE "position=\(" /opt/humanoid/src/sensors/camera_cfg.py

  # (2) Render in a fresh single-env process (chase + side).
  mkdir -p "/opt/humanoid/${OUTDIR}"
  docker run --rm --gpus all --network host \
    -v /opt/humanoid:/work -w /work \
    "${IMAGE}" \
    bash -lc "/workspace/isaaclab/isaaclab.sh -p scripts/record_demo.py \
        --checkpoint ${CKPT} --output-dir ${OUTDIR} \
        --label v2_best --num-envs ${NUM_ENVS} --max-steps ${MAX_STEPS}"
  RC=$?
  echo "record_demo rc=${RC}"
  ls -la "/opt/humanoid/${OUTDIR}/" 2>&1

  # (3) Upload whatever mp4s were produced to inspect/.
  for f in /opt/humanoid/${OUTDIR}/*.mp4; do
    [ -e "$f" ] || continue
    base=$(basename "$f")
    aws s3 cp "$f" "s3://${BUCKET}/runs/inspect/v2_${base}" --region "${REGION}"
  done

  echo "RERENDER_DONE_${RC}"
} >> "${LOG}" 2>&1
