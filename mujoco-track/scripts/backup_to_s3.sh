#!/usr/bin/env bash
# Back up the box's local-only artifacts to S3 before the EC2 is terminated.
#
# What this saves (the things NOT already durable elsewhere):
#   - run logs (/data/mujoco/runs/*.log) — training curves, eval_reward history,
#     [iter]/[warm-start] lines. These are NEVER uploaded by the loop, so they
#     are the primary at-risk artifact.
#   - any local checkpoints/metrics for an in-progress run not yet auto-uploaded.
#
# What is ALREADY safe and intentionally NOT re-backed-up here:
#   - source/configs/docs   -> committed + pushed to GitHub.
#   - completed-iteration artifacts -> the loop already uploads them per
#     iteration to runs-mujoco/<run-id>/iteration-NN/.
#   - Qwen model weights (57G) -> already in s3://.../models/qwen3-coder-30b/.
#   - mjxvenv (8G) -> rebuildable via scripts/b200_setup_mjx.sh.
#
# Destination: runs-mujoco/_box_backup/ (the instance role can only PutObject
# under runs/* and runs-mujoco/*, so we stay inside runs-mujoco/).
#
# Run ON the box (it uses the instance role), e.g. via SSM:
#   scripts/ssm_run.sh 'bash /data/mujoco/scripts/backup_to_s3.sh'
set -uo pipefail

BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
PREFIX="${PREFIX:-runs-mujoco/_box_backup}"
SRC="${SRC:-/data/mujoco/runs}"
DEST="s3://${BUCKET}/${PREFIX}"

echo "[backup] $(date -u +%FT%TZ)  ${SRC}  ->  ${DEST}"
aws s3 sync "${SRC}/" "${DEST}/" --exclude "*.lock"

echo "[backup] verifying..."
n_log=$(aws s3 ls "${DEST}/" --recursive 2>/dev/null | grep -c "\.log" || true)
n_all=$(aws s3 ls "${DEST}/" --recursive 2>/dev/null | wc -l | tr -d ' ')
echo "[backup] done: ${n_all} objects in backup (${n_log} logs)."
echo "[backup] NOTE: also confirm code is pushed to GitHub and model weights"
echo "[backup]       remain at s3://${BUCKET}/models/qwen3-coder-30b/."
