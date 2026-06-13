#!/usr/bin/env bash
#
# RESTORE bootstrap for the capacity-block instance.
#
# Primary path (default): hydrate the Qwen3-Coder weights from S3 straight to
# the local NVMe instance store, restore the vLLM/FlashInfer compile caches from
# S3, verify, and print the launch command — so the first vLLM launch is warm
# (no JIT compile) and reads weights at full NVMe speed.
#
# WHY NOT EBS-FROM-SNAPSHOT: a volume created from a snapshot lazy-hydrates its
# blocks from S3 on first read. Without Fast Snapshot Restore, the first 57 GB
# read crawls (~4 MB/s observed) and stalls vLLM's weight load for hours. Pulling
# S3 -> local NVMe over the S3 gateway endpoint runs at >170 MB/s instead.
# See docs/lesson-ebs-snapshot-hydration.md and
# docs/lesson-flashinfer-jit-blackwell.md.
#
# Set RESTORE_MODE=ebs to use the legacy snapshot-attach path (kept as a
# fallback; only sensible if Fast Snapshot Restore is enabled on the snapshot).
#
# Run ON the capacity-block instance (needs the instance role: s3 read on
# models/* and runs/cache/*, ec2 perms only for the EBS fallback).
set -euo pipefail

# --- config --------------------------------------------------------------- #
RESTORE_MODE="${RESTORE_MODE:-s3}"                     # s3 (default) | ebs
BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
S3_MODEL_PREFIX="${S3_MODEL_PREFIX:-s3://${BUCKET}/models/qwen3-coder-30b/}"
S3_CACHE_PREFIX="${S3_CACHE_PREFIX:-s3://${BUCKET}/runs/cache}"
PROJECT_TAG="${PROJECT_TAG:-humanoid-llm-rl-poc}"

# Serve weights from fast local NVMe instance store by default.
NVME_ROOT="${NVME_ROOT:-/opt/dlami/nvme}"
MODEL_DIR="${MODEL_DIR:-${NVME_ROOT}/qwen3-coder-30b}"
RESTORE_CACHES="${RESTORE_CACHES:-1}"                  # 1 = pull compile caches from S3

# EBS-fallback knobs (only used when RESTORE_MODE=ebs)
SNAPSHOT_ID="${SNAPSHOT_ID:-snap-EXAMPLE00000000}"
DATA="${DATA:-/data}"
DEVICE="${DEVICE:-/dev/sdf}"
VOLUME_TYPE="${VOLUME_TYPE:-gp3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
log() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# --- discover instance via IMDSv2 ----------------------------------------- #
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
  -H "X-aws-ec2-metadata-token-ttl-seconds: 300")
imds() { curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" "http://169.254.169.254/latest/meta-data/$1"; }
REGION="$(imds placement/region || echo us-west-2)"

# --- compile-cache restore (skips the ~13 min Blackwell JIT/compile warmup) - #
# vLLM torch.compile + AOT graphs and the FlashInfer sm100 JIT kernels are
# keyed to GPU arch + model + vLLM/FlashInfer/torch versions; they hit on an
# identical p6-b200 with the same wheels. A miss just recompiles (and we
# re-upload below), so this is always safe to attempt.
restore_caches() {
  [ "${RESTORE_CACHES}" = "1" ] || { log "cache restore disabled"; return 0; }
  log "Restoring compile caches from ${S3_CACHE_PREFIX}"
  mkdir -p /root/.cache/vllm /root/.cache/flashinfer
  aws s3 sync "${S3_CACHE_PREFIX}/vllm/" /root/.cache/vllm/ \
    --region "${REGION}" --only-show-errors || log "no vllm cache (will compile + re-upload)"
  aws s3 sync "${S3_CACHE_PREFIX}/flashinfer/" /root/.cache/flashinfer/ \
    --region "${REGION}" --only-show-errors || log "no flashinfer cache (will JIT + re-upload)"
}

# --- PRIMARY: S3 -> local NVMe -------------------------------------------- #
restore_from_s3() {
  log "Hydrating weights S3 -> NVMe: ${S3_MODEL_PREFIX} -> ${MODEL_DIR}"
  mkdir -p "${MODEL_DIR}"
  # In-region over the S3 gateway endpoint; fast and needs no public IP.
  aws s3 sync "${S3_MODEL_PREFIX}" "${MODEL_DIR}/" \
    --region "${REGION}" --exclude ".cache/*" --only-show-errors
  log "Weights hydrated: $(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)"
}

# --- FALLBACK: EBS volume from snapshot ----------------------------------- #
restore_from_ebs() {
  local AZ IID VOL_ID DEV
  IID="$(imds instance-id)"; AZ="$(imds placement/availability-zone)"
  MODEL_DIR="${DATA}/models/qwen3-coder-30b"
  log "EBS fallback: snapshot=${SNAPSHOT_ID} az=${AZ} (ensure Fast Snapshot Restore is enabled, else this is slow)"
  VOL_ID=$(aws ec2 create-volume --region "${REGION}" --availability-zone "${AZ}" \
    --snapshot-id "${SNAPSHOT_ID}" --volume-type "${VOLUME_TYPE}" \
    --tag-specifications "ResourceType=volume,Tags=[{Key=project,Value=${PROJECT_TAG}},{Key=Name,Value=qwen3-coder-weights}]" \
    --query "VolumeId" --output text)
  aws ec2 wait volume-available --region "${REGION}" --volume-ids "${VOL_ID}"
  aws ec2 attach-volume --region "${REGION}" --volume-id "${VOL_ID}" \
    --instance-id "${IID}" --device "${DEVICE}" >/dev/null
  aws ec2 wait volume-in-use --region "${REGION}" --volume-ids "${VOL_ID}"
  DEV=""
  for _ in $(seq 1 30); do
    for cand in "${DEVICE}" /dev/xvdf /dev/nvme1n1 /dev/nvme2n1; do
      if [ -b "${cand}" ] && blkid "${cand}" 2>/dev/null | grep -qi 'TYPE="xfs"'; then DEV="${cand}"; break; fi
    done
    [ -n "${DEV}" ] && break; sleep 2
  done
  [ -n "${DEV}" ] || { echo "ERROR: could not resolve attached device for ${VOL_ID}" >&2; exit 1; }
  mkdir -p "${DATA}"; mountpoint -q "${DATA}" || mount "${DEV}" "${DATA}"
  log "Mounted ${DEV} at ${DATA}"
}

# --- run ------------------------------------------------------------------ #
if [ "${RESTORE_MODE}" = "ebs" ]; then
  restore_from_ebs
else
  restore_from_s3
fi

[ -d "${MODEL_DIR}" ] || { echo "ERROR: ${MODEL_DIR} missing after restore" >&2; exit 1; }

# --- verify weights before serving (Task 0.3) ---------------------------- #
log "Verifying weights at ${MODEL_DIR}"
python3 "${SCRIPT_DIR}/verify_weights.py" --model-dir "${MODEL_DIR}"

restore_caches

log "Restore complete. Launch vLLM with:"
echo "  MODEL_DIR=${MODEL_DIR} ${SCRIPT_DIR}/launch_vllm.sh"
echo
echo "After a successful FIRST launch on a new arch/version, persist the warmed"
echo "caches back to S3 so future launches skip the JIT/compile step:"
echo "  aws s3 sync /root/.cache/vllm/       ${S3_CACHE_PREFIX}/vllm/       --region ${REGION}"
echo "  aws s3 sync /root/.cache/flashinfer/ ${S3_CACHE_PREFIX}/flashinfer/ --region ${REGION}"
