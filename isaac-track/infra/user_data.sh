#!/bin/bash
# Capacity-block host bootstrap (Ubuntu 22.04 DLAMI).
#
# Hydrates the Qwen3-Coder weights from S3 straight to the local NVMe instance
# store (__MODEL_DIR__) and restores the vLLM/FlashInfer compile caches from S3,
# so the first vLLM launch reads weights at full NVMe speed AND skips the
# ~13 min Blackwell JIT/compile warmup.
#
# WHY S3 -> NVME (not EBS-from-snapshot): a snapshot-restored volume lazy-loads
# blocks from S3 on first read; without Fast Snapshot Restore that first 57 GB
# read crawls (~4 MB/s) and stalls vLLM for hours. S3 -> NVMe over the S3
# gateway endpoint runs at >170 MB/s. See docs/lesson-ebs-snapshot-hydration.md
# and docs/lesson-flashinfer-jit-blackwell.md.
#
# Operator access is via SSM Session Manager only (no inbound SG rule). Service
# binding to loopback (vLLM/TensorBoard --host 127.0.0.1) is enforced by the
# launch scripts in scripts/, not here.
set -uo pipefail
exec > >(tee -a /var/log/humanoid-bootstrap.log) 2>&1
echo "=== bootstrap start $(date -u) ==="

REGION="__REGION__"
BUCKET="__BUCKET__"
MODEL_DIR="__MODEL_DIR__"
S3_MODEL_PREFIX="s3://${BUCKET}/models/qwen3-coder-30b/"
S3_CACHE_PREFIX="s3://${BUCKET}/runs/cache"

mark() { echo "$1" > "$(dirname "${MODEL_DIR}")/BOOTSTRAP_STATUS" 2>/dev/null || true; echo ">>> STATUS: $1"; }
mkdir -p "${MODEL_DIR}" /root/.cache/vllm /root/.cache/flashinfer

# --- hydrate weights: S3 -> local NVMe ------------------------------------ #
mark "HYDRATING_WEIGHTS"
echo "syncing ${S3_MODEL_PREFIX} -> ${MODEL_DIR}"
if aws s3 sync "${S3_MODEL_PREFIX}" "${MODEL_DIR}/" --region "${REGION}" --exclude ".cache/*" --only-show-errors; then
  echo "weights hydrated: $(du -sh "${MODEL_DIR}" 2>/dev/null | cut -f1)"
  mark "WEIGHTS_READY"
else
  echo "ERROR: weight hydration from S3 failed" >&2
  mark "WEIGHTS_FAILED"
fi

# --- restore compile caches (skip Blackwell JIT/compile warmup) ----------- #
mark "RESTORING_CACHES"
aws s3 sync "${S3_CACHE_PREFIX}/vllm/" /root/.cache/vllm/ --region "${REGION}" --only-show-errors \
  && echo "vllm cache restored" || echo "no vllm cache (will compile on first launch)"
aws s3 sync "${S3_CACHE_PREFIX}/flashinfer/" /root/.cache/flashinfer/ --region "${REGION}" --only-show-errors \
  && echo "flashinfer cache restored" || echo "no flashinfer cache (will JIT on first launch)"
mark "READY"

echo "=== bootstrap finished $(date -u) ==="
echo "Next: clone the repo, then run:  MODEL_DIR=${MODEL_DIR} scripts/launch_vllm.sh"
