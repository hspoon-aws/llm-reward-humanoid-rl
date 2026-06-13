#!/usr/bin/env bash
# Persist the built Isaac Lab image + repo to S3 so they survive even instance
# TERMINATION (a stop/start already keeps them on the root EBS volume).
#
# Saves: docker image -> zstd tar -> s3://<bucket>/runs/cache/images/
#        repo tarball  -> s3://<bucket>/runs/code/
# Idempotent; skips the image upload if an identically-named object already
# exists unless FORCE=1.
set -uo pipefail
BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
REGION="${REGION:-us-west-2}"
IMAGE="${IMAGE:-humanoid-from-scratch:latest}"
IMG_KEY="s3://${BUCKET}/runs/cache/images/humanoid-from-scratch.tar.zst"
WORK="${WORK:-/opt/dlami/nvme/_cache_tmp}"   # use big NVMe if present, else /tmp
[ -d /opt/dlami/nvme ] || WORK="/tmp/_cache_tmp"
mkdir -p "${WORK}"
log() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# --- image: docker save | zstd -> S3 -------------------------------------- #
if [ "${FORCE:-0}" != "1" ] && aws s3 ls "${IMG_KEY}" --region "${REGION}" >/dev/null 2>&1; then
  log "image already in S3 (${IMG_KEY}); skip (FORCE=1 to overwrite)"
else
  log "saving ${IMAGE} -> zstd"
  if command -v zstd >/dev/null 2>&1; then
    docker save "${IMAGE}" | zstd -T0 -3 > "${WORK}/image.tar.zst"
  else
    log "zstd not present; installing"
    (apt-get update -y && apt-get install -y zstd) >/dev/null 2>&1 || true
    docker save "${IMAGE}" | zstd -T0 -3 > "${WORK}/image.tar.zst"
  fi
  log "uploading image ($(du -h "${WORK}/image.tar.zst" | cut -f1)) -> ${IMG_KEY}"
  aws s3 cp "${WORK}/image.tar.zst" "${IMG_KEY}" --region "${REGION}" --only-show-errors
  rm -f "${WORK}/image.tar.zst"
fi

# --- repo: tar -> S3 ------------------------------------------------------- #
if [ -d /opt/humanoid ]; then
  log "archiving /opt/humanoid -> S3 runs/code/"
  tar -C /opt/humanoid -czf "${WORK}/repo.tar.gz" . 2>/dev/null
  aws s3 cp "${WORK}/repo.tar.gz" "s3://${BUCKET}/runs/code/humanoid-repo-host.tar.gz" \
    --region "${REGION}" --only-show-errors
  rm -f "${WORK}/repo.tar.gz"
fi

log "cache-to-S3 complete"
echo "Restore on a fresh instance with: scripts/restore_image_from_s3.sh"
