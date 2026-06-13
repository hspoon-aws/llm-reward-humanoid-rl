#!/usr/bin/env bash
# Restore the Isaac Lab image + repo from S3 onto a FRESH instance (i.e. after a
# terminate, or a brand-new host). After a stop/start this is NOT needed — the
# root EBS volume already retains /var/lib/docker and /opt/humanoid.
set -uo pipefail
BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
REGION="${REGION:-us-west-2}"
IMG_KEY="s3://${BUCKET}/runs/cache/images/humanoid-from-scratch.tar.zst"
WORK="${WORK:-/opt/dlami/nvme/_cache_tmp}"
[ -d /opt/dlami/nvme ] || WORK="/tmp/_cache_tmp"
mkdir -p "${WORK}"
log() { echo "=== [$(date -u +%H:%M:%S)] $* ==="; }

# --- image ---------------------------------------------------------------- #
if docker image inspect humanoid-from-scratch:latest >/dev/null 2>&1; then
  log "image already present locally; skip download"
else
  log "downloading image from ${IMG_KEY}"
  aws s3 cp "${IMG_KEY}" "${WORK}/image.tar.zst" --region "${REGION}" --only-show-errors
  command -v zstd >/dev/null 2>&1 || (apt-get update -y && apt-get install -y zstd) >/dev/null 2>&1 || true
  log "loading image"
  zstd -d -c "${WORK}/image.tar.zst" | docker load
  rm -f "${WORK}/image.tar.zst"
fi

# --- repo ----------------------------------------------------------------- #
mkdir -p /opt/humanoid
log "restoring repo -> /opt/humanoid"
aws s3 cp "s3://${BUCKET}/runs/code/humanoid-repo-host.tar.gz" "${WORK}/repo.tar.gz" \
  --region "${REGION}" --only-show-errors \
  && tar -C /opt/humanoid -xzf "${WORK}/repo.tar.gz" && rm -f "${WORK}/repo.tar.gz" \
  || log "no host repo archive; falling back to runs/code/humanoid-repo.tar.gz"

log "restore complete"
docker images humanoid-from-scratch --format "{{.Repository}}:{{.Tag}} {{.Size}}"
