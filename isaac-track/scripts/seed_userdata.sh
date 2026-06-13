#!/bin/bash
# User-data for the model-seeding instance.
# Downloads Qwen3-Coder-30B-A3B-Instruct from HuggingFace onto a dedicated EBS
# volume mounted at /data, then syncs it into S3. The /data volume is what we
# snapshot afterwards so the 21:00 capacity-block instance can restore weights
# instantly instead of re-downloading.
#
# All output is logged to /var/log/seed.log and a status marker is written to
# /data/SEED_STATUS so we can poll progress over SSM.
set -uo pipefail
exec > >(tee -a /var/log/seed.log) 2>&1
echo "=== seed start $(date -u) ==="

BUCKET="humanoid-from-scratch-123456789012"
S3_PREFIX="s3://${BUCKET}/models/qwen3-coder-30b/"
MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
DATA=/data
MODEL_DIR="${DATA}/models/qwen3-coder-30b"
PROJECT_TAG="humanoid-llm-rl-poc"
REGION="us-west-2"

mark() { echo "$1" > ${DATA}/SEED_STATUS 2>/dev/null || true; echo ">>> STATUS: $1"; }

# --- mount the dedicated EBS data volume ---------------------------------- #
# The extra volume is the second NVMe device. Find the unmounted one.
DEV=""
for d in /dev/nvme1n1 /dev/nvme2n1 /dev/xvdb /dev/sdb; do
  if [ -b "$d" ]; then DEV="$d"; break; fi
done
echo "data device: ${DEV:-NONE}"
if [ -n "$DEV" ]; then
  if ! blkid "$DEV" >/dev/null 2>&1; then
    mkfs -t xfs "$DEV"
  fi
  mkdir -p ${DATA}
  mount "$DEV" ${DATA} || true
fi
mkdir -p ${MODEL_DIR}
mark "MOUNTED"

# --- tag this instance and its EBS volumes with the project tag ----------- #
# Best-effort: uses IMDSv2 to discover the instance, then tags the instance
# and all attached volumes so the seed instance and the /data volume we later
# snapshot are discoverable by project.
tag_resources() {
  local token iid region vol_ids
  token=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null) || return 0
  iid=$(curl -sS -H "X-aws-ec2-metadata-token: ${token}" \
    http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null) || return 0
  region=$(curl -sS -H "X-aws-ec2-metadata-token: ${token}" \
    http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null) || return 0
  vol_ids=$(aws ec2 describe-volumes --region "${region}" \
    --filters "Name=attachment.instance-id,Values=${iid}" \
    --query "Volumes[].VolumeId" --output text 2>/dev/null) || true
  aws ec2 create-tags --region "${region}" \
    --resources ${iid} ${vol_ids} \
    --tags "Key=project,Value=${PROJECT_TAG}" 2>/dev/null || true
  echo "tagged ${iid} ${vol_ids} with project=${PROJECT_TAG}"
}
tag_resources

# --- tooling -------------------------------------------------------------- #
dnf install -y python3-pip >/dev/null 2>&1
pip3 install -q "huggingface_hub>=0.34" hf_transfer >/dev/null 2>&1
export HF_HUB_ENABLE_HF_TRANSFER=1
export PATH="$PATH:/usr/local/bin"
mark "DOWNLOADING"

# --- download from HF to EBS ---------------------------------------------- #
echo "=== downloading ${MODEL} -> ${MODEL_DIR} $(date -u) ==="
hf download "${MODEL}" --local-dir "${MODEL_DIR}" --exclude ".gitattributes"
echo "=== download complete $(date -u) ==="
du -sh "${MODEL_DIR}"
mark "SYNCING_S3"

# --- sync EBS -> S3 (in-region, fast) ------------------------------------- #
echo "=== syncing to ${S3_PREFIX} $(date -u) ==="
aws s3 sync "${MODEL_DIR}" "${S3_PREFIX}" --region us-west-2 --exclude ".cache/*"
echo "=== s3 sync complete $(date -u) ==="
aws s3 ls "${S3_PREFIX}" --region us-west-2 --recursive --summarize | tail -25
mark "DONE"
echo "=== seed finished $(date -u) ==="
