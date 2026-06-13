#!/usr/bin/env bash
set -euo pipefail

MODEL="Qwen/Qwen3-Coder-30B-A3B-Instruct"
STAGING="/hf-staging/qwen3-coder-30b"
BUCKET="humanoid-from-scratch-123456789012"
S3_PREFIX="s3://${BUCKET}/models/qwen3-coder-30b/"
PROFILE="your-aws-profile"
REGION="us-west-2"
PROJECT_TAG="humanoid-llm-rl-poc"

export HF_HUB_ENABLE_HF_TRANSFER=1

echo "=== [$(date)] Starting HF download of ${MODEL} ==="
hf download "${MODEL}" \
    --local-dir "${STAGING}" \
    --exclude ".gitattributes"

echo "=== [$(date)] Download complete. Local contents: ==="
du -sh "${STAGING}"
ls -la "${STAGING}"

echo "=== [$(date)] Syncing to ${S3_PREFIX} ==="
aws s3 sync "${STAGING}" "${S3_PREFIX}" \
    --profile "${PROFILE}" \
    --region "${REGION}" \
    --exclude ".cache/*"

echo "=== [$(date)] S3 sync complete. Verifying: ==="
aws s3 ls "${S3_PREFIX}" --profile "${PROFILE}" --region "${REGION}" --recursive --summarize | tail -25

echo "=== [$(date)] Applying project tag to bucket ==="
aws s3api put-bucket-tagging \
    --bucket "${BUCKET}" \
    --tagging "TagSet=[{Key=project,Value=${PROJECT_TAG}}]" \
    --profile "${PROFILE}" \
    --region "${REGION}"

echo "=== [$(date)] ALL DONE ==="
