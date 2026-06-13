#!/usr/bin/env bash
#
# Persist the warmed vLLM/FlashInfer compile caches to S3 so a future launch on
# an identical p6-b200 (same vLLM/FlashInfer/torch versions + model) skips the
# ~13 min Blackwell JIT + torch.compile warmup. Run ONCE after a successful
# first vLLM launch. See docs/lesson-flashinfer-jit-blackwell.md.
set -euo pipefail
BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
S3_CACHE_PREFIX="${S3_CACHE_PREFIX:-s3://${BUCKET}/runs/cache}"
TOKEN=$(curl -sS -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 300" 2>/dev/null || true)
REGION="${REGION:-$(curl -sS -H "X-aws-ec2-metadata-token: ${TOKEN}" http://169.254.169.254/latest/meta-data/placement/region 2>/dev/null || echo us-west-2)}"

echo "=== syncing compile caches -> ${S3_CACHE_PREFIX} (region ${REGION}) ==="
aws s3 sync /root/.cache/vllm/       "${S3_CACHE_PREFIX}/vllm/"       --region "${REGION}" --only-show-errors && echo "VLLM_CACHE_SAVED"
aws s3 sync /root/.cache/flashinfer/ "${S3_CACHE_PREFIX}/flashinfer/" --region "${REGION}" --only-show-errors && echo "FLASHINFER_CACHE_SAVED"
echo "=== done ==="
