#!/usr/bin/env bash
# Rehydrate a FRESH B200 (or any CUDA GPU box) after the capacity block ended
# and the old instance was terminated. Pairs with scripts/backup_to_s3.sh.
#
# Assumes the new instance has:
#   - an instance role (or creds) that can read the project S3 bucket
#     (s3:GetObject on models/* and runs-mujoco/*), and
#   - git + python3 + an NVIDIA driver already present (DLAMI provides these).
#
# Idempotent-ish: re-running skips a clone if the repo dir exists.
#
# Usage (on the NEW box):
#   bash restore_from_s3.sh
# Env overrides: BUCKET, PROJECT_DIR, VENV, GIT_URL, GIT_REF.
set -uo pipefail

BUCKET="${BUCKET:-humanoid-from-scratch-123456789012}"
PROJECT_DIR="${PROJECT_DIR:-/data/mujoco}"
VENV="${VENV:-/data/mjxvenv}"
GIT_URL="${GIT_URL:-https://github.com/YOUR_USER/YOUR_REPO.git}"
GIT_REF="${GIT_REF:-main}"
MODEL_DIR="${MODEL_DIR:-/opt/dlami/nvme/qwen3-coder-30b}"

echo "=== [1/5] fetch source code (git) ==="
# The MuJoCo project lives under humanoid-mujoco-from-scratch/ in the repo.
WORK="${WORK:-/data/_restore_repo}"
if [ ! -d "$WORK/.git" ]; then
  git clone --depth 1 --branch "$GIT_REF" "$GIT_URL" "$WORK"
else
  git -C "$WORK" fetch --depth 1 origin "$GIT_REF" && git -C "$WORK" checkout "$GIT_REF"
fi
mkdir -p "$PROJECT_DIR"
# Copy the MuJoCo project into the working PROJECT_DIR (src/config/scripts/etc.)
cp -r "$WORK/humanoid-mujoco-from-scratch/." "$PROJECT_DIR/"
echo "    code at $PROJECT_DIR"

echo "=== [2/5] recreate the MJX venv (pinned versions) ==="
if [ ! -x "$VENV/bin/python" ]; then
  bash "$PROJECT_DIR/scripts/b200_setup_mjx.sh"
else
  echo "    venv already present at $VENV (skipping)"
fi

echo "=== [3/5] pull Qwen model weights from S3 (only if using local vLLM) ==="
# Skip for Bedrock-only runs. Set SKIP_WEIGHTS=1 to skip.
if [ "${SKIP_WEIGHTS:-0}" != "1" ]; then
  mkdir -p "$MODEL_DIR"
  aws s3 sync "s3://${BUCKET}/models/qwen3-coder-30b/" "$MODEL_DIR/" --only-show-errors
  echo "    weights at $MODEL_DIR"
else
  echo "    SKIP_WEIGHTS=1 -> skipping model weight download (Bedrock run)"
fi

echo "=== [4/5] restore prior run artifacts + logs from the box backup ==="
mkdir -p "$PROJECT_DIR/runs"
# Per-iteration artifacts the loop uploaded live under runs-mujoco/<run-id>/...;
# the box-only logs + in-progress checkpoints live under runs-mujoco/_box_backup/.
aws s3 sync "s3://${BUCKET}/runs-mujoco/_box_backup/" "$PROJECT_DIR/runs/" --only-show-errors
echo "    prior runs/logs restored to $PROJECT_DIR/runs"

echo "=== [5/5] sanity checks ==="
"$VENV/bin/python" -c "import jax,mujoco,brax; print('jax',jax.__version__,'mujoco',mujoco.__version__,'devices',jax.devices())" 2>&1 | tail -3 || \
  echo "    WARN: import check failed — inspect the venv / GPU driver"
"$VENV/bin/python" -c "import sys; sys.path.insert(0,'$PROJECT_DIR'); from src.config import load_config; print('config OK:', load_config('$PROJECT_DIR/config/run_config_6gpu_v4_warmstart.yaml').llm_provider)" 2>&1 | tail -2 || \
  echo "    WARN: config load failed"

cat <<EOF

=== RESTORE COMPLETE ===
Next steps:
  - To RESUME a run from its last best checkpoint, relaunch with the same
    --run-id and --checkpoint-dir; the loop resumes from loop_checkpoint.json
    (Req 16.2) and warm-start picks up the best policy.
  - For a Bedrock run, ensure the new instance role has bedrock:InvokeModel on
    the Opus 4.8 inference profile (see infra/instance-role-policy.json + the
    InvokeBedrockClaude statement).
  - For a local-vLLM run, restart vLLM:
      vllm serve $MODEL_DIR --served-model-name Qwen3-Coder-30B-A3B-Instruct \\
        --dtype bfloat16 --max-model-len 32768 --gpu-memory-utilization 0.90 \\
        --tensor-parallel-size 1 --host 0.0.0.0 --port 8000 \\
        --enable-prefix-caching --trust-remote-code
EOF
