#!/usr/bin/env bash
# Snapshot both A/B runs: liveness, GPU util, per-iteration progress, latest metrics.
set -uo pipefail

echo "===== $(date -u +%H:%M:%SZ) ====="
echo "--- processes ---"
pgrep -af "run-1gpu-v4" | head -1 || echo "1GPU run: GONE"
pgrep -af "run-6gpu-v4" | head -1 || echo "6GPU run: GONE"

echo "--- gpu util ---"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# tag -> (dir, log)
for pair in "run-1gpu-v4:run-1gpu-v4" "run-6gpu-v4:run-6gpu-v4"; do
  dir="${pair%%:*}"; rid="${pair##*:}"
  log="/data/mujoco/runs/${rid}.log"
  echo "--- ${rid}: completed-iter checkpoints ---"
  ls -1 /data/mujoco/runs/${dir}/iter_*_metrics.json 2>/dev/null | wc -l
  echo "--- ${rid}: latest train-progress ---"
  grep "train-progress" "$log" 2>/dev/null | tail -1 || echo "(no progress line yet)"
  echo "--- ${rid}: latest iter status ---"
  grep -E "\[iter\]" "$log" 2>/dev/null | tail -3 || true
  echo "--- ${rid}: run summary (if finished) ---"
  grep -E "best policy|iterations attempted" "$log" 2>/dev/null | tail -2 || true
done
echo "===== end ====="
