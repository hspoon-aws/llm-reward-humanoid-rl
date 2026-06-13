# Multi-GPU run recipe (validated data-parallel Brax PPO on the B200)

**Status:** the data-parallel sharding path is **validated live** — Brax PPO trained across 4
B200s (512 envs/device × 4 = 2048) with the trainer's batch derivation, checkpoint, and
`goal/*` eval metrics all working (`scripts/smoke_test_multigpu.py` → `MULTI-GPU SMOKE: PASS`).
The first full run used a single GPU (the only config validated at the time); this recipe is
for the **next** run to use all the spare silicon.

## Why multi-GPU helps (and why it's the right kind of parallelism)

The Eureka loop is **sequential across iterations** (iteration N's reward depends on N-1's
metrics), so you cannot parallelize *iterations*. But a single PPO training is **data-parallel**:
Brax shards `num_envs` across visible devices, so more GPUs = more parallel envs = more samples
per step = faster convergence per wall-clock iteration. With GPU 0 reserved for vLLM, that's
**GPUs 1-7 = 7 devices** available for one training.

On the first single-GPU run, 6 of 8 B200s sat idle for hours. Multi-GPU fixes that.

## The two hard constraints (both Brax asserts; see lessons §5.3, §5.4)

1. `num_envs % device_count == 0` — Brax shards envs evenly across devices.
2. `batch_size * num_minibatches % num_envs == 0` — the trainer derives
   `num_minibatches=8`, `batch_size = (num_envs // 8) * 8`, falling back to
   `batch_size=num_envs, num_minibatches=1` if needed, so this holds for any `num_envs`.

**The rule that satisfies both:** pick `num_envs = K × device_count`. For 7 GPUs use a multiple
of 7 (e.g. `7 × 512 = 3584`, or `7 × 1024 = 7168`). Do NOT use 2048/4096 with 7 GPUs — not
divisible by 7 → assert #1 fires.

## Launch command (7 GPUs, vLLM stays on GPU 0)

```bash
# On the B200, in the MJX venv. num_envs MUST be a multiple of 7.
cd /data/mujoco
nohup env \
  PROJECT_ROOT=/data/mujoco \
  CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 \
  XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  MUJOCO_GL=egl \
  /data/mjxvenv/bin/python -m src.run_loop \
    --config config/run_config.yaml \
    --run-id full-run-7gpu \
    --checkpoint-dir /data/mujoco/runs/full7 \
  > /data/mujoco/runs/full-run-7gpu.log 2>&1 &
```

Set `training.num_envs: 7168` (and `oom_fallback_num_envs: 3584`) in
`config/run_config.yaml` first — both multiples of 7.

## Validation before the real run (cheap, on free GPUs only)

Never smoke-test on GPU 0 (vLLM) or a GPU running a live job. Use idle GPUs:

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5 XLA_PYTHON_CLIENT_MEM_FRACTION=0.4 MUJOCO_GL=egl \
  /data/mjxvenv/bin/python scripts/smoke_test_multigpu.py   # → MULTI-GPU SMOKE: PASS
```

For the exact 7-GPU count, run the smoke with `CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7` **only when
no full run is using GPU 1** (e.g. before launching, or between runs). The smoke auto-picks
`num_envs = 512 × device_count`, so with 7 visible it uses 3584 — exactly the divisibility you
need to confirm.

## Expected payoff

7× the parallel envs per step vs single-GPU. PPO sample throughput scales close to linearly with
devices in Brax/MJX, so wall-clock per iteration drops substantially and the policy sees far
more experience per iteration — a better shot at actually reaching the goal within the epoch
budget. Capture the throughput delta (steps/s single vs 7-GPU) as a comparison-blog metric
(Req 21.5).

## Caveats

- **VRAM:** at `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` each B200 (183 GB) gives ~155 GB to JAX;
  3584-7168 envs of H1 fit comfortably (the 2048-env single-GPU run used ~156 GB total).
- **vLLM headroom:** keep vLLM on GPU 0 only; never add GPU 0 to `CUDA_VISIBLE_DEVICES` for
  training (Req 8.2 / 18.5 — the config loader already rejects GPU 0 in `training_gpus`).
- **Determinism:** more devices changes the effective batch composition; metrics won't match the
  single-GPU run exactly. That's expected, not a bug.
