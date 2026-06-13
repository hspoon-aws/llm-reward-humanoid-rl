# Lesson: EBS snapshot lazy-hydration nearly stalled the vLLM bring-up

**Date:** 2026-06-11 (capacity-block sprint, hour 0)
**Where it bit us:** loading Qwen3-Coder-30B weights into vLLM on the `p6-b200.48xlarge`
**Blog section:** "Self-hosting the language model" → bring-up gotchas

## TL;DR

We staged the 57 GB model weights on an EBS snapshot so the capacity-block host
could restore them "instantly" instead of re-downloading. But a volume created
from a snapshot **hydrates its blocks lazily from S3 on first read**, and we had
**not** enabled Fast Snapshot Restore. The first full read of the weights ran at
**~4.4 MB/s** — vLLM sat at "Loading safetensors checkpoint shards 0/16" for 13+
minutes with the GPU idle. At that rate, loading 57 GB would have taken ~3.5 hours
of paid B200 capacity-block time.

The fix that worked: **bypass the cold EBS volume entirely** and pull the same
weights from **S3 → local NVMe instance store** over the S3 gateway endpoint,
then serve vLLM from local NVMe. Throughput climbed past **170 MB/s** immediately.

## How we diagnosed it

The symptom was ambiguous (vLLM "stuck" at shard 0/16, no error). What turned a
guess into a diagnosis:

1. **Process was alive, GPU memory resident, GPU util 0%.** `nvidia-smi` showed
   ~59 GB used on GPU 0 (weights partially resident) but 0% compute — so it was
   blocked on I/O or CPU, not training/compiling.
2. **`py-spy dump` on the EngineCore PID** showed the main thread parked in:
   ```
   _load_w2 (fused_moe/layer.py)
   load_weights (qwen3_moe.py)
   ...
   ```
   i.e. blocked inside **weight loading**, not CUDA-graph capture or torch.compile.
3. **`/proc/<pid>/io` read deltas** quantified it: `read_bytes` advanced ~22 MB in
   5 s ≈ **4.4 MB/s**. That single number proved the bottleneck was disk hydration,
   not the model or the GPU.

> Takeaway: when a load "hangs," confirm liveness (`nvidia-smi`), get a real stack
> (`py-spy dump --pid`), and measure I/O rate (`/proc/<pid>/io`) before changing
> anything. Three cheap signals, one clear root cause.

## Root cause

- A gp3 volume **restored from a snapshot** does not contain the data locally at
  attach time. Each block is fetched from S3 **on first touch** (lazy loading).
- Until a block is hydrated, reads are gated by the hydration path, **far below**
  the volume's provisioned 125 MB/s / 3000 IOPS. Sequentially reading 57 GB of
  cold blocks is close to the worst case.
- **Fast Snapshot Restore (FSR) was not enabled** on the snapshot, so there was no
  pre-warmed full-performance restore in the AZ.

## What actually fixed it

Serve from the instance's **local NVMe** (`/opt/dlami/nvme`, ~28 TB ephemeral),
hydrated straight from S3:

```bash
# stop the stalled server
pkill -f "vllm serve"

# pull weights S3 -> local NVMe over the S3 gateway endpoint (in-region, fast)
aws s3 sync \
  s3://humanoid-from-scratch-123456789012/models/qwen3-coder-30b/ \
  /opt/dlami/nvme/qwen3-coder-30b/ --region us-west-2

# serve from local NVMe instead of the cold EBS mount
MODEL_DIR=/opt/dlami/nvme/qwen3-coder-30b bash scripts/launch_vllm.sh
```

S3→NVMe throughput ramped past 170 MB/s, so the same 57 GB lands in minutes, and
subsequent vLLM reads come off local NVMe at full speed.

## Prevention for next time (in priority order)

1. **Serve weights from local NVMe instance store**, not from a snapshot-restored
   EBS volume. On `p6`/`p5` the instance store is huge and fast; hydrate it from S3
   at boot. This is the simplest robust pattern.
2. **If EBS-from-snapshot is required, enable Fast Snapshot Restore** on the
   snapshot in the target AZ *before* launch (note: FSR has its own cost and a
   warm-up period — enable it ahead of the sprint, not at hour 0).
3. **Pre-warm the volume** if neither of the above: after attach, force-read all
   blocks before launching the consumer, e.g.
   `fio --rw=read --bs=1M --iodepth=32 --name=warm --filename=/dev/nvmeXn1 --direct=1`
   or `dd if=/dev/nvmeXn1 of=/dev/null bs=1M`. This pays the hydration cost up
   front in parallel instead of serially through vLLM's loader.
4. **Provision higher gp3 throughput** (up to 1000 MB/s) for the weights volume —
   helps post-hydration, but does **not** fix cold-read hydration on its own.
5. **Keep S3 as the source of truth.** The recovery only worked because the weights
   were also in `s3://.../models/qwen3-coder-30b/` and the host had an S3 gateway
   endpoint. Always keep that in-region path available.

## Cost / time impact

- Wasted ~15 min of paid capacity-block time chasing the stall.
- The snapshot-restore design (Tasks 0.1/0.2) optimized the *download* but
  introduced a *hydration* cliff that negated the benefit. Net: the local-NVMe-from-S3
  path is both simpler and faster, and should be the default in the runbook.

## Follow-ups

- [ ] Update `scripts/restore_and_launch.sh` / bootstrap to prefer S3→local-NVMe,
      or add a volume pre-warm step after the snapshot mount.
- [ ] Consider enabling FSR on `snap-EXAMPLE00000000` if the EBS path is kept.
- [ ] Reference this lesson in the assembled Blog (Req 21) under LLM self-hosting.
