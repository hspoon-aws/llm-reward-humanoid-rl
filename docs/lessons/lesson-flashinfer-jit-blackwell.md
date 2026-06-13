# Lesson: vLLM's second "stall" was FlashInfer JIT-compiling a Blackwell kernel

**Date:** 2026-06-11 (capacity-block sprint, hour 0)
**Where it bit us:** vLLM startup on `p6-b200.48xlarge` (8× B200, sm_100 / Blackwell), right after weights loaded
**Blog section:** "Self-hosting the language model" → Blackwell / first-run kernel compilation

## TL;DR

After we fixed the EBS hydration problem (see `lesson-ebs-snapshot-hydration.md`),
weights loaded in ~13 s from local NVMe — but vLLM *still* didn't serve for several
more minutes. The log sat at `torch.compile took 26.31 s` with the GPU at **0%**.
It looked like a second hang.

It wasn't. A `py-spy dump` showed the engine was **JIT-compiling a CUDA kernel**
for Blackwell on first use — FlashInfer building its `sm100` TRTLLM MoE kernel via
`ninja`. New GPU architecture + first run = on-the-fly kernel compilation. The right
move is to **wait it out the first time** and **persist the compile cache** so it
never happens again.

## How we diagnosed it (same playbook, different culprit)

Reused the three-signal playbook from the hydration lesson:

1. **GPU resident but 0% util** (`nvidia-smi`: ~60 GB used, 0% compute) → blocked on
   CPU/build, not running kernels.
2. **`py-spy dump --pid <EngineCore>`** gave the smoking gun — the main thread was in:
   ```
   communicate (subprocess.py)            # waiting on a child process...
   run_ninja (flashinfer/jit/cpp_ext.py)  # ...which is ninja, compiling C++/CUDA
   build_and_load (flashinfer/jit/core.py)
   get_trtllm_moe_sm100_module (flashinfer/fused_moe/core.py)   # <-- sm100 = Blackwell
   trtllm_bf16_moe (...)
   apply (vllm/.../fused_moe/experts/trtllm_bf16_moe.py)
   ```
   `run_ninja` + `get_trtllm_moe_sm100_module` = FlashInfer is **compiling** the
   Blackwell MoE kernel, not hung.
3. The earlier log line `torch.compile took 26.31 s` confirmed we were already past
   weight load and into the compile/kernel-warmup phase.

> Same lesson as before: confirm liveness, get a real stack with `py-spy`, read the
> frames. The stack literally names the subsystem (`flashinfer.jit`) and the target
> arch (`sm100`).

## Root cause

- The B200 is **sm_100 (Blackwell)**. Prebuilt kernels for newer arches are often
  **not** shipped in the wheels; FlashInfer (and parts of vLLM/torch.compile)
  **JIT-compile** them **on first execution** for the exact GPU + dtype + shape.
- That first compile is single-shot, CPU-bound (`ninja`), and silent from the GPU's
  perspective — hence "0% util, looks stuck."
- It is **first-run only**. The artifacts are cached:
  - vLLM torch.compile: `~/.cache/vllm/torch_compile_cache/...`
  - FlashInfer JIT: its `ninja` build cache (under the FlashInfer cache dir).

## What to do

1. **Be patient on the very first launch.** Budget several minutes after
   "weights loaded" for `torch.compile` + FlashInfer JIT on a brand-new arch.
   Don't kill it; verify with `py-spy` that it's in `run_ninja`/compile frames.
2. **Persist the caches so subsequent launches are fast.** These directories must
   survive instance/process restarts:
   - `~/.cache/vllm/` (torch.compile + AOT graphs) — ~49 MB on this run
   - `~/.cache/flashinfer/` (Blackwell sm100 JIT kernels) — ~91 MB on this run
   We synced both to S3 after the first successful launch (see "What we did" below).
   Restore them before launching vLLM on a fresh instance and the 782 s warmup is skipped.
3. **Optional: reduce first-launch compile cost** while iterating (not for the final
   throughput run):
   - `--enforce-eager` skips CUDA-graph capture (slower steady-state, faster start),
   - trimming `compile` / cudagraph capture sizes reduces the number of graphs built.
   Keep full compile for the actual sprint; use these only to shorten dev cycles.
4. **Pre-warm as a deploy step.** After launch, send one tiny completion request and
   wait for it to return; that forces all first-use kernels to compile before you
   depend on the endpoint for the smoke test / orchestrator.

## Why it matters here

On a paid capacity block, two separate multi-minute "stalls" at hour 0 (EBS
hydration, then JIT compile) eat real money and nerve. Neither was a real failure —
but without `py-spy` they're indistinguishable from a hang, and the temptation is to
kill/restart, which on the JIT case just **throws away compile progress and restarts
the same compile**. Recognize the signature, let it finish once, and cache the result.

## Follow-ups

- [ ] Add the vLLM torch.compile cache and FlashInfer JIT cache to the weights
      snapshot/AMI so the next launch skips compilation.
- [ ] Add a post-launch "warm-up request" step to the bring-up runbook before the
      smoke test.
- [ ] Reference this lesson in the assembled Blog (Req 21) under LLM self-hosting,
      paired with the EBS hydration lesson as "two hour-0 stalls that were not hangs."
