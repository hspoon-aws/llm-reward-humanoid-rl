# Lesson: don't build the Docker image on the paid GPU host

**Date:** 2026-06-11 (Phase 2 Stage A bring-up)
**Where it bit us:** building the Isaac Lab runtime image on the `p6-b200.48xlarge` capacity block
**Blog section:** "Running Isaac Lab simulation" → bring-up cost gotchas

## TL;DR

We ran `docker build -t humanoid-from-scratch:latest .` **on the B200 capacity-block instance**.
It took ~10–12 min (pull the ~7.5 GB `nvcr.io/nvidia/isaac-lab:2.2.0` base + apt + `pip install
vllm`), produced a 41.4 GB image, and used **0% GPU** the whole time. That is paid Blackwell time
spent on `docker pull` + `apt-get` + `pip` — none of which needs a GPU. On a 24h paid block, every
rebuild is real money for work that belongs on a cheap CPU box.

## What actually happened

- `docker build` is **CPU + network + disk bound**. During the build `nvidia-smi` showed the GPUs
  idle; the cost was base-layer pull and dependency install.
- The Dockerfile `COPY src/ ...` near the end means **any code edit triggers a rebuild** — and
  Phase 2 shakeout *will* edit code to fix Isaac Lab API drift, so this isn't a one-time cost.
- Net: the most expensive compute we rent sat at 0% util while we downloaded Ubuntu packages.

## The learning

**Only run on the GPU box what actually needs CUDA** (vLLM inference, Isaac Lab/RSL-RL training).
Everything else — image build, dependency install, code sync — should be pre-staged or done off
the clock. The image is portable; the GPU is the scarce, expensive resource.

## What we'd do instead (for next time)

1. **Build off-GPU, pull on-GPU.** Build the image on a cheap CPU instance / CodeBuild, `docker
   push` to **ECR**, and have the B200 do a single in-region `docker pull`. Pull is
   bandwidth-bound (seconds–low minutes); build is minutes of CPU+net on the wrong machine.
2. **Build once, then `docker save` it.** We already paid for this 41.4 GB image —
   `docker save humanoid-from-scratch:latest | zstd | aws s3 cp s3://.../runs/cache/images/`, and
   future instances `docker load` instead of rebuilding. Even better, bake it into the AMI/EBS so
   the next capacity-block instance boots with it present (same "pay the one-time cost off the
   clock" pattern as the model weights and the vLLM/FlashInfer caches).
3. **Bind-mount the repo for code iteration.** Run the container with
   `-v /opt/humanoid:/opt/humanoid-from-scratch` during shakeout, so a code fix is `aws s3 cp` +
   re-run — **no image rebuild**. Reserve the Dockerfile `COPY` for the final immutable run.
4. **Keep layer order cheap.** System deps → pip (pinned, e.g. `vllm==0.11.0`) → `COPY src/` last,
   so a code-only change rebuilds only the tiny COPY layers, never the multi-GB pip layer. (The
   Dockerfile already does this — keep it.)
5. **Parallelize the wait.** Any GPU-idle wait (build, pull, warmup) is time for off-GPU work
   (entrypoint review, committing, planning) — not time to watch a progress bar. We did this
   correctly this round.

## Connection to the other hour-0 lessons

This is the same theme as `lesson-ebs-snapshot-hydration.md` and `lesson-flashinfer-jit-blackwell.md`:
a one-time, non-GPU cost (weight hydration, kernel JIT, image build) that we paid at GPU rates
because we discovered it live instead of pre-staging it. The fix is always the same — **identify
what truly needs the GPU, pre-stage everything else.**

## Status / follow-ups (not done yet — documented for later)
- [ ] `docker save` the current image to S3 so this block never rebuilds it.
- [ ] Bind-mount the repo for Stage A/C iteration instead of rebuilding on code edits.
- [ ] (Future) Move image builds to a cheap CPU host + ECR; bake the image into the AMI/EBS.
