# Lesson: bringing up Isaac Lab for the goal-reaching loop

**Date:** 2026-06-11 (pre-bring-up; verified learnings + live watch-list)
**Where it bites:** Phase 2 Isaac Sim / Isaac Lab setup
**Blog section:** "Running Isaac Lab simulation" → environment bring-up

> **TL;DR / how this doc reads.** It's chronological. An early section concludes the
> data-center Blackwell box (no RT cores) **cannot run Isaac Sim at all** — that finding
> is correct. The project then moved the simulator to a **`g6`/`g6e` (NVIDIA L4 / L40S,
> Ada, has RT cores)** instance, where the hour-0 smoke ultimately **PASSES end-to-end**
> (see the final "RESOLUTION" section). So: Isaac Sim runs on the g6/L4, not on the
> data-center card. Read to the end before quoting any single section.

> Status: this doc has TWO confirmed learnings (verified off-GPU this session) and a
> **watch-list** of things expected to bite during the live Stage A/C bring-up. The
> watch-list items are filled in with real diagnoses once we run on the box — same
> way `lesson-flashinfer-jit-blackwell.md` was written *after* it bit us. Do not
> present watch-list items as if they already happened.

## Confirmed learning 1 — the H1 task id is not guessable

The intuitive name `Isaac-Velocity-Flat-Unitree-H1-v0` is **wrong**. The Unitree H1
velocity task registers as:

```
Isaac-Velocity-Flat-H1-v0          # flat terrain (what we reframe)
Isaac-Velocity-Rough-H1-v0         # rough terrain
Isaac-Velocity-Flat-H1-Play-v0     # play/eval variant
```

— **no `Unitree-` segment**, even though the robot is a Unitree H1 (the Go2 / G1
tasks *do* carry vendor segments, which is what makes this easy to get wrong).
Confirmed against Isaac Lab source:
`IsaacLab/source/isaaclab_tasks/.../locomotion/velocity/config/h1/__init__.py`
(`gym.register(id="Isaac-Velocity-Flat-H1-v0", ...)`).

**Takeaways:**
- Task-registry ids drift between Isaac Lab versions and are **not** guessable from
  the robot name. Never hardcode; verify against the *installed* registry.
- We made the id a **config field** (`training.task` in `run_config.yaml` →
  `Config.env_id` → `TrainConfig.env_id` / `EvalConfig.env_id`). A registry-name
  mismatch is now a one-line config edit, not a code change.
- Verify inside the actual container before depending on the string:
  ```bash
  isaaclab.sh -p -c "import isaaclab_tasks, gymnasium as gym; \
    print(sorted(e for e in gym.envs.registry if 'H1' in e))"
  ```

## Confirmed learning 2 — make the RL pipeline import + test with zero Isaac Sim

The entire Phase 2 stack (trainer, evaluator, env wiring, reward binding, orchestrator)
imports and unit-tests on a laptop with **no Isaac Sim, no GPU** — 250 tests. The trick:
every Isaac Lab / RSL-RL / gymnasium / torch import is **lazy and guarded inside one
adapter boundary**:
- training behind `build_isaaclab_trainer` → `_IsaacLabTrainer` (imports inside `__init__`),
- evaluation behind `IsaacLabPolicyRollout` / `IsaacLabDemoVideoRecorder` (imports inside methods),
- both reached through a dependency-injection seam so tests pass in-memory fakes.

On a controller host the factories raise a clear `RuntimeError` ("inject a trainer
instead") rather than an `ImportError` mid-run.

**Takeaways:**
- Isolate the simulator to one adapter class; keep all control flow, recovery logic,
  reward math, and config validation pure and host-testable.
- This is *why* we could complete and verify Phase 2 before the GPU instance existed —
  the simulator is only needed at `train()` / `evaluate()`, never at import.
- Cost: the adapter bodies are `# pragma: no cover`; their real API assumptions are
  **unverified until the live run** (see watch-list).

## Watch-list for the live bring-up (Stage A/C) — fill in when it bites

These are the High/Medium risks from `phase2-ec2-run-plan.md`, framed as "what to look
for + where it'll show up." Capture the real traceback + fix here as each is hit.

- [ ] **`isaaclab_tasks` / H1 id actually registered.** The bare instance has *no Isaac
      env*; it arrives only via the `nvcr.io/nvidia/isaac-lab:2.2.0` container. Confirm
      `import isaaclab_tasks` works and the H1 id from learning 1 is present. If the 2.2.0
      image names it differently, edit `training.task`.
- [ ] **`parse_env_cfg` call shape.** `_IsaacLabTrainer` tries `device=`/`num_envs=` then
      falls back to `use_gpu=`. Record which the installed version wants.
- [ ] **RSL-RL vec-env wrapper import path.** Code tries `isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`
      then legacy `omni.isaac.lab_*`. Record the real path.
- [ ] **`OnPolicyRunner` train-cfg schema (HIGHEST RISK).** The `_build_train_cfg` dict
      (policy / algorithm sub-dicts) is a guess against rsl_rl's expected schema. Most
      likely thing to need a tweak — capture the exact error + corrected dict.
- [ ] **`attach_goal_conditioning` vs. the real H1 cfg.** Confirm the stock obs group and
      `track_lin_vel_*` / `track_ang_vel_*` reward term names match so the Goal_Observation
      append and reward-zeroing actually land.
- [x] **AppLauncher first-run.** RESOLVED: launches headless on the L4 in ~25s (GPU foundation
      init + benign ECC/PCIe warnings); the only blocker was the NumPy ABI crash (learning 3).
      `parse_env_cfg` / RSL-RL wrapper / `OnPolicyRunner` cfg schema / `attach_goal_conditioning`
      remain unverified and surface at Stage C.
- [x] **`--network host` loopback to host vLLM.** CONFIRMED working: `curl
      http://127.0.0.1:8000/v1/models` from inside the container returned the host vLLM's
      model list (shared netns). Topology option A is validated.
- [ ] **GPU-0 exclusion lands.** Verify with `nvidia-smi` that Isaac Lab training occupies
      GPUs 1–7 only and vLLM keeps GPU 0. (Blocked behind the CUDA-13 incompatibility below.)

## Confirmed learning 3 — the REAL Stage-A blocker was a NumPy 1.x/2.x ABI crash (not RT cores)

Both the B200 and the g6/L4 crashed identically during `AppLauncher(headless=True)` with a
segfault unwinding through `pinocchio_pywrap` → `eigenpy::enableEigenPy()` → boost.python. The
head of the crash (not the unwind) names the true cause:

```
isaaclab/devices/openxr/retargeters/...gr1_t2_dex_retargeting_utils.py
  -> dex_retargeting.robot_wrapper -> import pinocchio
  -> pinocchio_pywrap ...
ImportError: A module that was compiled using NumPy 1.x cannot be run in NumPy 2.2.6
             as it may crash. ... downgrade to 'numpy<2' ...
```

**Root cause:** importing the `isaaclab` namespace eagerly pulls in an OpenXR teleop retargeter
(`dex_retargeting` → `pinocchio`). The bundled `pinocchio_pywrap`/`eigenpy` C-extensions are
compiled against **NumPy 1.x**, but our Dockerfile's `pip install vllm` upgraded the Isaac Sim
interpreter's NumPy to **2.2.6** → the native module **segfaults at load**, taking down
AppLauncher before any task registers.

**Fix (verified — Stage A passes on the L4):** pin `numpy==1.26.4` in the Isaac Sim python after
the vLLM install. Now baked into the Dockerfile. With it, Stage A prints:
```
H1_IDS: ['Isaac-Velocity-Flat-H1-Play-v0', 'Isaac-Velocity-Flat-H1-v0',
         'Isaac-Velocity-Rough-H1-Play-v0', 'Isaac-Velocity-Rough-H1-v0']
HAS_EXPECTED[Isaac-Velocity-Flat-H1-v0]: True
STAGE_A: PASS
```

**Correction to the earlier RT-core conclusion:** the crash itself was NOT caused by missing RT
cores — it was this NumPy ABI conflict, identical on both GPUs. HOWEVER the RT-core limitation is
still real and separate: data-center cards (A100/H100/**B200**) lack RT cores, which Isaac Sim's
**RTX renderer / RTX sensors** require. So:
- **Headless physics training** (what PPO needs) may run on a non-RTX card once the NumPy crash
  is fixed — but is unsupported/untested by NVIDIA and we did not get far enough on the B200 to
  prove it (we moved to the L4).
- **The L4 (g6.4xlarge) HAS RT cores** and is the supported, working choice — Stage A passed
  there. This is the path.

The lasting lesson is the FlashInfer one again: a segfault's *unwind* (pinocchio/eigenpy) is not
the *cause*; read the head of the crash (the NumPy ImportError) to find it.

## Confirmed learning 4a — B200 lacks RT cores; the L4/g6 is the right sim host

(Still true and the reason we run the sim on the g6, independent of the NumPy fix above.)

**Options (decision needed — do NOT brute-force live):**
- **A. Newer Isaac Lab image (RECOMMENDED).** Our pull was `isaac-lab:2.2.0` = Isaac Sim
  **5.0.0-rc.45** — a release *candidate*. Per the current Isaac Sim requirements docs (updated
  2026-06-05), the GA line targets **Linux driver 580.95.05** (our host: 580.159.04 ✓) and lists
  **Blackwell** GPUs as supported, and newer tags exist (Isaac Sim 5.1 / Isaac Lab **2.3**, and
  6.0.0). The rc build simply predates B200/CUDA-13 validation. Try `nvcr.io/nvidia/isaac-lab`
  tags `2.3.x` (Sim 5.1) or the latest GA — rebuild the image with `--build-arg ISAACLAB_TAG=...`.
- **B. Older driver / CUDA-12 path** — not viable: cannot downgrade the 580 driver on a
  capacity-block host, and B200 needs a recent driver.
- **C. Verify B200 is actually supported, not just "Blackwell".** The docs note A100/H100 (no RT
  cores) are unsupported; B200 is data-center Blackwell and likewise lacks RT cores. Isaac Sim's
  rendering wants RT cores — **headless physics-only training may still work, but demo-video
  rendering (Req 10) may not.** Confirm against NVIDIA's data-center support matrix before
  committing the full run. This is a material risk worth checking even with a newer image.
- **D. Bare-metal (no container) Isaac Lab install** against the host CUDA-13 stack — more work,
  uncertain, loses the reproducible image.
- Until resolved, **Stage C smoke and Stage D are blocked.** Phase 1 (vLLM + reward generation)
  remains fully functional; the loop's non-sim half is unaffected.

## Confirmed learning 4b — the data-center Blackwell box CANNOT run Isaac Sim at all (hardware, not version)

Investigating whether a newer image would help surfaced a **hard architectural blocker that no
image change fixes**: Isaac Sim requires **RT cores** (RTX), and data-center GPUs — A100, H100,
H200, H20, **and B200** — have **no RT cores and no NVENC**. NVIDIA's own developer forum is
unambiguous and repeated across threads:

- "GPUs without RT Cores (such as H100, A100, and H20) are not supported for Isaac Sim."
- On an H100: "An H100 is for compute ONLY. It is not suitable for anything Isaac Sim. This will
  not work. You need an A5000, A6000 or a L40/L40s." (NVIDIA forum, SOLVED, Jun 2026)
- The Isaac Sim requirements page lists consumer/workstation RTX (RTX 4080/5080, RTX PRO 6000
  Blackwell) — never data-center cards — and explicitly notes A100/H100 are unsupported.

The `pinocchio`/`eigenpy` segfault and Warp CUDA-13 errors we saw are downstream symptoms; the
**root cause is that the `p6-b200.48xlarge` is the wrong hardware class for Isaac Sim.** Blackwell
*workstation* cards (RTX PRO 6000 Blackwell) have RT cores and are supported; Blackwell
*data-center* cards (B200) do not and are not. "Blackwell is supported" in the docs refers to the
RTX PRO line, not B200.

**Implication for this project:** Phase 2 (Isaac Lab H1 training + the whole sim half of the
Eureka loop) **cannot run on this capacity block, with any image.** This is not a Stage-A bug to
fix — it is a hardware/software-stack mismatch baked into the instance choice. The original
PROJECT-PLAN risk "vLLM on B200 incompatibility" was watched; the unwatched risk was **Isaac Sim
on B200**, which is the one that actually bites.

**Why the off-GPU prep could not catch it:** the simulator only loads on the GPU box, and the
incompatibility is at RTX/Vulkan init — exactly the surface that is `# pragma: no cover` and
unverifiable until launch. The cheap Stage-A probe caught it in ~30s of GPU time, which is the
system working as designed: fail the bring-up gate before any 4096-env training run.

### Options now (real, not cosmetic)
1. **Run Phase 2 on an RTX GPU instance instead of the B200 (CONFIRMED FIX).** AWS's own
   "NVIDIA Isaac Lab on AWS" workshop runs *this exact task* (Unitree H1 humanoid RL on Isaac
   Lab) on a **`g6.4xlarge`**: 16 vCPU, 128 GiB, **1× NVIDIA L40S** (Ada, has RT cores ✓),
   Ubuntu 22.04, with NVIDIA driver/toolkit + ROS2 + Amazon DCV pre-installed, ~$3/hr on-demand.
   That is the supported, validated hardware for the sim half. Other RT-core options:
   `g6e` (L40S, more VRAM), `g6` (L4), `g5` (A10G). **This is the path.**
2. **Keep the B200 only for the LLM half** (Phase 1, already working). Two viable topologies:
   (a) vLLM stays on the B200 and the g6 reaches it cross-host (needs a network path, breaks the
   loopback assumption — would require a private route + auth), or (b) **run vLLM on the g6 too**
   — a 30B-MoE (~3B active) fits the L40S (48 GB) at reduced throughput, keeping the clean
   single-host loopback design. For a from-scratch g6 run, (b) is simpler and matches the design.
3. Swap the simulator to a non-RTX engine — unnecessary now that g6/L40S is confirmed.
4. **Release the B200 capacity block** once the LLM-side work to preserve (weights, caches,
   Phase 1 validation) is captured to S3 — to stop the per-minute burn, since the sim half will
   run on the cheaper g6.

**Recommended plan:** move Phase 2 to a `g6.4xlarge` (L40S) per the AWS workshop; run vLLM +
the Eureka loop co-resident there (option 2b) so the `--network host` loopback design carries
over unchanged; release the B200. The Phase-1 artifacts (weights in S3, validated reward
generation) transfer directly — only the GPU under the sim changes.

## Why it matters here

On a paid capacity block, the env bring-up is the riskiest, least-rehearsed phase: the
simulator API surface is unverified by construction (we couldn't import it off-GPU). The
hour-0 `--smoke` run exists precisely to flush these out cheaply before the full block.
Recognize that an Isaac Sim "hang" on first launch is most likely compilation/warmup, not
a crash — same playbook as the FlashInfer lesson: confirm liveness, get a real stack, read
the frames.

## Follow-ups

- [ ] Fill each watch-list item with the real diagnosis during Stage A/C.
- [ ] Reference this lesson in the assembled Blog (Req 21) under "Running Isaac Lab
      simulation," paired with the FlashInfer + EBS hydration lessons as the hour-0
      bring-up trilogy.

---

## Stage C live findings (g6/L4) — the bring-up bug chain

Running the hour-0 smoke (`run_loop --smoke`, 64 envs, 5 epochs) on the g6/L4 surfaced a
**chain** of blockers — each fix revealed the next, exactly as the run plan predicted for the
`# pragma: no cover` Isaac Lab adapter. Logged in the order hit:

1. **CUDA masking (FIXED).** `PPO_Runner` set `CUDA_VISIBLE_DEVICES=1..7` (the 8-GPU B200
   layout) on a single-GPU L4 whose only device is 0 → `No CUDA-capable device detected` /
   `RuntimeError: No CUDA GPUs are available`. Root cause: config carried `training_gpus=[1..7]`.
   Fix: `config/run_config.g6.yaml` (`training_gpus=[0]`, cross-host vLLM endpoint, scaled envs).
   *Lesson:* the GPU-allocation assumptions are host-shaped; a single-GPU host needs its own config.

2. **GPU-0 reservation (FIXED).** Both the config loader (Req 18.5) and
   `select_cuda_visible_devices` hard-excluded GPU 0 (reserved for co-resident vLLM), so `[0]`
   reduced to empty. Fix: relax the exclusion when the LLM endpoint is **cross-host**
   (non-loopback) — `_endpoint_is_loopback()` gate in both places; `TrainConfig.from_config`
   derives `llm_gpu=-1` for cross-host. *Lesson:* a "reserved GPU 0" invariant only holds when
   the LLM is actually on this host.

3. **Goal-buffer ordering (FIXED).** The Goal_Observation term is evaluated **during
   `gym.make`**, before the trainer attaches the per-env `GoalBuffer` → `AttributeError: env
   exposes no 'goal_buffer'`. Fix: `_resolve_goal_buffer` lazily builds+attaches the buffer from
   the cfg-stashed Goal on first use. *Lesson:* manager-based env constructs observations at
   build time; anything an obs term reads must exist before `gym.make`, not after.

4. **Reward-term registration (FIXED).** `LiveRewardBinding` used
   `RewardManager.set_term_cfg(name, term)`, but that Isaac Lab API only **updates an existing**
   term and raises `Reward term 'llm_generated_reward' not found` for a new one. Fix: append to
   the manager's `_term_names`/`_term_cfgs` and seed the `_episode_sums` buffer for a genuinely
   new term; keep `set_term_cfg` only for updates / fake managers. *Lesson:* "set cfg" ≠ "add
   term" in Isaac Lab's managers.

5. **`isaaclab_tasks` typing crash (NOT FIXED — root-caused, needs adapter change).**
   `parse_env_cfg`/`gym.make` import the task registry through the **kit extension autoloader**
   (`omni.ext`), which fails with:
   `TypeError: Type parameter ~_T1 without a default follows type parameter with a default`
   from `typing_extensions._collect_parameters`. Key observations:
   - The base Isaac Sim 5.0-rc image already ships `typing_extensions==4.15.0`; pinning to
     `4.12.2` did **not** help (the strict PEP-696 ordering check exists across 4.x).
   - **Crucially, `import isaaclab_tasks` DIRECTLY in Python WORKS** — Stage A imported it and
     listed the H1 ids fine. Only the **kit-extension autoload path** (triggered by
     `parse_env_cfg`/`gym.make`) hits the crash.
   - So this is **not** a simple version pin: it is an Isaac Sim 5.0-rc + Python 3.11 +
     typing_extensions incompatibility specific to how the extension manager imports the task
     package. The real fix is in `_IsaacLabTrainer`: import the task registry directly (the way
     `stage_a_check.py` does) and/or avoid the `omni.ext` autoload, or move to an Isaac Lab GA
     image where this is resolved. Deferred rather than brute-forced on the paid clock.

### What did NOT block (confirmed working)
- AppLauncher headless launch on the L4 (~15–25s, no crash) after the NumPy pin.
- The H1 task id `Isaac-Velocity-Flat-H1-v0` is registered (direct import).
- Cross-host vLLM over the VPC (`g6 → B200 :8000`) — the loop's LLM half is reachable.
- The Warp `cuDeviceGetUuid` CUDA-13 message is a **warning**, not the failure.

### Recording / artifacts
No recording or artifacts were produced: every Stage C attempt failed during env construction
(`gym.make`/registry load), **before** any reward generation, training epoch, checkpoint, or
demo-video. `s3://.../runs/` contains only `cache/` and `code/` — no `stage-c-smoke/` iteration
artifacts. So there was nothing to inspect on the recording side; the loop never reached the
Evaluator's video step (Req 10).

### Net status of Stage C
- **5 of 6 blockers fixed** (committed). The 6th (kit-extension typing crash) is root-caused and
  has a clear next action (direct task-registry import in the trainer, or an Isaac Lab GA image),
  but is **not yet fixed** — so the hour-0 smoke has **not** produced a checkpoint and Stage D
  remains blocked.
- All fixes preserve the 250-test controller suite (still green) and the default 8-GPU config
  behavior (GPU 0 still reserved for loopback vLLM).

## Lessons (summary)
- **The adapter boundary is where reality hits.** Everything host-testable passed; every Stage C
  blocker was in the `# pragma: no cover` Isaac Lab adapter (`_IsaacLabTrainer`, live reward
  binding, goal-env wiring) — exactly the code that could not be exercised off-GPU. Budget for a
  *chain* of these, not a single fix.
- **Read the head of the crash, not the unwind** — held again (numpy→pinocchio, typing→omni.ext).
- **Dep upgrades from one tool silently break another co-installed stack.** vLLM's installs
  perturbed NumPy (fixed) and the typing stack; co-hosting a 30B LLM and Isaac Sim in one image
  is a version-conflict minefield. A cross-host split (vLLM on its own box) sidesteps most of it.
- **Single-GPU vs multi-GPU is a first-class config axis**, not a tweak — device masking and
  GPU reservation invariants must be host-shaped.
- **Direct import beats extension autoload** for the task registry: `stage_a_check.py`'s plain
  `import isaaclab_tasks` worked where `gym.make`'s `omni.ext` autoload crashed. The trainer
  should follow the former.

---

## Follow-up: fixes for blocker #6 + recording (committed, PENDING GPU re-validation)

After Stage C, attempted to fix the 6th blocker and the recording path. Both are committed and
keep the 250-test host suite green, but **could not be re-validated on GPU**: the stopped
g6.4xlarge could not be restarted —

```
StartInstances ... InsufficientInstanceCapacity: Insufficient capacity. (us-west-2b)
```

Our VPC has subnets only in us-west-2b, and a stopped instance can only restart in its original
AZ, so neither a restart nor a fresh in-VPC launch works until g6 capacity returns there. This is
the **on-demand-stop risk**: stopping a GPU instance does not reserve its capacity; you may not
get it back immediately. (Capacity Block, by contrast, guarantees the window — but we deliberately
used on-demand for the cheap, stoppable g6.)

### Blocker #6 fix — typing_extensions 4.11.0 (reasoned, not yet GPU-verified)
- The crash (`Type parameter ~_T1 without a default follows ...`) comes from
  `typing_extensions._collect_parameters`. typing_extensions **>= 4.12** *monkeypatches stdlib
  `typing`* to enforce PEP-696 default ordering, which isaaclab_tasks' generics violate. The
  traceback showing stdlib `typing.py` calling into `typing_extensions` is the tell.
- Python 3.11's own `_collect_parameters` does NOT enforce this. Pinning to **4.11.0** (the last
  release before the backported patch) should let the stdlib behavior stand. **4.12.2 was tested
  live and still failed** (it patches too); 4.11.0 is the reasoned fix, baked into the Dockerfile,
  pending a GPU re-run to confirm.
- *Caveat:* if 4.11.0 still trips it, the fallback is to import the task registry the way
  `stage_a_check.py` does (plain `import isaaclab_tasks`, which worked) and avoid the
  `parse_env_cfg` autoload, or move to an Isaac Lab GA image.

### Recording fix — AppLauncher(enable_cameras=True) (the real prerequisite)
- Isaac Sim renders **no camera RGB in pure-headless mode** unless the SimulationApp is launched
  with `enable_cameras=True`. The trainer launched `AppLauncher(headless=True)` only, so any
  demo-video (Req 10) or training-capture (Req 20) render would have produced no frames even once
  the loop ran.
- Fix: `_IsaacLabTrainer` now launches `AppLauncher(headless=True, enable_cameras=<env>)`
  (default on; `HUMANOID_ENABLE_CAMERAS=0` to disable for a no-camera perf run). The evaluator's
  recorder runs in the same process and inherits the camera-enabled app.
- *Not yet exercised:* the recording loop itself (`IsaacLabDemoVideoRecorder.record`) is deeper
  `# pragma: no cover` code that has never run; expect another short chain of fixes there
  (camera prim spawn, `data.output["rgb"]` read, imageio mux) once the loop reaches evaluation.
  No recording has been produced yet because the run still cannot get past env construction.

### Honest status
- #6 and recording: **fixes written + committed + host-suite-green, but UNVERIFIED on GPU** due
  to the capacity outage. Do not claim Stage C passes or that a recording exists — neither is true
  yet.
- To validate: when g6 capacity returns in us-west-2b (retry `start-instances`, or launch fresh +
  `restore_image_from_s3.sh`), rebuild the image (picks up the typing_extensions 4.11.0 pin) and
  re-run `scripts/run_stage_c.sh`. Next likely blockers: the remaining `parse_env_cfg`/RSL-RL
  `OnPolicyRunner` cfg-schema items, then the recorder render loop.


---

## RESOLUTION — Stage C hour-0 smoke PASSES end-to-end (g6/L4)

**Date:** 2026-06-12. After the capacity outage cleared, the g6 restarted and the full bug
chain was driven to completion. The hour-0 smoke now prints:

```
[run] iterations attempted: 1
[run]   iteration 0: status=completed checkpoint=runs/checkpoints/model_final.pt
[run] best policy: iteration 0 score=0.0 checkpoint=runs/checkpoints/model_final.pt
=== HOUR-0 SMOKE: PASS ===
```

Single `AppLauncher` in the log, zero errors. The pipeline runs Isaac Sim init → goal-conditioned
env build → LLM reward generation (cross-host vLLM on the B200) → reward registration → 5 epochs
PPO → checkpoint → evaluation → PASS.

### Blockers #5 and #6 — confirmed fixed on GPU
- **#5 typing_extensions:** the **4.11.0** pin (baked into the Dockerfile) resolved the
  `Type parameter ~_T1 without a default follows ...` crash. 4.12+ monkeypatches stdlib `typing`
  to enforce PEP-696 ordering, which `isaaclab_tasks`' generics violate; 4.11.0 is the last
  release before that backport. Confirmed: `parse_env_cfg`/`gym.make` now load the registry.
- **#6 enable_cameras:** `AppLauncher(headless=True, enable_cameras=True)` confirmed; cameras +
  replicator initialize. (Recording itself is now disabled in smoke — see new blocker #9.)

### The bug chain continued past env construction — three MORE blockers, each fixed live

Once env construction worked, training started and revealed deeper issues. Logged in order:

7. **obs/actions were `None` in the generated reward (FIXED).** Isaac Lab's `RewardManager`
   calls reward terms as `fn(env)` with no obs/actions, and a manager-based env does **not**
   expose `env.obs`/`env.actions`. The wrapper fell back to those absent attrs → the generated
   reward called `.device` on `None` → `AttributeError: 'NoneType' object has no attribute
   'device'`. **Fix:** `_resolve_live_obs` (reads `env.obs_buf` policy group →
   `observation_manager.compute()`) and `_resolve_live_actions` (reads
   `env.action_manager.action`), with legacy/fake fallbacks. *Lesson:* a manager-based env's
   live observation lives in `obs_buf`/`observation_manager`, never `.obs`.

8. **`IndexError: index 14 out of bounds for dimension 1 with size 14` (FIXED — our own bug).**
   This came from **inside** Isaac Lab's `RewardManager.compute()` at
   `self._step_reward[:, term_idx] = value/dt`, NOT from the generated reward. The manager
   pre-allocates `_step_reward` as `(num_envs, num_terms)` at init; when blocker #4's fix
   appended a new term to `_term_names`/`_term_cfgs` (and seeded `_episode_sums`), it left
   `_step_reward` at the old width, so the new term's column index overflowed on the first step.
   **Fix:** widen `_step_reward` (and `_episode_sum_reward`) to the new term count when appending.
   *Lesson:* appending a term to a live RewardManager means matching **every** per-term buffer it
   pre-sized, not just the obvious `_episode_sums`.

9. **Two `SimulationApp`s in one process → indefinite hang (FIXED — the big one).** After
   training completed cleanly, the Evaluator's rollout called `gym.make` to build a **second**
   Isaac Lab env, which relaunched a second `SimulationApp`. **Isaac Sim allows only one
   SimulationApp per process** — the second stalled forever on the RTX rendering-kit reinit (GPU
   0%, log frozen). Disabling demo recording was not enough because the **metrics rollout** also
   built its own env. **Fix (two parts):**
   - `record_demo_video` toggle on `EvalConfig` (auto-off in `--smoke`) so the additive video
     recorder is skipped for the gate.
   - **Env reuse:** the trainer exposes its live env + inference policy via
     `_IsaacLabTrainer.live_handles()`; `PPO_Runner` surfaces them on `TrainResult.live_env/
     live_policy`; the Orchestrator threads them into `Evaluator.evaluate(...)` (with a
     `TypeError` fallback for fakes); `IsaacLabPolicyRollout` reuses them instead of building a
     second env (and does **not** close a trainer-owned env). *Lesson:* in a single-process
     train→eval loop, eval must **reuse** the trainer's live env; you cannot build a second one.

10. **`RuntimeError: Inplace update to inference tensor outside InferenceMode` (FIXED).** With
    env reuse working, the rollout's initial `env.reset()` wrote to env state buffers (e.g.
    `root_link_pose_w`) that had been tagged as torch **inference tensors** during RSL-RL rollout
    collection. In-place writes to those are only allowed **inside** `torch.inference_mode()`, but
    `reset()` ran just outside the `with` block. **Fix:** move `env.reset()` inside the existing
    `with torch.inference_mode():` block. *Lesson:* once a reused env's buffers are inference
    tensors, **all** subsequent mutation (reset included) must stay inside inference mode.

### Orchestrator hardening done alongside the chain
- **Runtime reward failures now re-prompt instead of crashing.** A reward that *validates* but
  *raises at runtime during training* (`ExecutionError`/`TimeoutError`) previously killed the run
  (only `DivergenceError` was caught). Now INJECT+TRAIN is folded into the bounded re-prompt loop:
  the runtime error is fed back to the LLM (with a targeted hint — slice `root_pos_w[:, :2]` for
  ground-plane xy, return `(num_envs,)` tensors) up to the re-prompt bound, then recorded as a new
  `SKIPPED_RUNTIME` iteration. This is the Eureka "a bad generation must not stall the loop"
  principle extended from validation-time to runtime.

### Updated watch-list status
- [x] `isaaclab_tasks` / H1 id registered — yes (direct import + via registry after typing pin).
- [x] `parse_env_cfg` call shape — `device=`/`num_envs=` form works.
- [x] RSL-RL vec-env wrapper — `isaaclab_rl.rsl_rl.RslRlVecEnvWrapper`.
- [x] `OnPolicyRunner` train-cfg schema — the built cfg trains 5 epochs without schema error.
- [x] `attach_goal_conditioning` vs. real H1 cfg — obs in_features=73 (69 proprio + 4 goal),
      RewardManager reports the LLM term active; goal wiring lands.
- [x] AppLauncher first-run, `--network host` loopback, single-SimulationApp eval — all confirmed.

### Net status
- **Stage C hour-0 smoke: PASS.** Full generate→train→evaluate→checkpoint loop verified on the
  g6/L4 with cross-host vLLM on the B200.
- Demo-video recording (Req 10) is **deferred for the gate** (disabled in smoke). The recorder's
  own render loop (`IsaacLabDemoVideoRecorder.record`) is still unexercised `# pragma: no cover`;
  re-enabling it for the full run will need the env-reuse path extended to the recorder + the
  camera RGB read/mux validated. Tracked as the next follow-up.
- All fixes keep the controller-host test suite green (267 tests) and preserve the default 8-GPU
  config behavior.

### Lessons (additions)
- **The adapter bug chain is long but finite — and most of the back half was OUR integration
  logic, not Isaac Lab.** Blockers #7–#10 were about how our trainer/evaluator/reward-binding
  drive the live manager env (obs source, buffer sizing, env lifecycle, inference-tensor rules) —
  exactly the seams that `# pragma: no cover` hid. Budget for "fix, redeploy, hit the next one"
  as the steady-state rhythm of GPU bring-up.
- **One process = one SimulationApp.** The single most expensive hang. Design train→eval to share
  the live env from the start; do not let eval call `gym.make` again.
- **Inference tensors are sticky.** Once RSL-RL collects rollouts under inference mode, the env's
  buffers stay inference tensors; every later touch (reset/step/randomization) must stay inside
  `inference_mode` or torch raises.
- **Make manager-internal failures legible.** The `IndexError` from deep inside
  `RewardManager.compute()` looked like a generated-reward bug but was our term-registration not
  sizing every per-term buffer. When you mutate a framework's internal lists, mirror **all** of
  its derived state.


---

## Demo-video recording — VALIDATED on GPU (the render loop works)

**Date:** 2026-06-12. Restarted the g6 (after one `InsufficientInstanceCapacity` retry — the
single-AZ on-demand-stop risk again) and validated the `IsaacLabDemoVideoRecorder` render loop
end-to-end with a standalone harness (`scripts/validate_recorder.py` + `run_validate_recorder.sh`).
It now renders real chase + side `.mp4` files from the smoke run's trained checkpoint:

```
[validate] recorder returned 2 path(s): ['.../validate_best_chase.mp4', '.../validate_best_side.mp4']
  validate_best_chase.mp4 -> 120335 bytes   (ISO Media, MP4 Base Media v1)
  validate_best_side.mp4  ->  41069 bytes   (ISO Media, MP4 Base Media v1)
VALIDATE_RECORDER: PASS
```

### Why a standalone harness
The recorder's hang in the full loop was the single-SimulationApp problem (blocker #9). To shake
out the **render loop itself** without that confound, the harness runs the recorder in its **own
process** — one SimulationApp, no train/eval coexistence — so each render-loop bug surfaces
cleanly. (Wiring the recorder into the live loop via env-reuse is a separate follow-up; see below.)

### Three render-loop blockers, each fixed
11. **No rsl_rl train cfg on the checkpoint (FIXED).** `_load_policy` built an `OnPolicyRunner` to
    reload the policy but `_infer_train_cfg` raised `Could not determine the rsl_rl train cfg` —
    the checkpoint is a plain rsl_rl `.pt` (weights, no cfg), not TorchScript. **Fix:**
    `_infer_train_cfg` falls back to a default cfg (`_default_rsl_rl_eval_cfg`) that **matches the
    trainer's `_build_train_cfg`** ActorCritic [256,256,256]/ELU + PPO block, so the weights load
    into the right architecture for inference. *Lesson:* an rsl_rl `.pt` carries weights only; the
    network architecture must be supplied to reload it — keep eval's default cfg in lockstep with
    training's.
12. **`'OrderEnforcing' object has no attribute 'get_observations'` (FIXED).**
    `OnPolicyRunner.__init__` calls `env.get_observations()`, which exists only on the **RSL-RL
    vec-env wrapper**, not the raw gym env. `_build_camera_env` returned the unwrapped env (unlike
    the rollout's `_build_env`, which wraps). **Fix:** wrap with `RslRlVecEnvWrapper` before
    returning; camera handles are still resolved from the **unwrapped** scene first, and the
    step/read helpers go through `.unwrapped`, so wrapping is safe. *Lesson:* anything that builds
    an OnPolicyRunner must hand it the rsl_rl-wrapped env, not the bare gym env.
13. **CUDA device-side assert during camera sensor reset (FIXED).** With 4 envs, the rsl_rl
    wrapper's init `reset()` → camera `sensor.reset(env_ids)` → `_timestamp_last_update[env_ids] =
    0.0` triggered a cascade of physx `GpuRigidBodyView` / `DirectGpuHelper` device-side asserts
    (`cubricDeviceBufferMemcpyHtoD ... Assertion (cuerr) == (CUDA_SUCCESS) failed`). **Fix:** the
    recorder renders a **single env** (it only ever captures `env_index` 0). Multi-env tiled
    camera rendering is the trigger on a single-GPU L4; one env is both correct and sufficient.
    *Lesson:* attach RTX cameras to a one-env scene for demo capture — multi-env + cameras is a
    known device-assert minefield on a single GPU.

### Status
- **Demo-video recording: WORKS.** Valid chase + side mp4s rendered from a trained policy and
  persisted to `s3://.../runs/demo_videos/`. The recorder render loop (camera spawn →
  `data.output["rgb"]` read → imageio mux) is now proven, not just unit-tested with fakes.
- **Still a follow-up:** re-enabling recording **inside the full loop** (`run_loop` without
  `--smoke`). The recorder currently builds its own one-env camera scene via `gym.make`, which in
  the live loop would be a second SimulationApp (blocker #9 all over again). The clean fix is to
  attach the demo cameras to the trainer's env at build time (opt-in, default-off so the 4096-env
  training run stays camera-free per Req 10.4) and have the recorder reuse that env. The standalone
  validation de-risks that work — the render loop is known-good; only the env-sharing plumbing
  remains.

### Lessons (additions)
- **Validate hard-to-reach code paths in isolation.** A one-process harness against a saved
  checkpoint turned an un-runnable `# pragma: no cover` render loop into a 3-fix, ~15-min shakeout —
  far cheaper than chasing it inside the full loop where the SimulationApp-reuse confound masks it.
- **Single GPU + RTX cameras = one env.** The multi-env camera device assert is the kind of thing
  only the real hardware tells you; budget for it whenever cameras meet a single-GPU box.

---

## Camera framing fix — the robot was at the frame edge, not centered (VALIDATED on GPU)

**Date:** 2026-06-12. The recorder produced valid mp4s, but inspecting the S3 demo video showed
**the humanoid was not actually in frame** — its pixels were jammed against the far-right edge
(centroid x ≈ 1843/1920 ≈ 96% of width). The render loop worked; the **aiming math was wrong**.

### TL;DR
`CameraSpec._look_at_quat_wxyz` built the look-at quaternion for the **OpenGL/ROS optical
convention** (camera looks down its local **−Z**, up = +Y), but `CameraCfg.OffsetCfg` was
instantiated with `convention="world"`, which defines **forward = +X, up = +Z**. The mismatched
basis rotated the view ~90° off, sliding the robot to the edge. Aligning the math to the world
convention re-centers it.

### Root cause
Isaac Lab's `CameraCfg.OffsetCfg(convention=...)` supports `"world"`, `"ros"`, and `"opengl"`,
and each defines the camera's local forward/up axes differently. We pass `convention="world"`
(confirmed by reading the installed `isaaclab/sensors/camera_cfg.py` source), where:
- local **+X** = forward (view direction),
- local **+Z** = up,
- local **+Y** = completes the right-handed frame.

The old quaternion solved for `−Z = forward, +Y = up` (the optical/ROS convention). Feeding that
quaternion into a world-convention OffsetCfg points the camera in the wrong direction.

### Fix
Rewrote `_look_at_quat_wxyz` to build the rotation for the **world** convention:
- local +X ← normalized `(look_at − position)` (forward),
- local +Z ← world-up orthogonalized against forward (with a +X fallback when forward is
  near-vertical, so a top-down camera doesn't degenerate),
- local +Y ← `Z × X` (right-handed),
- convert that basis matrix → quaternion.

Re-aimed both cameras at the **A→Goal path midpoint at torso height** so the robot stays framed for
the whole run rather than aiming at the empty goal:
- chase `pos=(-2.5, 0, 1.8)  look_at=(2.5, 0, 1.0)`
- side  `pos=(2.5, -4.0, 1.2) look_at=(2.5, 0, 1.0)`

Updated `tests/test_camera_cfg.py` to assert the **world** convention (rotate local +X by the
quaternion → must equal the desired view direction; local +Z up has positive world-z).

### Verification (off-GPU frame analysis, no ffmpeg needed)
Re-rendered from the trained checkpoint on the L4, pulled the mp4s locally, and localized the
robot three ways. The decisive one is **inter-frame motion differencing** — the robot is the only
moving object, so ground texture/sky gradients don't confound it:

| video | pre-fix centroid | post-fix motion centroid | verdict |
|---|---|---|---|
| chase | x ≈ 96% (right edge) | x ≈ 63%, y ≈ 66%, tight cluster | FRAMED |
| side  | n/a | x ≈ 40%, spread along path | FRAMED |

(The naive "darkest-2%-of-pixels" centroid is unreliable here — it latches onto ground shadow at
the bottom of the frame. Motion-diff and color-saturation blobs both put the robot near center.)

### Lessons
- **`convention=` is load-bearing.** A look-at quaternion is only correct *relative to a stated
  camera-axis convention*. Always confirm which convention the consuming API expects (read the
  installed source — Isaac's `world` is +X-forward/+Z-up, not the optical −Z-forward you might
  assume) and build the basis to match.
- **"The recorder works" ≠ "the video is usable."** A render loop that muxes frames can still aim
  at empty space. Add a cheap automated framing check (motion-diff centroid) so "robot in frame"
  is verified, not eyeballed.
- **Aim at the path, not the destination.** Pointing at the goal point leaves the robot a tiny
  figure walking *toward* the lens edge; aiming at the path midpoint at torso height keeps it
  framed throughout.
