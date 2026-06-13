# Lessons: every problem hit bringing up the MuJoCo track on the B200 (and how it was fixed)

**Date:** 2026-06-12 (MuJoCo track, full bring-up Phases 0–6 on `i-EXAMPLE0000000001`)
**Purpose:** a single, blog-ready log of every problem encountered standing up the
Eureka-style LLM→Reward→RL loop on **MuJoCo MJX + Brax + JAX** on a `p6-b200.48xlarge`,
paired with the diagnosis and the fix. Companion to `lesson-mjx-b200-bringup.md` (the
narrative bring-up) and `lesson-self-host-vllm-vs-bedrock.md` (cost).

> Context: this track exists because **Isaac Sim cannot run on the B200** (data-center
> Blackwell has no RT cores). See `../../docs/lesson-isaac-lab-bringup.md` learning 4. MuJoCo
> MJX runs physics as JAX/XLA on CUDA compute, which is exactly the path that works on this
> hardware. The whole point of the gates below was to de-risk that claim cheaply before
> writing a full trainer.

## How to read this

Each entry: **Symptom → Root cause → Fix → Lesson**. Grouped by the phase it bit in. The
recurring meta-lesson: *test the riskiest unknown on the real hardware with the cheapest
possible probe, before building on top of it.* Every problem below was caught by a smoke gate
in minutes, not by a failed 24-hour run.

---

## 0. Environment / tooling

### 0.1 SSM `InvalidDocument` on `AWS-RunShellCommand`
- **Symptom:** `aws ssm send-command --document-name "AWS-RunShellCommand"` →
  `An error occurred (InvalidDocument)`.
- **Root cause:** the correct managed document for Linux shell is **`AWS-RunShellScript`**,
  not `AWS-RunShellCommand` (that's a different/older name and not present in this account).
- **Fix:** use `--document-name "AWS-RunShellScript"`.
- **Lesson:** when scripting SSM, confirm the document name with
  `aws ssm list-documents --filters Key=Name,Values=...`; the shell document is
  `AWS-RunShellScript`.

### 0.2 SSM `--parameters` quoting failures
- **Symptom:** `Parameter validation failed: Invalid type for parameter Parameters.commands[0]`
  / broken commands when passing a complex shell string inline.
- **Root cause:** shell + JSON + nested quotes don't survive inline `--parameters
  'commands=[...]'`.
- **Fix:** build the parameters JSON with a tiny Python helper and pass
  `--parameters file://params.json`. Captured in `scripts/ssm_run.sh` (and `ssm_push.sh` for
  pushing files via base64). One robust helper beats fighting quoting per-call.
- **Lesson:** for non-trivial remote commands over SSM, serialize the payload to a JSON file;
  never hand-quote.

### 0.3 `python3 -m venv` fails on the DLAMI
- **Symptom:** `venv create failed ... You may need to use sudo ... install the python3-venv
  package`.
- **Root cause:** the base image ships Python 3.10 but **not** `python3.10-venv` (no
  `ensurepip`).
- **Fix:** `apt-get install -y python3.10-venv` (the host has NAT egress for apt), then
  `python3 -m venv /data/mjxvenv`. Kept the MJX stack in an isolated venv so it never disturbs
  the system Python that serves vLLM.
- **Lesson:** on a fresh DLAMI, install `pythonX.Y-venv` before creating a venv; isolate the
  new GPU stack from the vLLM Python.

---

## 1. The MuJoCo / MJX / Playground version matrix (the biggest single time sink)

This is the one that would have eaten hours mid-sprint if not caught by a 30-second probe.

### 1.1 `mjx.make_data() got an unexpected keyword argument 'nconmax'`
- **Symptom:** loading `H1JoystickGaitTracking` crashed inside `mjx_env.make_data(... nconmax=)`.
- **Root cause:** PyPI **`playground==0.1.0`** (the only version installable on Python 3.10)
  calls `mjx.make_data(nconmax=...)`, an argument **removed in `mujoco-mjx` 3.9** (what
  `pip install mujoco-mjx` pulls by default).
- **Fix attempt A (too old):** pin `mujoco==3.2.7` / `mujoco-mjx==3.2.7` →
  `XML Error: Schema violation: unrecognized element 'contact'` (the menagerie H1 model XML is
  newer than 3.2.7 understands).
- **Fix attempt B (git main):** `pip install git+.../mujoco_playground` → `requires Python
  >=3.11` (host has 3.10).
- **Fix that worked:** pin the trio to the **3.4.0 era**: `mujoco==3.4.0`,
  `mujoco-mjx==3.4.0`, `playground==0.1.0`. 3.4 still has `nconmax` AND understands the H1
  `<contact>` schema.
- **Lesson:** the Playground PyPI release lags the MuJoCo release line. On Python 3.10, pin
  the whole trio to **3.4.x**; never take the default `pip install mujoco` (pulls 3.9, breaks
  on `nconmax`). For git-main Playground + mujoco 3.9, install Python 3.11 first. This is now
  pinned in `scripts/b200_setup_mjx.sh` and `pyproject.toml`'s `gpu` extra.

### 1.2 `Failed to import warp / mujoco_warp: No module named 'warp'`
- **Symptom:** noisy warnings on every MJX run.
- **Root cause:** `warp` is an *optional* MJX collision backend; absent by default.
- **Fix:** none needed — the default backend works. Filter the warning from logs.
- **Lesson:** harmless; don't chase it.

---

## 2. Hardware-compatibility gates (the reason this track exists)

### 2.1 Does MJX physics run on the B200? (the gate Isaac failed)
- **Risk:** B200 has no RT cores; would JAX/MJX even see/use it?
- **Probe:** `scripts/smoke_test_mjx.py` — a 4096-env `vmap`'d + `jit`'d `mjx.step`.
- **Result:** **PASS.** JAX reports all 8 B200s as `CudaDevice`; a free body falls
  1.00→0.95 m over 20 steps. MJX physics is pure CUDA compute — no RT cores, no Vulkan.
- **Lesson:** the thing that killed Isaac Sim (RTX renderer needs RT cores) does not apply to
  MJX physics. This single probe justified the whole track in ~30 s of GPU time.

### 2.2 Does headless rendering work without RT cores? (Req 10 risk)
- **Risk:** demo-video rendering is the one MJX surface that touches GL.
- **Probe:** `scripts/smoke_test_render.py` with `MUJOCO_GL=egl`.
- **Result:** **PASS** with EGL (`libEGL_nvidia` present on the DLAMI); a 240×320×3 RGB frame
  rendered headless. **OSMesa is NOT installed** (`MUJOCO_GL=osmesa` → PyOpenGL
  `'NoneType' object has no attribute 'glGetError'`).
- **Fix:** always render with `MUJOCO_GL=egl`; don't rely on the OSMesa fallback here.
- **Lesson:** headless GL on a data-center GPU works via **EGL**, not OSMesa. Set
  `MUJOCO_GL=egl` everywhere rendering happens.

---

## 3. The H1 env surface (write code against reality, not assumptions)

### 3.1 obs is 113-dim flat, not the ~69-dim Isaac layout
- **Symptom:** the initial prompt/obs-space text described the Isaac ~69-dim vector.
- **Root cause:** `H1JoystickGaitTracking` is a different env with its own observation.
- **Fix:** probed the live env (`scripts/probe_h1_env.py`): obs is a flat `(113,)` float32;
  action `(19,)`; dt 0.02 s; `state.data` exposes `qpos(26)`, `qvel(25)`, `xpos(21,3)`,
  `actuator_force(19)`, `cfrc_ext(21,6)`. Updated `DEFAULT_OBS_SPACE_DESCRIPTION` + prompts.
  Our goal wrapper appends 4 Goal dims → **117**.
- **Lesson:** always probe the real env for obs/action/`data` shapes before writing the
  wrapper or the prompt; the qpos/qvel index comments were right but the obs dim was not.

---

## 4. The JAX reward contract (highest-risk piece)

### 4.1 Non-jittable generated reward must be caught BEFORE training
- **Risk:** an LLM will eventually emit reward code with value-dependent Python control flow
  (`if dist < r:`), which silently breaks under `jax.jit`/`vmap`.
- **Fix:** `WrappedReward.maybe_jit_check()` runs `jax.eval_shape` on a synthetic `mjx.Data`
  at `wrap()` time. Verified live: a Python `if` on a traced value raises
  `TracerBoolConversionError`, surfaced as a typed `ExecutionError` with an actionable message
  ("use jnp.where/lax.select ... no .item()") so the Orchestrator re-prompts (Req 5.5, 12.1).
- **Lesson:** for JAX-generated code, a jit trace-check at wrap time is the equivalent of the
  `ast.parse` gate — it turns a mid-training crash into a cheap, recoverable re-prompt.

### 4.2 `import jax.numpy as jnp` inside the sandbox → `ImportError: __import__ not found`
- **Symptom:** the (correct, jit-safe) generated reward failed because it began with
  `import jax.numpy as jnp`, and the sandbox's restricted builtins omit `__import__`.
- **Root cause:** the model naturally writes the import even though `jnp` is pre-provided;
  the allowlist-only sandbox blocked all imports.
- **Fix:** provide a **restricted `__import__`** that resolves only `math` / `jax` / `jax.*`
  (mirroring real `__import__`/`importlib` semantics, incl. the `fromlist` behavior for
  `import a.b as c`), and pre-bind `jnp`/`jax` in the namespace. Any other import still raises
  `ImportError` → captured as `ExecutionError`.
- **Sub-bug:** first cut returned the submodule for `import jax.numpy as jnp`, which Python
  then tried to attribute-access (`cannot import name 'numpy' from 'jax.numpy'`). Fixed by
  returning the **top-level package** when `fromlist` is empty and the named module only when a
  `fromlist` is present.
- **Lesson:** a sandbox that allowlists modules must still emulate `__import__` faithfully for
  the allowed ones, including `import a.b as c` semantics — or instruct the model to never
  import. We did both (restricted importer + a prompt note that `jnp` is pre-provided).

---

## 5. Brax PPO integration (two bugs the live train gate caught)

### 5.1 `scan body carry input/output must have the same pytree structure` (metrics)
- **Symptom:** Brax training crashed in `jax.lax.scan` with a symmetric-difference of metric
  keys: `{goal/effort, goal/arrival, goal/progress, goal/alive, goal/upright}`.
- **Root cause:** Brax's training wrapper scans the env step and requires `state.metrics` to
  have **identical pytree structure** at `reset` and at `step`. Our goal reward adds `goal/*`
  metric keys in `step`; `reset` didn't have them.
- **Fix:** seed the `goal/*` keys in `reset` by computing the goal reward once on the reset
  data (same keys + dtypes).
- **Lesson:** any MJX env that adds metrics in `step` must add the **same keys in `reset`**, or
  Brax's scan rejects it. This is a general Brax+Playground gotcha for custom rewards.

### 5.2 `Can't pickle local object 'make_inference_fn.<locals>.make_policy'`
- **Symptom:** `save_checkpoint` crashed pickling the training output.
- **Root cause:** Brax PPO returns `make_inference_fn`, a **closure** that `pickle` can't
  serialize.
- **Fix:** persist the policy **params pytree** (numpy leaves) instead of the function;
  rebuild the inference fn from params + the network factory at eval time.
- **Lesson:** checkpoint the params, never the closure. (Follow-up: the eval-time
  reconstruction of `make_inference_fn` from saved params is still a TODO — see §8.)

### 5.3 `assert num_envs % device_count == 0`
- **Symptom:** the full loop crashed in Brax PPO when run with `CUDA_VISIBLE_DEVICES=1..7`
  (7 GPUs) and `num_envs=256` (256 % 7 ≠ 0).
- **Root cause:** Brax shards envs across visible devices and requires divisibility.
- **Fix (smoke):** run on a single training GPU (`CUDA_VISIBLE_DEVICES=1`). For the scaled
  multi-GPU run, choose `num_envs` divisible by the training-GPU count (e.g. 7×N) or pin to a
  device count that divides it.
- **Lesson:** `num_envs` must be divisible by the number of visible JAX devices. Decide the
  GPU count and env count together. (MJX often saturates one GPU anyway, so single-GPU
  training is a legitimate default.)

### 5.4 `assert batch_size * num_minibatches % num_envs == 0`
- **Symptom:** at `num_envs=2048` Brax PPO asserted on `batch_size * num_minibatches %
  num_envs == 0` (the defaults that happened to fit 256 envs did not fit 2048).
- **Root cause:** Brax's PPO ties `batch_size`/`num_minibatches` to `num_envs`; the defaults
  are not valid for arbitrary env counts.
- **Fix:** the trainer now derives `num_minibatches=8` and `batch_size = (num_envs //
  num_minibatches) * num_minibatches`, falling back to `batch_size=num_envs,
  num_minibatches=1` if that doesn't divide — so the assertion holds for any `num_envs`.
- **Lesson:** set `batch_size`/`num_minibatches` explicitly as a function of `num_envs`; never
  rely on Brax PPO defaults across different env counts.

### 5.5 `InvalidInputException: ... is not a valid JAX type` (generated-reward components)
- **Symptom:** with a *different* live-generated reward (a later iteration of the multi-iteration
  run), Brax training crashed: `Argument 'Traced<float32[]>...' ... is not a valid JAX type`.
- **Root cause:** the generated reward returned a component as a **Python float** (e.g.
  `"alive": 0.1`) rather than a `jnp` array. The goal env put it straight into `state.metrics`,
  and Brax's `lax.scan` cannot carry a non-JAX scalar as a metric. The earlier smoke happened
  to use a reward whose components were all `jnp` expressions, so it never tripped this.
- **Fix:** the goal env now coerces **every** reward component (and the reward itself) to a
  `jnp.float32` scalar via `jnp.asarray(..., dtype=jnp.float32)` before placing it in
  `metrics`, in BOTH `reset` (the seeded keys) and `step`.
- **Lesson:** untrusted generated reward code will return scalars in inconsistent forms
  (Python float, int, 0-d array). The env boundary that feeds Brax metrics must coerce them to
  a uniform `jnp` dtype — don't assume the model returns clean arrays. This is the runtime
  analog of the jit-check: the jit-check catches control-flow, this catches dtype/leaf-type.

---

## 6. Evaluator + demo recorder

### 6.1 `MjxDemoVideoRecorder.record() got an unexpected keyword argument 'rollout'`
- **Symptom:** the loop completed training + metrics but crashed at demo-video recording.
- **Root cause:** the carried-over `DemoVideoProducer` calls
  `recorder.record(rollout=, cameras=, label=, output_dir=)` and expects a list of file paths
  back; the first MJX recorder draft used a different signature.
- **Fix:** rewrote `MjxDemoVideoRecorder.record` to that exact contract, rendering each demo
  camera (chase + side) to its own `.mp4` via the EGL `mujoco.Renderer`.
- **Lesson:** when porting a track, the *injected* adapter must match the carried-over
  caller's signature exactly — the producer is framework-agnostic and unchanged.

### 6.2 `imageio` macro-block + `os.fork()` warnings — RESOLVED
- **Symptom:** `input image is not divisible by macro_block_size=16, resizing 1080→1088`; and
  `os.fork() was called ... JAX is multithreaded, so this will likely lead to a deadlock`.
- **Root cause:** ffmpeg pads odd resolutions; `imageio-ffmpeg` spawns the encoder via
  `os.fork()` which interacts badly with JAX's threadpool — a real deadlock risk over a long
  unattended run.
- **Fix (DONE):** the recorder now collects frames during the rollout, saves them to a
  temporary `.npy`, and encodes via a **separate subprocess** (`python -m src.eval.encode_video`,
  fork+exec, no JAX imported) using `subprocess.run`. The encoder pads frames to multiples of
  16 and passes `macro_block_size=1`, so there is no resize warning either. Verified live:
  both `best_chase.mp4`/`best_side.mp4` encode and **zero `os.fork()` warnings** in a fresh run.
- **Lesson:** never let `imageio-ffmpeg` fork from inside a live JAX process. Render frames in
  JAX, then hand them to a clean fork+exec subprocess for encoding; pad to ×16 to avoid the
  resize warning entirely.

### 6.3 Demo video IS recorded per-iteration (self-wired recorder) — verify in S3, not locally
- **Question that prompted it:** "video keep per iteration?" then "in s3, i see the best and
  worst mp4 in the iteration already?"
- **What actually happens:** the Evaluator **self-wires** the demo recorder. When no recorder is
  injected, `Evaluator._resolve_recorder` lazily calls `build_mjx_demo_recorder` (a GPU-only
  builder that no-ops off-GPU). So on the B200 **every completed iteration renders best/worst
  demo video and uploads 4 `.mp4`s to S3** (best + worst × chase + side cameras), e.g.
  `runs-mujoco/run-1gpu-v4/iteration-00/best-0-best_chase.mp4`, `…/worst-1-best_side.mp4`.
- **The trap that misled an initial answer:** the videos are rendered into a temp dir, uploaded,
  and **not** left in the run's local checkpoint folder (`/data/mujoco/runs/run-*/`) or the
  per-iteration artifact temp dir (`/tmp/humanoid-artifacts-*/`, which holds only
  `metrics.json` + `reward.py`). Checking those local dirs showed no `.mp4` and produced a
  wrong "no per-iteration video" conclusion. **The artifacts live in S3** — verify there:
  `aws s3 ls s3://<bucket>/runs-mujoco/ --recursive | grep mp4`.
- **Cost/timing note:** per-iteration rendering adds an EGL render + a fork+exec encode (§6.2)
  to every completed iteration. For an A/B *timing* comparison this is fine as long as **both
  runs do it equally** (they do) — the relative wall-clock holds; only the absolute time is
  higher than a video-free run. If you want the fastest possible run, inject `recorder=None`
  (or a flag) to skip it and render end-only from the best checkpoint instead.
- **End-only alternative (kept as a tool):** `scripts/render_best_policy.py` reconstructs the
  trained policy from any finished run's checkpoint (`_load_policy` + `.netcfg.json` sidecar)
  and renders best/worst on demand — useful for re-rendering at higher resolution or for a run
  where per-iteration video was disabled, without re-training.
- **Lesson:** confirm where an artifact is *persisted* before claiming it doesn't exist —
  local staging dirs and the durable store (S3) are not the same place. And know your
  framework's lazy self-wiring: an "optional, injected" dependency can still be auto-built at
  runtime on the host that can supply it.

---

## 7. AWS IAM / persistence

### 7.1 `AccessDenied: s3:PutObject on .../runs-mujoco/...`
- **Symptom:** the hour-0 loop reported PASS and wrote local artifacts, but **nothing landed
  in S3**.
- **Root cause:** the instance role's inline policy scoped `s3:PutObject` to `runs/*` only;
  this track writes under the **`runs-mujoco/*`** prefix (chosen so the two tracks don't
  collide). The fail-soft persistence (Req 11.3) correctly **kept the run alive** and retained
  local copies instead of crashing — which is exactly why the failure was easy to miss.
- **Fix:** extended the `WriteRunArtifacts` statement's `Resource` to include
  `arn:aws:s3:::humanoid-from-scratch-123456789012/runs-mujoco/*` (additive, least-privilege —
  no new actions, no wildcard broadening). Re-ran → all artifacts landed:
  `reward.py`, `iter_01_policy.pkl`, `metrics.json`, `training_metrics.json`,
  `best/worst *.mp4`, the `best_policy/` export, and `loop_checkpoint.json`.
- **Lesson:** when a new track uses a new S3 prefix, the IAM write scope must include it.
  Fail-soft persistence is a double-edged sword: it prevents a crash but can hide a misconfig —
  always verify artifacts actually appear in S3 after the first run, don't trust the PASS line
  alone.

---

## 7B. Eureka feedback-loop robustness (multi-iteration run)
### 7B.1 Trained-policy reconstruction (the readiness blocker)
- **Symptom:** every iteration's eval showed identical all-zero metrics — no learning signal
  to feed back to the model.
- **Root cause:** `mjx_rollout._load_policy` was a zero-action stub; eval never loaded the
  trained weights. For a Eureka loop this is fatal-to-purpose: Qwen refines against metrics, so
  identical garbage metrics = no refinement signal.
- **Fix:** `save_checkpoint` now persists the Brax params via `brax.io.model.save_params` plus a
  `<path>.netcfg.json` sidecar (obs/action size, hidden-layer sizes, normalize flag); the
  trainer passes an explicit `network_factory` so the architecture is reproducible.
  `_load_policy` rebuilds the policy with `ppo_networks.make_ppo_networks` +
  `make_inference_fn(params, deterministic=True)`. Verified: reconstructed policy emits
  non-zero actions (`max|a|=1.0`) and per-iteration metrics now **vary** (upright_time
  0.215→0.305→0.465 s across 3 iters).
- **Lesson:** checkpoint params + the network config needed to rebuild the policy; the closure
  Brax returns is not enough and not picklable (§5.2). Always verify eval loads real weights
  (non-zero actions), not a silent fallback.

### 7B.2 Generated-reward output variability breaks Brax `lax.scan` carry
- **Symptom:** iteration 0/1 trained fine; iteration 2's (different, valid-looking) reward
  crashed Brax at `reset_fn_`/`step` with `InvalidInputException: ... is not a valid JAX type`.
- **Root cause:** untrusted generated rewards return components in inconsistent forms (Python
  float, int, 0-d array, even a per-joint vector). Brax's scan can only carry uniform scalar
  metrics. The jit-check (§4.1) catches control-flow but not leaf-type/shape variability.
- **Fix:** the goal env now reduces EVERY component and the reward to a finite float32 scalar at
  the boundary — `_scalar_f32`: `jnp.asarray(..., float32)` → `mean` if non-scalar →
  `nan_to_num` — in both `reset` and `step`.
- **Lesson:** the env boundary that feeds Brax metrics must normalize whatever the model emits;
  never assume clean uniform arrays from generated code.

### 7B.3 A bad candidate must SKIP, not abort the unattended run
- **Symptom:** before the fix, one odd reward at iteration 2 killed the whole loop.
- **Root cause:** `run_iteration` only caught `DivergenceError` (which has its own
  revert+reduce-LR path); any other train/eval exception propagated and terminated the run.
- **Fix:** wrapped the train+eval block so any non-divergence failure records a
  `skipped_gen_failure` `IterationRecord` and the loop continues (Req 7.6 spirit). Divergence
  keeps its dedicated recovery.
- **Lesson:** the Eureka loop is *designed* around some generations being unusable — a
  per-candidate failure must be contained to that iteration. An unattended multi-hour run can
  never abort on a single bad reward.

> **Is it an LLM quality issue?** No. Across 3 iterations Qwen3-Coder produced sound, varied,
> jit-safe multi-term goal rewards every time. All failures were *harness* gaps in handling the
> natural variability of untrusted generated code (leaf types, output shape) and missing
> skip-on-failure containment. The robot not yet reaching the goal (fall_rate 1.0 at 40 epochs)
> is a *training-budget* issue, not a reward-quality one — the full run uses 1500 epochs.

---

## 7C. Training-budget semantics + pre-restart config audit

### 7C.1 "epochs" was a 1500× landmine — know what your timestep budget actually is
- **Symptom:** the first full run's iteration 0 was still training after >1 hour; at
  `epochs: 1500` a single Eureka iteration would take ~4+ hours and the 12-iteration loop
  ~50 hours (multi-day, not overnight).
- **Root cause:** the trainer computes `num_timesteps = num_envs × episode_length × epochs`.
  With `num_envs=2048`, `episode_length=1000` (the `TrainConfig` default — NOT set in
  `run_config.yaml`), `epochs=1500` → **~3.07 BILLION env steps per iteration**. Published
  MuJoCo Playground / Brax humanoid gaits converge in ~50-200M steps, so this was **15-60×
  overkill per iteration** — wasting hours over-training one reward when the Eureka method wants
  *many refinement iterations* at a sane per-iteration budget.
- **Fix:** `epochs: 75` (≈150M steps at 2048 envs; ≈230M at 3072) — squarely in the
  literature's sweet spot. Also dropped `checkpoint_interval: 500 → 25` (it exceeded the epoch
  count, so mid-iteration checkpoints never fired). Stopped both runs and relaunched.
- **Lesson:** "epochs" is meaningless without knowing the steps-per-epoch mapping. Always
  compute the **total env steps** (`num_envs × episode_length × epochs`) and sanity-check it
  against published convergence budgets for the task BEFORE a long run. And `episode_length`
  living as a code default (not in the config) is a silent multiplier — surface every budget
  knob in the config. For Eureka specifically: prefer **more iterations × a sane per-iteration
  budget** over few iterations × a giant one — the value is in reward *refinement*, not
  over-training a single reward.

### 7C.2 Audit the full effective config before every (re)launch
- **What we did:** before relaunch, dumped the complete `run_config.yaml` and `diff`ed the two
  A/B configs (confirmed they differ ONLY in `num_envs`/`oom_fallback`), then loaded both
  through `load_config` to confirm the loader accepts the new values, and re-checked vLLM was
  up. Caught the stale `checkpoint_interval` in the same pass.
- **Lesson:** a launch is cheap to start and expensive to waste. Spend two minutes dumping the
  *effective* config (not what you think it is), diffing A/B variants, round-tripping through
  the loader, and pinging dependencies (vLLM) — every time. Most of this session's wasted GPU
  hours traced to a config value (`epochs`) nobody had eyeballed at full scale.

---

## 7D. Concurrent A/B runs starved the shared vLLM (all iterations skipped)

- **Symptom:** launching the 1-GPU and 6-GPU runs simultaneously, the 1-GPU run finished in
  ~8 minutes with **all 12 iterations `skipped_gen_failure`** and no training at all; the 6-GPU
  run also stalled with GPUs at 0%.
- **Root cause:** **both Eureka loops share the single vLLM endpoint** (GPU 0, loopback). When
  they fire reward-generation requests concurrently at startup, vLLM serializes them; with
  `llm.max_retries: 3` and a burst of contention, one loop's requests exhaust their retries and
  the orchestrator correctly records `skipped_gen_failure` (Req 7.6) — every iteration. A direct
  single-client generation test right after confirmed vLLM itself was healthy (returned a 2842-
  char reward), so it was **contention, not a vLLM fault**. The skip-don't-abort robustness
  (§7B.3) worked exactly as designed — it just skipped *everything* because generation kept
  losing the race.
- **Fixes (choose per goal):**
  1. **Run the A/B sequentially** (1-GPU run, then 6-GPU run) — cleanest; no shared-endpoint
     contention, and the wall-clock comparison stays valid since each run is timed independently.
  2. **Raise `llm.max_retries`/`retry_backoff_s`** so a loop tolerates a busy vLLM, and/or
     stagger launches by a few minutes so their reward-gen phases don't collide.
  3. **Bedrock backend** (per `lesson-self-host-vllm-vs-bedrock.md`) — a managed endpoint
     removes the single-vLLM contention point entirely; both loops call it independently.
- **Lesson:** a single self-hosted vLLM is a **shared, serializing resource**. Concurrent
  agentic loops that each generate against it will contend, and a strict retry bound turns
  contention into skipped iterations. For parallel A/B runs, either serialize them, give the
  client generous retry/backoff, stagger startups, or move generation to a managed endpoint.
  The operational flip-side of the cost lesson: one vLLM is cheap, but it is *one* server.

> **Postscript (see §7E):** when the relaunch *still* skipped every iteration even with the
> shared-vLLM lock in place, the contention theory was disproven. The real cause was a
> train-time bug (`_n_calls`) that the **ambiguous `skipped_gen_failure` label was hiding**.
> §7D's mitigations are still worth keeping for true concurrent runs, but they were treating a
> symptom that, this time, had a different root cause. Lesson-within-the-lesson: *verify the
> skip reason from data before theorizing about the cause.*

---

## 7E. The "all iterations skipped" mystery: an ambiguous status label hid a train-time crash

- **Symptom:** after adding the shared-vLLM lock (§7D), **both** A/B runs *still* finished with
  all 12 iterations `skipped_gen_failure` — in ~14 min, far too fast for real training. The
  shared lock had not helped, so "vLLM contention" could not be the whole story.
- **Why we were blind:** three different failure paths in `Orchestrator.run_iteration` all
  recorded the **same** `skipped_gen_failure` status — (a) generation `RequestError`,
  (b) divergence after the reduced-LR retry, and (c) the generic train/eval `except Exception`.
  The actual reason text was stored in `IterationRecord.behavior_description`, but **the run
  log never printed it** (`_print_result` only showed status + checkpoint). So a *training*
  crash looked identical to a *generation* failure in the log. We had been debugging the LLM
  for hours when the LLM was never the problem.
- **Root cause:** a fast 1-iteration smoke with reason-logging added revealed
  `AttributeError: '_MjxTrainer' object has no attribute '_n_calls'`. The recently-added
  train-progress log line (`src/train/mjx_trainer.py`) referenced `self._n_calls` from inside
  `_MjxTrainer.run_epoch`'s `_progress` closure, but `_n_calls` is a counter on **`PPORunner`**,
  not on `_MjxTrainer`. So **every** iteration crashed the instant Brax called `progress_fn`,
  was caught by the generic `except Exception`, and was mislabeled `skipped_gen_failure`. The
  user's own hypothesis — *"is it the progress logging we changed?"* — was exactly right.
- **Fixes:**
  1. **The bug:** drop the cross-object reference; the progress line now prints
     `step=<n>/<total> (~pct%)`, which is the meaningful signal and needs no iteration counter.
  2. **The blindness (so this can't recur):** (i) print each iteration's reason live in
     `Orchestrator.run` and in `_print_result`; (ii) add a **distinct** `skipped_train_failure`
     status for the divergence-after-retry and generic train/eval paths, so a run log makes it
     obvious at a glance that *generation succeeded and the failure was downstream in training*.
- **Confirmation:** the next smoke **completed** with a real checkpoint and a learning policy
  (eval_reward climbed −88 → −27 across one iteration's training). Under the full A/B config,
  the 1-GPU run trained cleanly (eval_reward → **−3.8**, near the goal) and the 6-GPU run
  correctly logged two `skipped_train_failure` iterations (an intermittent reward-specific
  tracer leak — see note below) and **kept going** to the next candidate, exactly as the
  skip-don't-abort design (§7B.3) intends.
- **Lesson:** **a status label that collapses distinct failure modes will eventually cost you
  hours.** When an unattended loop "skips everything," the *reason string* is the first thing to
  read — and it must be in the log, not buried in a checkpoint field. Give each failure class its
  own status. And a progress/telemetry line added for observability is still **executed code**:
  it can crash the very run it was meant to observe.

> **Side note — the tracer leak (`InvalidInputException: ... is not a valid JAX type`).** Some
> LLM-generated rewards pass `wrap()`'s `jax.eval_shape` jit-check yet still fail inside the full
> Brax PPO graph, because `eval_shape` traces abstract shapes but does not exercise the same
> vmap+`lax.scan` control flow the trainer does. These are correctly contained as
> `skipped_train_failure` and the loop moves on. Hardening the up-front check to catch them
> (e.g. tracing through a closer analog of the training step) is a possible future improvement,
> not a correctness blocker.

---

## 8. Known-good end state + open follow-ups

**End state (all verified live on the B200):**
- MJX physics (4096-env jit+vmap), EGL headless render, H1 env load — PASS.
- Goal reframing (obs 113→117, goal reward, vmap) — PASS.
- JAX reward contract (validate → sandbox → jit-check; non-jittable rejected) — PASS.
- Brax PPO trains the goal env, checkpoint written — PASS.
- Evaluator rollout → metrics + staged gates → JSON, EGL demo videos — PASS.
- **Hour-0 end-to-end Eureka loop** (live Qwen → reward → train → eval → S3) — **PASS**.
- **3-iteration feedback loop** — **PASS**: all 3 iterations completed, none aborted, trained
  policies reconstructed, per-iteration metrics vary (real feedback signal).
- Off-GPU suite: 193 passed; DI seam (no jax/mujoco/brax at import) intact.

**Open follow-ups (carry into the scaled run):**
- [x] **Trained-policy reconstruction** — DONE (§7B.1); eval loads real weights, metrics vary.
- [x] **Skip-on-failure containment** — DONE (§7B.3); bad candidates skip, loop continues.
- [x] **Generated-reward output coercion** — DONE (§7B.2).
- [ ] **Multi-GPU env count:** the validated runs use a single training GPU
      (`CUDA_VISIBLE_DEVICES=1`); MJX saturates one GPU well. **Multi-GPU data-parallel is now
      validated** (`scripts/smoke_test_multigpu.py` PASS across 4 B200s) — recipe in
      `docs/multi-gpu-run-recipe.md`: set `num_envs = K × device_count` (multiple of 7 for the
      7-GPU run, GPU 0 stays vLLM). Ready for the next run.
- [ ] **Video encoding for long runs:** DONE — frames encoded in a separate fork+exec
      subprocess (`src.eval.encode_video`), padded to ×16; zero os.fork warnings (§6.2).
- [ ] Pin `mujoco==3.4.0` / `mujoco-mjx==3.4.0` in `pyproject.toml` `gpu` extra + Dockerfile
      so §1.1 can't recur.
- [ ] **Full run:** 12 iterations × 1500 epochs (config restored) — the training budget that
      gives the policy a real chance to reach the goal.
- [ ] **Comparison blog (Req 21.5):** pair these lessons with the Isaac/RTX track.

## The one-line meta-lesson for the blog

Every blocker above was a *version, hardware-capability, framework-contract, or IAM* mismatch —
not an algorithm problem. Cheap smoke gates on the real hardware (≈30 s each) caught all of
them before any expensive run, exactly the discipline that caught the Isaac RT-core wall.
Build the gates first; build on them second.
