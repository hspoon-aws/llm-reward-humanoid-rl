# Phase 2 — Running the Eureka loop on the EC2 p6-b200 host

A plan to take the implemented Phase 2 code (tasks 8–15) from "unit-tested on the
controller host" to "running a real LLM→Reward→RL loop on the capacity-block
instance." Written from a review of `src/train/ppo_runner.py`, `src/orchestrator.py`,
`src/envs/goal_env.py`, `src/rewards/reward_executor.py`, `src/eval/*`, `src/capture.py`,
`src/blog.py`, and the `Dockerfile`.

## Where the code stands

**Implemented and unit-tested (controller host, no GPU):**
- Goal data models + goal-conditioned env wiring (`goal_env.py`) — Goal buffer,
  robot-frame Goal_Observation, stock-reward zeroing
- Reward_Executor: validate → sandbox → wrap → live `RewardManager` binding, with the
  `_EnvAccessProxy` guard
- PPO_Runner: device selection (excludes GPU 0), checkpoint scheduling, metrics JSON,
  divergence detection, OOM fallback, capture hook
- Evaluator + metrics + capability gates; Camera_Config; Orchestrator loop + all recovery
  paths; training capture; blog assembly
- 250 tests pass

**The design is DI-clean:** Isaac Lab / RSL-RL are imported **lazily inside
`_IsaacLabTrainer`** (`build_isaaclab_trainer`), so nothing simulation-related is needed to
import the modules. That's exactly what makes the run plan tractable — the only place the
real simulator is touched is one adapter class.

## The real gaps before a run

1. **Production entrypoint — DONE.** `src/run_loop.py` builds the *real* QwenClient +
   RewardExecutor + PPORunner (real Isaac Lab trainer factory) + Evaluator + S3Store from
   `config/run_config.yaml` and calls `run()`. `build_orchestrator(...)` exposes every
   collaborator as an injectable override (the DI seam the unit tests use), and
   `python -m src.run_loop --smoke` runs the hour-0 smoke. (Named `src/run_loop.py`, not the
   originally-suggested `src/run.py`; invoke with `python -m src.run_loop`.)
2. **Isaac Lab / Isaac Sim / RSL-RL not installed on the host.** The instance is the bare
   DLAMI with vLLM only. Isaac Lab must be brought up (container or pip into the Isaac Sim
   interpreter).
3. **Unverified version assumptions in `_IsaacLabTrainer`.** The adapter guards for API drift
   (`parse_env_cfg(device=|use_gpu=)`, the RSL-RL wrapper import path, `OnPolicyRunner` train
   cfg schema) but **none of it has run against a real install.** This is the highest-risk
   item and must be shaken out with a tiny run before the full loop.
4. **GPU-0 contention.** vLLM holds GPU 0; training must use 1–7. The config already sets
   `training_gpus: [1..7]` and the runner re-enforces the exclusion — but it must be verified
   live that Isaac Lab actually lands off GPU 0.
5. **Env id — FIXED + now a config field.** The registered Isaac Lab id is
   `Isaac-Velocity-Flat-H1-v0` (no `Unitree-` segment; the earlier
   `Isaac-Velocity-Flat-Unitree-H1-v0` was wrong). It is now sourced from `training.task` in
   `run_config.yaml` → `Config.env_id` → `TrainConfig.env_id` / `EvalConfig.env_id`, so a
   future registry-name drift is a one-line config edit, not a code change. Still worth a live
   `parse_env_cfg(...)` confirmation in Stage A.3.

## Recommended path: container for Isaac Lab, reuse the host vLLM

The `Dockerfile` already targets `nvcr.io/nvidia/isaac-lab:2.2.0` + RSL-RL + project code.
Use it for the Isaac Lab/training side — it pins Isaac Sim + Isaac Lab + RSL-RL together and
avoids fighting the DLAMI's Python. The instance has Docker and the NVIDIA container runtime
(DLAMI ships both).

**vLLM does NOT need to move into the container.** It is already running on the host (GPU 0),
with weights and compile caches warm — re-loading 57 GB and re-paying the ~13-min Blackwell
JIT inside a fresh container would be pure waste. The key fact: `127.0.0.1` is per **network
namespace**, so the design's loopback assumption is satisfied as long as the loop process
shares the namespace where vLLM listens. Run the loop container with **`--network host`** and
it shares the host's network namespace, so `http://127.0.0.1:8000/v1` (what `run_config.yaml`
already points at) resolves to the host vLLM — no code or config change.

Topology options considered:

| Option | vLLM | Loop/training | Loopback works? | Verdict |
|---|---|---|---|---|
| **A (recommended)** | host (already up, warm) | container `--network host` | yes — shared host netns | reuse warm vLLM, no reload/re-JIT |
| B | inside loop container | same container | yes — same netns | clean but re-loads weights + may re-JIT |
| C | host | container, bridge net | NO — container's own `127.0.0.1` | would need host IP + an inbound path; avoid |

**Why A over B:** A keeps the validated host vLLM and its warm caches; B is only worth it if
you want a single self-contained artifact and are willing to mount `runs/cache/{vllm,flashinfer}`
into the container's cache dirs to skip the re-JIT (lesson docs). Default to A.

**Security note for `--network host`:** the container shares the host network, so the
default-deny SG still governs all inbound (good — nothing new is exposed). vLLM and TensorBoard
must keep binding `127.0.0.1` (they do). No inbound rule is added; operator access stays SSM-only.

## Step-by-step

### Stage A — bring up Isaac Lab (de-risk early, cheap)
1. Build the image on the host: `docker build -t humanoid-from-scratch:latest .`
   (the host has NAT egress to nvcr.io; this is the long pole — pull is tens of GB).
2. Smoke the simulator import **inside the container** (`--network host`, GPUs visible),
   headless: `isaaclab.sh -p -c "import isaaclab, isaaclab_tasks, rsl_rl; print('ok')"`.
3. Confirm the env id is registered and parses:
   `parse_env_cfg("Isaac-Velocity-Flat-H1-v0", num_envs=4)`.
   If the installed registry uses a different name, set `training.task` in
   `config/run_config.yaml` — no code change needed (it flows through `Config.env_id`).
4. From inside the container, confirm it can reach the host vLLM over shared loopback:
   `curl -sS http://127.0.0.1:8000/v1/models` (proves the `--network host` topology before
   the loop depends on it).

### Stage B — production entrypoint (DONE — `src/run_loop.py`)
Implemented as `src/run_loop.py` (the plan originally proposed `src/run.py`). `build_orchestrator`:
1. `load_config("config/run_config.yaml")`
2. builds `QwenClient(QwenClientConfig(endpoint=cfg.llm_endpoint, prompts_dir="prompts"))`
3. builds `RewardExecutor(SandboxConfig(time_limit_s=cfg.sandbox_time_limit_s))`
4. builds `PPORunner(TrainConfig.from_config(cfg, checkpoint_dir=...))` with the **real**
   trainer factory (`build_isaaclab_trainer`) — no fake injected
5. builds `Evaluator(EvalConfig.from_config(cfg), CameraConfig.from_config(cfg))`
6. builds `S3Store(cfg.s3_location, run_id=...)`
7. `Orchestrator(cfg, qwen, executor, runner, evaluator, store).run()`

The CLI provides Stage-C overrides via `--smoke` (1 iteration), `--smoke-epochs`,
`--smoke-num-envs`, and `--eval-episodes`; every collaborator is an injectable kwarg so the
wiring is unit-tested off-GPU (`tests/test_run_loop_smoke.py`).

### Stage C — hour-0 smoke (spec Task 16.1) BEFORE the full loop
Run the entrypoint inside the loop container (`docker run --gpus all --network host ...`) with
a **deliberately tiny** budget to prove the whole pipe end-to-end on the GPU without burning
the block:
- `python -m src.run_loop --smoke --smoke-num-envs 64 --smoke-epochs 5 --eval-episodes 2`
  (`--smoke` already pins `max_iterations=1`).
- Success = one reward generated by the host vLLM (`127.0.0.1:8000` via shared netns) →
  validated/wrapped → registered on a live H1 env → a few PPO epochs run on GPUs 1–7 → a
  checkpoint written → evaluator produces EvalMetrics → artifacts land in S3 `runs/`. The CLI
  prints `=== HOUR-0 SMOKE: PASS ===` and exits 0 when the iteration completed with a
  checkpoint.
- This is the single most important gate; it flushes out every version/wiring problem at low
  cost. Keep the existing `_EnvAccessProxy` + divergence/OOM guards on — they turn a bad first
  reward into a recoverable iteration rather than a crash.

### Stage D — scale to the real run
- Bump to `num_envs: 4096` (OOM fallback `2048` already configured), `epochs: 1500`,
  `checkpoint_interval: 500`, `max_iterations: 12`, `selection_metric: success_rate`.
- Start TensorBoard (in the container, `--network host`) bound to `127.0.0.1:6006`; reach it
  via SSM port-forward (no inbound rule). vLLM stays on the host at `127.0.0.1:8000`.
- Re-run `verify_security_posture.py` once TensorBoard is up so 22.3 covers :6006 too.
- Let the unattended loop run; artifacts + best policy + blog land in S3 `runs/`.

## Risk register (run-specific)

| Risk | Likelihood | Mitigation |
|---|---|---|
| `_IsaacLabTrainer` API drift (parse_env_cfg / RSL-RL cfg schema) | High | Stage A/C shake-out; the adapter already has `try`/`getattr` fallbacks to adjust |
| Env id not registered under that exact name | Low | Fixed to `Isaac-Velocity-Flat-H1-v0` and now a config field (`training.task` → `Config.env_id`); a registry-name drift is a one-line config edit. Confirm in Stage A.3 |
| vLLM + Isaac Sim co-resident VRAM/contention on GPU 0 | Medium | vLLM pinned to GPU 0 only (host); training `CUDA_VISIBLE_DEVICES=1..7`; verify with `nvidia-smi` |
| First generated reward references real-but-wrong `env.*` | Medium | Prompt STATE API + `_EnvAccessProxy` → recoverable ExecutionError, re-prompt |
| torch.compile/FlashInfer re-JIT (only if vLLM moved into a container, option B) | Medium | Default to option A (reuse host vLLM); if B, mount `runs/cache/{vllm,flashinfer}` into the container (lesson docs) |
| `--network host` exposes a service inadvertently | Low | Default-deny SG governs all inbound; vLLM/TensorBoard bind `127.0.0.1`; re-run `verify_security_posture.py` |
| 24h block runs out mid-run | Low-Med | Loop checkpoints each iteration to S3; best policy exported on termination; resume supported |

## Definition of done for Phase 2 run
- Stage C hour-0 smoke produces a checkpoint (Task 16.1) and EvalMetrics from a live env.
- Task 17 final checkpoint: full suite green + smoke checkpoint exists.
- At least one full iteration's artifacts (reward code, checkpoint, metrics JSON, video,
  logs) present under `s3://humanoid-from-scratch-<acct>/runs/`.

## Open decisions for the user
1. **Topology** — recommend **option A**: keep the warm host vLLM and run the Isaac Lab loop
   container with `--network host` (loopback satisfied via shared netns, no weight reload, no
   re-JIT). Option B (vLLM inside the loop container) only if you want one self-contained
   artifact and will mount the compile caches.
2. **Isaac Lab via container vs. pip-into-DLAMI** — recommend the container (matches Dockerfile;
   pins Isaac Sim + Isaac Lab + RSL-RL together).
3. **`src/run_loop.py` entrypoint — DONE** (Stage B). Pure wiring against the existing
   interfaces; every collaborator is injectable and the smoke path is unit-tested off-GPU.

> Note: the `Dockerfile` currently also installs vLLM into the image. Under option A that vLLM
> is unused (we reuse the host's). It's harmless to leave, but we can trim it from the image to
> speed the build if we commit to option A.
