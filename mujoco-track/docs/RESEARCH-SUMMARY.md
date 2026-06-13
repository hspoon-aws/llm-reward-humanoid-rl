# Research Summary — LLM-Designed Rewards for Humanoid Goal-Reaching on MuJoCo MJX

A self-contained capstone write-up of the whole project: what we set out to do,
what we built, every problem we hit and how we resolved it, the results, and the
open questions. Detailed per-topic lessons are linked; this is the narrative that
ties them together (the blog spine).

---

## 1. Goal

Build an **Eureka-style loop** — an LLM writes a reward function, RL trains a
policy on it, the results feed back to the LLM, which refines the reward —
and use it to teach a **Unitree H1 humanoid** to walk from a start pose (point A)
to a configurable goal (point B) and stop there, upright, on flat terrain.

Originally targeted **NVIDIA Isaac Lab** on an RTX box. Pivoted to **MuJoCo MJX**
(JAX) on a **B200** when we found Isaac Sim can't run on data-center Blackwell
(no RT cores). That pivot is itself a core lesson (§3.1).

Stack: MuJoCo MJX 3.4.0 + MuJoCo Playground (`H1JoystickGaitTracking`, reframed
for goal-reaching) + Brax PPO + JAX 0.6.2, on a `p6-b200.48xlarge` (8× B200).
LLM: self-hosted Qwen3-Coder-30B via vLLM, later Amazon Bedrock Claude Opus 4.8.

---

## 2. Architecture (what we built)

- **Orchestrator loop**: generate reward → validate/inject → train (PPO) →
  evaluate → record → feed metrics back → refine. Resumable via an S3 loop
  checkpoint (Req 16.2).
- **JAX reward contract**: LLM emits `compute_reward(data, action, goal_xy,
  success_radius) -> (reward, components)`; validated (`ast.parse`) and
  jit-checked before training.
- **Goal-conditioned env**: wraps the Playground H1 env, appends a robot-frame
  goal observation (vec-to-goal, distance, heading), swaps the stock reward for
  the LLM's, optional fall-termination + standing-reward floor.
- **Dual LLM backends**: `QwenClient` (self-hosted vLLM) and `BedrockClient`
  (Claude Opus 4.8), selectable by `llm.provider`. Same prompt/extraction logic.
- **Evaluator**: reward-independent metrics (success_rate, distance_to_goal,
  upright_time, fall_rate, path_efficiency) + staged capability gates + best/
  worst demo video.
- **Recovery**: skip-don't-abort on bad candidates, divergence revert at reduced
  LR, service-outage wait/resume, warm-start fine-tuning, best-policy tracking.
- **DI seam**: jax/mujoco/brax imported lazily so the whole loop unit-tests
  off-GPU (200 tests).

---

## 3. The findings (every problem → resolution)

### 3.1 Hardware capability gates — why this project exists
Isaac Sim needs RT cores; the B200 (data-center Blackwell) has none, so Isaac
Sim won't run. MJX is pure-CUDA GPU physics and runs fine. Lesson: match the
simulator to the silicon's capability, not just its FLOPs.
→ `lesson-gpu-architecture-purpose-fit.md`, `lesson-mjx-b200-bringup.md`

### 3.2 Version / framework-contract mismatches (the bring-up tax)
Pinned `mujoco==3.4.0`/`mujoco-mjx==3.4.0` (Playground 0.1.0 calls
`mjx.make_data(nconmax=)` removed in 3.9; H1 XML needs ≥3.4). Sandbox couldn't
`import jax.numpy` (`__import__ not found`); Brax `lax.scan` carry-structure and
unpicklable `make_inference_fn` bugs; metrics pytree mismatch. Every blocker was
a version/contract issue, not an algorithm one — cheap on-hardware smoke gates
(~30s each) caught them before expensive runs.
→ `lesson-problems-and-resolutions.md` (§1–§6)

### 3.3 The reward jit-check is too weak (tracer-leak skips)
`wrap()` validates via `jax.eval_shape` (abstract, unbatched). Some LLM rewards
pass it but fail under the real `vmap`+`lax.scan`+`jit` PPO graph with
`InvalidInputException: ... is not a valid JAX type`. ~30% of candidates skipped
this way, burning the refinement budget. Contained correctly (skip + continue),
but a two-tier check (cheap eval_shape + a real batched jit-check) would convert
most skips into re-prompts. Deferred.
→ `lesson-problems-and-resolutions.md` §7E

### 3.4 An ambiguous status label hid a real bug
"All 12 iterations skipped" looked like LLM/vLLM contention. Real cause: a
progress-logging line referenced `self._n_calls` on the wrong object, crashing
every training run; the generic `except` mislabeled it `skipped_gen_failure`.
The reason text wasn't printed. Fixes: print the reason live, and add a distinct
`skipped_train_failure` status so generation vs training failures never conflate.
Meta-lesson: a status that collapses distinct failure modes will cost you hours;
log the reason, not just the label.
→ `lesson-problems-and-resolutions.md` §7E

### 3.5 Self-hosted vLLM is a shared, serializing resource
Two concurrent loops against one vLLM contended and exhausted retries → all
iterations skipped. Fix: cross-process file lock + generous retries, or run
sequentially, or use a managed endpoint (Bedrock).
→ `lesson-problems-and-resolutions.md` §7D

### 3.6 eval_reward is NOT task success (reward hacking)
With no fall-termination, the H1 learned to "fall toward the goal": eval_reward
went positive while the robot toppled in ~0.4s and never arrived (success_rate
0.0, fall_rate 1.0, but path_efficiency ~0.97 because falling forward is a
straight line). This is exactly why Best_Policy is selected on a reward-
independent metric, never on eval_reward.
→ `lesson-run-v4-findings-and-next-run-plan.md`

### 3.7 Balance before locomotion — and termination is part of the reward
The dominant humanoid lesson. A point-to-point reward lets RL find the
fall-forward cheat. Fix attempt: terminate-on-fall + standing reward. But
terminate-on-fall only teaches standing if **alive is net-positive per step** —
otherwise ending the episode early becomes a *reward* for falling ("die fast").
The env borrowed from a gait-tracking task inherited its no-terminate-on-fall
behavior, silently shaping the wrong policy. Lesson: when you add a terminal
condition, check the sign of per-step return in the surviving state; with an LLM
writing the reward, guarantee the alive signal via an env-side floor + explicit
prompt guidance.
→ `lesson-fall-termination-needs-alive-dominant-reward.md`,
  `lesson-run-v4-findings-and-next-run-plan.md`

### 3.8 Why the loop couldn't continuously improve (the central result)
Three coupled causes, all fixed:
1. **The LLM never saw its own reward code** — the refine prompt sent only
   metrics. "Keep what worked" was impossible; it rewrote blind each iteration.
2. **success_rate tied at 0.0 froze best_policy on iteration 0** — strict-
   improvement tracking never updated, so warm-start always re-seeded from the
   first iteration and a good later policy was never recognized.
3. **Warm-start + big reward rewrites degraded the policy** — a bad reward
   trained on the inherited good policy unwound its balance (1.4s → 0.5s).
Fixes: (1) put the **best reward code** in the prompt + demand incremental edits;
(2) **composite selection score** (primary + tiny upright/closeness tie-breakers)
so best_policy updates before any success; (3) warm-start from the **true best**,
persist `best_reward_code` across restarts. These are the changes that produced
the first working policy.
→ commit `494fc46`.

### 3.9 LLM reward design is bottlenecked by feedback + search budget, not IQ
The LLM gets outcome metrics, not training diagnostics — it can't see *why*
training failed (e.g. its own penalty made standing net-negative). It controls
only the reward, not LR/num_envs/termination. And we ran ~10 sequential single
samples vs. real Eureka's large per-iteration populations with selection. A
stronger model (Opus 4.8) helped, but the ceiling was architectural: richer
feedback (per-component reward breakdown), a sample population per iteration, and
guaranteed env-side scaffolding matter more than raw model quality.

### 3.10 Operational lessons
- **SSM push size limit**: a 73KB source file exceeds the 97KB RunShellScript
  cap when base64'd — gzip+base64 it instead.
- **Stale `.pyc`**: always clear `__pycache__` on the box after pushing changed
  files (a stale pyc caused a confusing signature error once).
- **EBS volume doesn't auto-mount**: after a reboot/stop the `/data` volume
  needed a manual `mount /dev/nvme2n1 /data`; add an fstab entry with `nofail`.
- **Persist the resume state in S3**: the loop checkpoint in S3 (with
  best_policy + best_reward_code) made a post-reboot resume seamless — the run
  picked up warm-starting from the best policy with eval_reward ~500, not a cold
  ~40.
→ `backup-and-restore.md`, `multi-gpu-run-recipe.md`

### 3.11 Self-host vLLM vs Bedrock, and goal-change / re-training
- Bedrock (managed) removes the single-vLLM contention point and frees GPU 0 for
  training; Opus 4.8 is inference-profile-only and rejects `temperature`.
- The policy is **goal-conditioned**, so it generalizes across goal positions
  and is a natural warm-start; "change the goal position" is a config edit,
  "change the objective" is a new experiment (history + eval gates are
  task-specific).
→ `lesson-goal-change-and-retraining.md`

---

## 4. Results

Progression across runs (selection on success_rate; upright_time_s = time before
fall; goal at 2 m, success radius 0.75 m in the tuned runs):

| Run | LLM | key settings | best upright_s | best success_rate |
|---|---|---|---|---|
| v4 (1-GPU, 6-GPU) | Qwen | LR 1e-3, goal 5 m, no fall-term | ~0.4–0.5 | 0.0 (falls/“fall-toward-goal”) |
| v2 | Qwen | fall-term, standing 0.5, LR 3e-4 | ~0.4 | 0.0 (“die fast” — alive too weak) |
| v3 | Opus 4.8 | fall-term, standing 4.0, 8190 env | 0.72 (bounced) | 0.0 (no continuous improve) |
| **v5** | **Opus 4.8** | **+ all §3.8 fixes, 8 GPUs, warm-start** | **1.87** | **0.19** |

**v5 is the breakthrough**: iteration 1 — upright 1.87 s, within 1.07 m of the
goal, **success_rate 0.19** (the first policy to actually reach the goal). Upright
stayed in the 1.5–1.9 s band across iterations (no catastrophic forgetting),
peaking at iteration 4 (116-step episodes, eval_reward 856 — its goal-success was
not re-evaluated before the instance stopped). Best checkpoint:
`runs-mujoco/run-8gpu-v5-improved/iter_02_policy.pkl`.

**Honest caveats**: improvement was not strictly monotonic (per-iteration reward
variance still causes dips; the composite score protects the best). Goal success
beyond iter 1 (notably the strong-looking iter 4) was not confirmed because the
box was stopped before those eval metrics uploaded.

---

## 5. What worked / what we'd do next

**Worked**: MJX on B200; DI seam + off-GPU tests; skip-don't-abort recovery;
reward-independent eval gates (caught reward hacking); warm-start from the true
best; incremental-refinement prompt with the best reward code; Bedrock backend;
8-GPU data-parallel; S3-checkpoint resume across a reboot.

**Next**: two-tier jit-check to reclaim skipped iterations;
per-iteration reward *population* with selection (true Eureka) to beat variance;
richer feedback to the LLM (per-component reward breakdown); cross-run warm-start
seed flag; re-evaluate iter-4/5 checkpoints; demo video of the best policy;
curriculum (stand → walk → farther goals).

---

## 6. Document index
- `lesson-problems-and-resolutions.md` — chronological problem/fix log (§1–§7E).
- `lesson-gpu-architecture-purpose-fit.md` — RTX vs Blackwell, B200 vs L40S.
- `lesson-mjx-b200-bringup.md` — MJX/Playground/Brax bring-up on the B200.
- `lesson-run-v4-findings-and-next-run-plan.md` — reward hacking, fall analysis.
- `lesson-fall-termination-needs-alive-dominant-reward.md` — termination = reward.
- `lesson-goal-change-and-retraining.md` — goal-conditioning, transfer, resume.
- `multi-gpu-run-recipe.md` — data-parallel num_envs rules.
- `backup-and-restore.md` — durability + restore procedure.

## 7. The one-line thesis
An LLM can design and refine humanoid reward functions well enough to produce a
real walking-to-goal policy — but only when the loop gives it (a) its own best
reward to build on, (b) a selection signal that can rank near-failures, and
(c) a policy that carries forward (warm-start from the true best). The hard part
was never the LLM's reasoning; it was the **scaffolding around it**.
