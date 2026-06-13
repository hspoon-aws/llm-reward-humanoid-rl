# Research Learnings: LLM → Reward → RL for Humanoid Locomotion

**Project:** Teach a Unitree H1 humanoid point-to-point goal-reaching in NVIDIA Isaac Lab
using an Eureka-style loop where an LLM authors the reward function, PPO (RSL-RL) trains the
policy, and evaluation feedback drives the next reward.
**Scope of this doc:** the cross-cutting learnings from the whole project — the conceptual
thesis, what worked, and the bugs that cost the most time. Detailed single-topic write-ups
live in the sibling `lesson-*.md` files (cross-referenced inline).

---

## 0. The thesis (why this project matters)

**You program a robot's behavior in plain English, and the system compiles that into a trained
policy.** The control surface is a natural-language task description; the LLM turns it into a
reward function; the RL loop turns the reward into behavior; evaluation verifies it and feeds
back. No human hand-writes or hand-tunes reward math.

This was demonstrated end-to-end: changing one English sentence ("walk forward facing the goal;
never walk backward") changes the generated reward, which changes the gait — without editing a
line of reward code. That is the practical payoff: the hard, expert-only part of RL (reward
engineering) is amortized into the LLM, and the simulator + loop verify it empirically so you
don't have to trust the LLM blindly.

### How can an LLM author a reward at all?
It is not deriving rewards from physics first principles. It pattern-matches against the large
body of RL/robotics code and papers it saw in pretraining (Isaac Lab, MuJoCo, legged_gym,
RSL-RL, the Eureka paper itself), where locomotion rewards are written out as canonical
recipes: progress term + upright term + alive bonus + smoothness/energy penalties. The prompt
supplies what the model *cannot* know — the exact env API and the specific task. The LLM
provides an *informed guess*; the simulator provides *ground truth*; the loop closes the gap.
The model cannot predict emergent exploits (e.g. that "reward distance reduction" yields
backward walking) — only training reveals those, which is precisely why the verification loop
is non-optional.

---

## 1. The headline technique: natural-language behavior control

- The task description is a first-class config field (`config.task_description`); set it in YAML
  and the operator steers behavior in plain English. Falls back to a built-in default.
- The same idea applies to *corrections*: when the robot did something wrong (walked backward),
  the fix was to add English guidance to the prompt ("face the goal, reward forward velocity,
  penalize backward/lateral"), not to write a reward term by hand.
- **Lesson:** treat the prompt/description as the program. Behavior bugs are first debugged in
  natural language. This is what makes the technology usable by non-RL-experts.

See also: `lesson-prompting-qwen-for-reward-code.md` (how to structure the prompt so the model
emits correct, vectorized, multi-term reward code on the first try).

---

## 2. Reward gaming is the default, not the exception

Two concrete reward-gaming exploits the loop produced, both fixed by better English guidance:

- **"Fall toward the goal" (MuJoCo track).** With no fall-termination, the policy learned to
  topple in the goal direction to collect distance-reduction reward. Fixed there with
  fall-termination + a standing reward. Isaac's stock H1 task already terminates on torso
  contact, so it did not exhibit this — a reminder that the *environment's* built-in
  terminations matter as much as the generated reward.
- **Backward / sideways walking (Isaac track).** The generated reward credited
  distance-reduction and signed velocity-toward-goal but had **no facing/heading term**, so a
  backward shuffle collected full reward and still "reached" the goal. Fix: prompt guidance to
  reward alignment of the robot's forward (+x base) axis with the goal bearing and reward
  positive forward base velocity, penalize backward/lateral — strong enough that backward
  locomotion is no longer reward-optimal.

**Lesson:** a reward that is *plausible* is almost never *complete*. Any objective the reward
does not explicitly constrain, the optimizer will exploit. Watch the video, not just the
number — see §5.

---

## 3. The metric must measure what you think (the env-frame bug)

The single most expensive correctness bug. A run reported a clean 0 → 0.98 `success_rate`
trend, but the **video showed the robot falling**, and `distance_to_goal_m` read ~30 m for a
2 m goal.

**Root cause — coordinate-frame mismatch.** Isaac Lab lays the parallel envs on a grid spaced
metres apart, so `robot.data.root_pos_w` is the **absolute world** position *including each
env's grid-origin offset* (`scene.env_origins[i]`, tens of metres). The Goal is stored
**env-local**. Comparing a world position against an env-local goal corrupts every
distance/heading/success computation — the "distance" was reading the env-grid offset, not the
real distance.

It corrupted **three** places, not just the displayed metric:
1. **Eval metrics** — `success_rate`, `distance_to_goal_m`, path efficiency.
2. **The Goal_Observation fed to the policy** — so the policy trained on a corrupted goal
   vector, not merely got mis-scored.
3. **The generated reward** — the prompt told the model to use raw `root_pos_w` against the
   local goal, so the reward signal was wrong too.

**Fix:** subtract `scene.env_origins` to get env-local position everywhere that compares to the
local goal (eval `_read_base_xy`, goal_env `_read_base_pose`, and the prompts so generated
rewards localize). Fall back to world==local when `env_origins` is absent (single-env scene /
test fakes).

**Lessons:**
- In a multi-env simulator, **always confirm which frame each quantity is in.** "World" position
  is per-env-offset; the goal is local. A frame mismatch produces numbers that look plausible
  (monotonic, bounded) but are meaningless.
- A metric and the behavior it claims to measure must be cross-checked against ground truth
  (the video). The metric lied; the video told the truth.
- Prior run metrics computed under a frame bug are **not salvageable** — re-run after the fix.

---

## 4. Isaac Sim process model: one app, one env, per process

Isaac Sim permits exactly **one `SimulationApp` and one live manager-based `SimulationContext`
per process**, and does not cleanly tear one down to build another. This single constraint drove
a chain of architecture decisions and bugs. See `lesson-isaac-lab-bringup.md` for the full
blocker chain; the load-bearing learnings:

- **In-process second env build = stall or crash.** Building a second manager-based env in a
  process that already built one either raises "Simulation context already exists" or hangs
  forever during the second scene setup (GPU idle, CPU spinning). This bit us via: the in-loop
  video recorder, the OOM-fallback retry, and the divergence-recovery retry — all of which
  silently tried to build a second env.
- **One iteration per process (the durable fix).** Run each Eureka iteration in its own fresh
  subprocess (`scripts/run_iteration.py` + a chunked driver). State crosses processes via the
  S3 `loop_checkpoint.json` (resume mechanics), so the multi-process run is behaviorally
  identical to a single-process loop but immune to the second-env problem.
- **Don't retry training in-process.** Divergence (non-finite loss) and runtime errors in the
  generated reward now record a `SKIPPED_DIVERGENCE` / `SKIPPED_RUNTIME` iteration and let the
  *next fresh process* refine, rather than rebuilding an env in-process. Pre-execution
  *validation* failures still re-prompt in-loop (they never build an env, so it's safe).
- **Hard-exit at process end.** `SimulationApp.close()` frequently hangs; since all artifacts
  are already on S3 by the time the iteration returns, `os._exit()` after printing the
  completion marker is the reliable way to release to the next iteration.
- **Cameras + single GPU = one env for recording.** Multi-env tiled RTX cameras trigger
  device-side asserts on a single GPU; the demo recorder renders one env in a fresh process.

---

## 5. Verify against video, and be honest about "verified"

Repeatedly, a green number hid a real failure:
- "Recording works" ≠ "the video is usable" — the camera aimed at empty space (see §6).
- "0.98 success" ≠ "the robot walks to the goal" — the frame bug (§3); the robot fell.
- "Reaches the goal" ≠ "walks forward" — reward gaming (§2); it walked backward.

**Lesson:** for an embodied task, the rendered episode is the ground truth. Automated metrics
are necessary (you can't watch 1024 envs) but must be cross-checked against video, and claims
should distinguish "the mechanism ran" from "the result is correct." A cheap automated
framing/behavior check (motion-diff centroid on the mp4) catches "robot not in frame" and
"robot not moving forward" without a human in the loop.

---

## 6. Demo-camera framing (fixed world-frame cameras)

The recorder produced valid mp4s but the robot wasn't in frame.
- **Convention bug:** the look-at quaternion was built for the OpenGL/ROS optical convention
  (−Z forward) while `CameraCfg.OffsetCfg` used `convention="world"` (+X forward, +Z up).
  Confirm which axis convention the consuming API expects — read the installed source.
- **Framing for a moving subject:** these are *fixed* world-frame cameras (they don't track).
  Use a wide FOV (short focal length) + generous standoff + aim at the path midpoint so the
  whole A→goal walk stays framed regardless of where the robot is.
- **Verification without ffmpeg:** inter-frame motion differencing localizes the (only moving)
  robot; centroid near frame-center ⇒ framed. Brightness/dark-pixel heuristics are unreliable
  (they latch onto ground shadow).

---

## 7. RL is continuous learning — warm-start across iterations

The loop originally rebuilt a randomly-initialized policy every Eureka iteration, throwing away
all prior training. On a fixed goal with successively-refined rewards, that re-climbs the
~15M-step curve each time.

**Fix (opt-in `warm_start`):** each iteration loads the previous iteration's trained checkpoint
(`OnPolicyRunner.load`) and continues. Prefer the tracked Best_Policy checkpoint, else the most
recent completed iteration's. The checkpoint persists on disk between the one-iteration-per-
process driver's containers (shared workspace mount).

**Caveats:**
- Canonical Eureka trains each reward candidate *from scratch* so reward comparisons are
  unbiased. Warm-start trades that purity for fast convergence on a fixed demo goal — the right
  call here, but know the tradeoff.
- Warm-start propagates *bad habits* too: if an early iteration learns backward walking, later
  iterations start from it. Pair warm-start with a corrected reward, or restart clean.

---

## 8. Config that's loaded vs config that's ignored

Two latent bugs where a YAML field existed but was never read (found on both tracks):
- `learning_rate` was hard-coded to 1e-3 in the trainer; the YAML value was ignored. PPO LR is
  one of the highest-impact knobs (1e-3 oscillates; 3e-4 is stable for this task).
- `num_steps_per_env`, the Qwen retry/backoff/timeout, and others were similarly stranded.

**Lesson:** a config field is only real if it's threaded `Config → consumer`. Audit that every
declared knob actually reaches the code that should use it; a defaulted-but-unread field is a
silent footgun. See `lesson-training-budget.md`.

---

## 9. Training budget on a single GPU

Total env-steps per iteration = `num_envs × num_steps_per_env × epochs`. Published Isaac/RSL-RL
H1 locomotion converges in ~50–200M steps. On one L4 at 1024 envs:
- Search phase (rank many reward candidates cheaply): `1024 × 24 × 600 ≈ 15M` steps/iter.
- The full 8-GPU sprint config uses `epochs=1500 → 4096×24×1500 ≈ 147M` (real convergence
  budget). Do **not** copy the big-iron epoch count to a single L4 — it's both slow and, if you
  shrink envs without raising epochs, below the learning floor.

See `lesson-training-budget.md` and `lesson-multi-gpu-scaling.md` (Isaac is multi-GPU-*config*
but single-process-*executing*; for Eureka, prefer parallel reward *candidates* over DDP).

---

## 10. Self-hosted vLLM vs Bedrock for the reward LLM

Both backends were built behind one provider switch (the prompt/extraction/validation are
provider-agnostic; only the chat call differs).
- **Self-hosted Qwen3-Coder-30B (vLLM):** full control, no per-call cost, but a real ops burden
  — a separate GPU box, a shared file lock, cross-host networking, FlashInfer/Blackwell JIT
  pain (`lesson-flashinfer-jit-blackwell.md`), and it occupies a GPU.
- **Amazon Bedrock (Claude Opus 4.8):** no self-hosting, frees the GPU for training, managed
  scaling. Gotchas found live: the model id is `global.anthropic.claude-opus-4-8` (no `-v1:0`
  suffix — list inference profiles to confirm); Opus 4.8 **rejects the `temperature` knob** in
  the Converse API; the instance role needs `bedrock:InvokeModel`.
- **Lesson:** keep the LLM behind a narrow interface (`generate_reward` + a `_chat` seam) so the
  backend is a config switch. See `lesson-self-host-vllm-vs-bedrock.md`.

---

## 11. Sandbox vs LLM code style

The generated reward runs in a restricted sandbox (allowlisted builtins, no `open`/`eval`/`os`).
Claude reliably writes `import torch` *inside* `compute_reward`; the sandbox's dict-`__builtins__`
has no `__import__`, so this failed with "ImportError: __import__ not found" every iteration.
**Fix:** provide a restricted `__import__` that resolves only allowlisted modules (torch, math)
and blocks everything else — generated in-body imports work without widening the sandbox.
**Lesson:** the sandbox contract must match how the model actually writes code, not how you wish
it would; meet the model where it is when it's safe to.

---

## 12. AWS / infra learnings (brief)

- **Single-AZ capacity is flaky.** g6 stop/start hits `InsufficientInstanceCapacity` in
  single-AZ us-west-2b; budget for 2–3 start retries. Prefer stopping when idle for cost, but
  expect restart friction. See `lesson-aws-launch-gotchas.md`.
- **SSM is the control plane.** Long/backgrounded commands go through uploaded script files +
  `nohup`; patches staged to S3 then pulled onto the box. Don't inline complex quoting.
- **Persist early, persist to S3.** Every iteration's artifacts (reward.py, checkpoint, metrics,
  videos) + `loop_checkpoint.json` are on S3 before the process can hang/exit, so a killed run
  loses nothing and resumes by run-id.
- Docker build on the GPU host, EBS-snapshot image hydration: see the respective `lesson-*.md`.

---

## 13. What a healthy run looks like (and the open caveat)

A Bedrock + warm-start run produced an *apparent* 0 → 0.98 `success_rate` climb over 7
iterations with rising upright time and falling fall-rate — a textbook Eureka curve. **But** it
ran before the env-frame fix (§3) and the forward-facing steering (§2), so those numbers are
not trustworthy and the policy did not cleanly walk to the goal. The mechanics (Bedrock reward
gen, warm-start handoff, single-env-per-process, recording) are all proven; a clean re-run with
the §2 and §3 fixes is required to claim a real result.

**Meta-lesson for the project:** the pipeline being *green* and the robot being *good* are
different claims. This project's recurring failure mode was trusting the former for the latter.
The discipline that caught every real bug was: read the video, check the frame, question the
number.

---

## Index of detailed lessons
- `lesson-prompting-qwen-for-reward-code.md` — prompt structure for reliable reward code
- `lesson-isaac-lab-bringup.md` — the Isaac Sim blocker chain, single-app/env constraint, recorder
- `lesson-training-budget.md` — env-steps math, the loaded-vs-ignored config audit
- `lesson-multi-gpu-scaling.md` — Isaac multi-GPU config vs single-process execution
- `lesson-self-host-vllm-vs-bedrock.md` — LLM backend tradeoffs
- `lesson-flashinfer-jit-blackwell.md` — vLLM on Blackwell JIT issues
- `lesson-aws-launch-gotchas.md`, `lesson-docker-build-on-gpu-host.md`,
  `lesson-ebs-snapshot-hydration.md` — infra
