# Lesson: prompting Qwen3-Coder for reliable reward-function code

**Date:** 2026-06-11 (capacity-block sprint, hour 0)
**Where it came from:** building `scripts/probe_qwen_capability.py` and running it against
the self-hosted Qwen3-Coder-30B-A3B endpoint to check the model actually generates usable
`compute_reward` functions (not just reachable plumbing).
**Blog section:** "Self-hosting the language model" → getting good reward code out of the model

## TL;DR

A liveness smoke test ("did the model return parseable Python?") tells you almost nothing
about capability. We wrote a **capability probe** that sends a tightly-specified prompt and
**auto-grades** the output against the hard constraints the Eureka loop depends on. With a
well-structured prompt, Qwen3-Coder produced a correct, vectorized, multi-term reward on the
first try — except it **guessed the `env.*` state-access API**. The lesson: give the model
exact interface facts and grade against machine-checkable rules, and it performs well; leave
any interface implicit and it will confidently hallucinate it.

## What worked in the prompt (do this)

1. **State the observation layout as an exact index map, not prose.**
   Listing `[0:3] base linear velocity … [6:9] projected gravity … [12:31] joint positions`
   made the model slice correctly: it wrote `proj_gravity = obs[:, 6:9]` and used
   `gravity_z` for the upright term. Vague "the obs contains joint info" would not have.

2. **Pin the signature and return contract verbatim.**
   Showing the literal `def compute_reward(env, obs, actions): -> (reward, components)` with
   tensor shapes produced exactly that interface and a proper `components` dict.

3. **Number the hard rules.** A short numbered list ("1. torch only. 2. no os/eval. 3. fully
   vectorized. 4. reward == sum(components). 5. guard NaN/Inf") maps 1:1 onto checks the model
   visibly satisfied — including `torch.nan_to_num` on the total *and* each component.

4. **Name the reward terms you expect.** "Include a goal-distance/progress term, an arrival
   bonus inside the radius, an upright term, an alive bonus, smoothness penalties" gave a
   well-shaped multi-component reward instead of a single vague scalar.

5. **Constrain the output envelope.** "Brief explanation, then the function in a single
   ```python block" makes extraction trivial and deterministic (regex the fenced block, then
   `ast.parse`).

6. **Negative constraints matter.** Explicitly saying "there is NO camera; reward operates on
   environment state only" kept the model from inventing a vision input — a real failure mode
   the spec calls out (Req 1.7).

## What still went wrong (and the real lesson)

The model wrote `base_pos = env.root_pos[:, :2]` and `env.prev_root_pos[...]`. Those attribute
names are **plausible but wrong** — the actual Isaac Lab `ManagerBasedRLEnv` exposes root state
via something like `env.scene["robot"].data.root_pos_w`. The obs *indexing* (which we specified
exactly) was correct; the `env.*` *accessors* (which we left implicit) were hallucinated.

> **The pattern:** Qwen fills any unspecified interface with a confident guess. Whatever you
> specify exactly, it gets right; whatever you leave to it, it invents. For code-gen against a
> real API, the prompt must carry the **exact accessor surface**, or you must provide a binding
> shim so the guesses can't matter.

Two fixes (both now applied):
- **Documented the real state-access API in the prompt template** — listed the exact calls for
  base pose/velocity, joint pos/vel, torques, foot contacts, and the Goal, the same way we
  listed the obs indices. Verified live: the model switched from `env.root_pos` to
  `env.scene["robot"].data.root_pos_w`.
- **Bound through a thin `env` guard in the Reward_Executor** (see next section) so even a stray
  hallucinated accessor fails loudly with actionable feedback instead of a cryptic error.

## Defense-in-depth: an env-access guard for the inevitable miss

The prompt fix handles the common case, but an untrusted model will eventually still invent an
attribute (higher temperature, a refine iteration that drifts, a new task). Relying on the
prompt alone means that miss surfaces as a bare `AttributeError: 'X' has no attribute
'root_pos'` — which tells the refine loop nothing useful and burns a training iteration.

So we wrapped the env handed to generated code in a transparent read-through proxy
(`_EnvAccessProxy` in `src/rewards/reward_executor.py`):

- **Valid access passes straight through** — `env.num_envs`, `env.scene["robot"].data.root_pos_w`,
  item access, and attribute writes all behave exactly as on the raw env. The proxy only
  intercepts the *top-level* `env.<attr>` lookup; nested objects (`.scene`, `.data`) are the real
  ones, so no valid behavior changes.
- **A missing attribute raises a named error** — referencing `env.root_pos` raises an
  `AttributeError` whose message appends `ENV_ACCESSOR_HINT`, the same accessor list we put in the
  prompt. The sandbox captures it as an `ExecutionError` (Req 5.3), and the Orchestrator
  re-prompts the model with that message (Req 12.1).

The lesson generalizes beyond this project: **when you hand untrusted generated code an object
to call into, give it a guarded surface, not the raw object.** The guard turns "silent wrong
attribute → cryptic failure → wasted iteration" into "wrong attribute → precise, self-correcting
feedback." Prompt-side specification and execution-side guarding are complementary: the prompt
maximizes first-try success, the guard makes the residual failures cheap and recoverable.

Covered by `tests/test_reward_executor_env_guard.py` (pass-through, named-error, write-through,
and end-to-end through `WrappedReward.__call__`).

## Grade against machine-checkable rules, not vibes

The probe (`scripts/probe_qwen_capability.py`) parses the returned code and checks:

- **Required:** `ast.parse` ok; defines `compute_reward`; signature `(env, obs, actions)`;
  torch-only imports; no forbidden calls; returns `(reward, components)`; builds a components
  dict; vectorized (no per-env Python loop).
- **Advisory:** has goal/distance, upright, arrival, alive terms; has a NaN/Inf guard.

Two meta-lessons from writing the grader itself:
- **Watch for false positives in your own checks.** Our first "forbidden calls" check used a
  bare substring `"os"`, which matched `pos`, `goal_pos`, `joint_positions`, `cos`, `loss` — so a
  perfectly clean function "failed." Use word boundaries / call patterns (`\bos\.`, `\bopen\s*\(`)
  when scanning generated code, or you'll reject good output.
- **Auto-grading is the point.** With a deterministic rubric you can run the probe at higher
  temperature, sample several generations, and measure a *pass rate* — a real capability signal —
  instead of eyeballing one sample.

## Reusable prompt skeleton

```
ROLE: expert RL reward engineer (Isaac Lab).
TASK: <one concrete sentence; include the concrete Goal/numbers>.
NEGATIVE CONSTRAINTS: <what does NOT exist — e.g. no camera>.
OBSERVATION: <exact index map of the obs tensor>.
STATE API: <exact env.* accessors the code may call>.   # <-- the piece we initially omitted
SIGNATURE: <literal def + return contract + tensor shapes>.
HARD RULES: <numbered, machine-checkable>.
EXPECTED TERMS: <name the reward components you want>.
OUTPUT: brief explanation + single ```python block.
```

## Follow-ups

- [x] Add the exact Isaac Lab state-access API to `prompts/initial_reward.txt` and
      `prompts/refine_reward.txt` (mirror the obs index map style). **Done** — added a
      STATE API / STATE ACCESS block listing `env.scene["robot"].data.root_pos_w`,
      `root_quat_w`, `root_lin_vel_b`, `root_ang_vel_b`, `joint_pos`, `joint_vel`,
      `applied_torque`, the contact-forces sensor, and `env.goal.*`. Verified live: the
      model now emits `env.scene["robot"].data.root_pos_w` etc. instead of the earlier
      hallucinated `env.root_pos` / `env.prev_root_pos`.
- [x] Provide a documented `env` binding shim in the Reward_Executor so generated `env.*`
      reads resolve against a stable surface (defense-in-depth beyond the prompt fix).
      **Done** — `WrappedReward.__call__` now passes the generated reward an
      `_EnvAccessProxy`: valid accessors (incl. `env.scene["robot"].data.*`) pass through
      unchanged, but an unknown top-level attribute (e.g. `env.root_pos`) raises an
      `AttributeError` that names the valid surface (`ENV_ACCESSOR_HINT`), captured as an
      `ExecutionError` so the Orchestrator re-prompts with actionable feedback instead of a
      cryptic error. Covered by `tests/test_reward_executor_env_guard.py`.
- [ ] Use `probe_qwen_capability.py` at temp 0.6–0.8 over N samples to record a pass-rate
      before the sprint, and again after the prompt-template fix to show the lift.
- [ ] Reference this lesson in the assembled Blog (Req 21) under LLM self-hosting.
```
