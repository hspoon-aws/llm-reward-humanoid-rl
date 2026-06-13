# Lesson: Fall-Termination Only Teaches Standing If the Alive Signal Dominates

A subtle but important RL-reward lesson surfaced when tuning run v2 to fix the
"H1 falls in 0.4 s" problem from run v4. Capturing it while the evidence is fresh.

---

## Background

Run v4 finding: the H1 fell almost immediately every episode and learned to
"fall toward the goal" because the base gait-tracking env **never terminates on a
fall**, so a fallen robot kept accumulating goal reward for the full ~1000-step
episode. (See `lesson-run-v4-findings-and-next-run-plan.md`.)

Run v2 fix attempt: add **fall termination** (`fall_terminate=true`, end the
episode when torso z < `fall_height_m=0.6`) + a small **standing reward floor**
(`standing_reward=0.5`/step). Lower LR (3e-4), more envs (8190), closer goal (2 m).

---

## What we observed (v2, first completed iteration = iter 6)

Trainer metrics:
- `avg_episode_length = 5.0` steps  → **fall termination fires correctly** (vs
  the full 1000 in v4). The mechanism works.
- `eval/episode_reward = -16.4` over those ~5 steps, with components:
  `alive_bonus +0.5`, `upright_penalty -8.99`, `pose -33.3`, `progress -9.97`.

Evaluator goal metrics (S3 `iteration-06/metrics.json`):
- `success_rate = 0.0`, `upright_time_s = 0.34`, `fall_rate = 1.0`,
  `distance_to_goal_m = 2.24` (goal at 2 m) → **still falling immediately.**

---

## The trap: terminate-on-fall + net-negative standing = "die fast" incentive

Fall termination changes the incentive landscape, but **not always in the
direction you want**. PPO maximizes expected return (sum of per-step reward until
the episode ends). With terminate-on-fall there are two regimes:

- **Alive is net-POSITIVE per step** → a longer episode accumulates more reward,
  so ending early (falling) is *bad*. PPO is pushed to **stay upright**. ✅ This is
  the regime that teaches standing.
- **Alive is net-NEGATIVE per step** → every extra upright step *subtracts* from
  return, so the fastest way to stop the bleeding is to **end the episode early**,
  i.e. **fall on purpose**. ❌ Termination then *accelerates* falling.

The v2 iter-6 numbers are squarely in the bad regime: total per-step reward while
alive ≈ −16/5 ≈ −3.3, dominated by `pose −33` and `upright_penalty −9`, against a
mere `alive_bonus +0.5`. So terminate-on-fall did NOT create a standing incentive
— it arguably reinforced a "die fast" one. This is the **inverse of the v4 hack**:
v4 = "fall toward the goal" (fall is free); v2 = "fall to end the pain" (fall is
profitable). Same root cause: the **alive/survival signal does not dominate**.

---

## The rule

> For terminate-on-fall to teach standing, **being upright must be net-positive
> per step**, by a margin large enough that no combination of the other
> (penalty) terms can make a longer episode worse than a short one.

Equivalently: `alive_bonus  >  |sum of all per-step penalties while upright|`.

This is exactly how production humanoid-locomotion rewards are built: a large,
near-unconditional **survival/alive bonus** is the dominant term, and everything
else (pose, energy, tracking) is a comparatively small shaping correction.

---

## Why the LLM-generated reward makes this harder

In this Eureka setup the **LLM writes the reward**, and it is free to emit large
standing penalties (`pose: -33`, `upright_penalty: -9`) that swamp our fixed
`standing_reward=0.5`. So even with fall termination on, the loop depends on the
LLM happening to make alive dominate — which it won't reliably do without being
told. Our `standing_reward` floor (0.5) is far too small to guarantee dominance.

---

## Fixes for the next run

1. **Raise `standing_reward` substantially** (e.g. 2–5 /step) so being upright is
   unambiguously net-positive regardless of the LLM's penalty terms. It is added
   by the env *outside* the LLM reward (`GoalReachingEnv.step`), so it's a
   guaranteed floor.
2. **Steer the prompt:** instruct the LLM that the survival/upright term MUST
   dominate and penalties must stay small relative to it; reward upright
   *duration*, not just instantaneous pose.
3. **Add tilt-based termination** (not just height) and consider a **curriculum**:
   Phase 1 = pure standing (no goal reward, big alive bonus, terminate on fall)
   until `upright_time_s` climbs past a few seconds; Phase 2 = add goal-reaching,
   warm-started from the standing policy. Humanoids almost always need this.
4. **Reduce the skip drain** (Tier-2 jit-check): v2 lost iters 0-5 to
   tracer-leak skips, leaving very few real refinement shots within
   `max_iterations`.

---

## Meta-lesson for the blog

**A termination condition is part of the reward, not separate from it.** "End the
episode on fall" only discourages falling if staying alive is worth more than
dying — otherwise early termination becomes a *reward* for failing. When you add a
terminal condition, always check the sign of the per-step return in the surviving
state. With an LLM writing the reward, you must additionally *guarantee* the alive
signal via an env-side floor and explicit prompt guidance, because the model will
not reliably balance the terms on its own.

> Status: hypothesis formed after v2 iteration 6. To be confirmed by watching
> whether `upright_time_s` stays flat (~0.3 s) over the next few completed
> iterations; if it does, the "die fast" diagnosis holds and fixes 1-3 apply.
