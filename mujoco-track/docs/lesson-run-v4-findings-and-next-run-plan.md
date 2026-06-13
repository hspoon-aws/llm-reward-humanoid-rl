# Run v4 (A/B) — Findings & Improvement Plan for the Next 6-GPU Run

First full A/B run of the MuJoCo Eureka loop on the B200. The pipeline worked
end-to-end; the policy did not. This note records what actually happened (from
metrics, not vibes) and the prioritized plan to get the H1 walking next time.

---

## 1. What we ran

| | 1-GPU (GPU 1) | 6-GPU (GPUs 2-7) |
|---|---|---|
| Config | `run_config.yaml` | `run_config_6gpu.yaml` |
| num_envs | 2048 | 3072 |
| epochs/iter | 75 (=153.6M steps) | 75 (=230.4M steps) |
| learning_rate | 1e-3 | 1e-3 |
| max_iterations | 12 | 12 |
| Goal | reach (x=5, y=0), success radius 0.5 m | same |

Both share one self-hosted vLLM (GPU 0) with a cross-process request lock.

---

## 2. The result (6-GPU run, completed all 12 iterations)

```
[run] best policy: iteration 2  score=0.0  checkpoint=.../iter_01_policy.pkl
```

**Best policy `success_rate = 0.0`.** Reward-independent eval metrics, every
completed iteration (2-8):

| iter | success_rate | distance_to_goal_m | upright_time_s | fall_rate |
|------|--------------|--------------------|----------------|-----------|
| 02   | 0.0 | 5.68 | 0.48 | 1.0 |
| 03   | 0.0 | 4.97 | 0.40 | 1.0 |
| 04   | 0.0 | 5.71 | 0.52 | 1.0 |
| 05   | 0.0 | 4.62 | 0.46 | 1.0 |
| 06   | 0.0 | 5.20 | 0.50 | 1.0 |
| 07   | 0.0 | 4.44 | 0.34 | 1.0 |
| 08   | 0.0 | 4.90 | 0.45 | 1.0 |

Goal is at 5 m; the robot ends ~4.4-5.7 m away (i.e. essentially at the start, or
drifted back). **`upright_time_s ≈ 0.3-0.5 s` and `fall_rate = 1.0`: the H1 topples
in under half a second, every episode.**

Of 12 iterations, **8 completed, 4 were `skipped_train_failure`** (the tracer-leak
class, §7E) — so only 8 real reward-refinement shots.

---

## 3. Diagnosis (why positive eval_reward ≠ success)

1. **Reward-hacking via "fall toward the goal."** Earlier we saw `eval_reward` go
   positive (+11 to +15) on some iterations and read it as progress. The ground
   truth: those rewards paid for *instantaneous velocity toward the goal* and an
   *alive bonus* in the first few frames, so PPO found the cheapest way to collect
   them — **lunge/topple forward**. `path_efficiency` was even ~0.97 because falling
   forward is a straight line. This is exactly why Best_Policy is selected on
   `success_rate`, not `eval_reward`.

2. **The env does not terminate on fall (root enabler).** `GoalReachingEnv.step`
   delegates `done` to the base `H1JoystickGaitTracking` env, which is a
   *gait/velocity-tracking* env and generally does **not** end the episode when the
   robot falls. So a robot that falls at 0.4 s keeps accumulating reward for the
   remaining ~19 s of the episode — training it to fall *efficiently in the goal
   direction*. **This is the single biggest enabler of the failure.**

3. **Balance was never learned as a prerequisite.** Walking to a point requires
   standing first. The rewards jumped straight to goal-seeking; with only 8
   refinement shots (4 lost to skips) and no fall-termination, the loop never
   escaped the fall-forward local optimum. The `fall_threshold_s=3.0`
   balance-priority guidance should have fired every iteration (upright was 0.4 s
   << 3 s) — verify it actually reached the prompt and make it far more forceful.

4. **Hyperparameters off the tuned recipe (confound).** LR 1e-3 (vs Playground
   tuned 3e-4) and num_envs 3072 (vs 8192) → noisy, oscillating gradients that make
   a hard balance-then-walk task harder. Step budget is NOT the problem (we already
   exceed the 100M reference per iteration).

**Headline:** the pipeline is sound (12/12 iterations, recovery worked, artifacts +
videos persisted); the result reveals the classic humanoid pitfall — *reward balance
before locomotion, or RL finds the "fall toward the goal" cheat.*

---

## 4. Improvement plan for the next 6-GPU run (prioritized)

### Tier 1 — directly attack "falls in 0.4 s" (do these first)
1. **Terminate the episode on fall.** Add a fall termination to `GoalReachingEnv`
   (e.g. `done = base_z < fall_height` and/or large torso contact). This removes the
   post-fall reward stream that trains fall-forward — the **highest-impact** change.
   Pair with a per-step **alive/standing reward floor** so standing is positively
   reinforced.
2. **Make goal reward conditional on being upright.** Gate the goal/velocity terms
   behind "upright for > N seconds" (the LLM already discovered a weak version of
   this in iter 6 — make it the default contract via stronger prompt guidance).
3. **Strengthen balance-priority guidance** and verify it reaches the prompt when
   `upright_time_s < fall_threshold_s`. Wording like: "the robot MUST remain upright
   for ≥2 s before any goal-directed reward applies; prioritize balance."

### Tier 2 — stability + more refinement shots
4. **Adopt the tuned PPO recipe:** `learning_rate: 3e-4`, `num_envs: 8192`
   (6-GPU box has headroom). Removes the instability confound.
5. **Tier-2 batched jit-check** to convert tracer-leak skips into re-prompts →
   ~12 real iterations instead of 8. More Eureka refinement shots.

### Tier 3 — task framing / curriculum
6. **Start the goal closer (1.5-2 m)** so `success_rate` can become non-zero and the
   reward-refinement loop has a real gradient (a flat 0.0 success gives the LLM
   almost nothing to refine against). Move it farther in later runs.
7. **Two-phase curriculum (stand → walk),** ideally with the warm-start seam
   (warm-start seam): learn standing first, then warm-start goal-reaching from a
   policy that can already balance.

### Minimal high-impact next run (recommended)
Config + one env change, no architecture work:
- `learning_rate: 3e-4`, `num_envs: 8192`
- **fall termination + standing reward floor** in `GoalReachingEnv`
- goal at **2 m**
- stronger balance-priority prompt guidance

Then evaluate on `success_rate` / `upright_time_s` again. If upright time climbs
past a few seconds, the rest (goal-reaching) becomes learnable.

---

## 5. What NOT to change
- **Step budget / epochs:** already above the 100M H1 reference per iteration; not
  the bottleneck.
- **Selection on `success_rate`:** working as intended — it correctly refused to
  call a fall-forward policy "good" despite positive eval_reward. Keep it.
- **Skip-don't-abort recovery:** working; the fix is to reduce skips (Tier-2 check),
  not to change the recovery.

---

## 6. Blog takeaways
1. **`eval_reward` is not success.** A reward can rise while the robot fails the
   task (reward-hacking). Always keep a reward-independent scorecard
   (`success_rate`, `upright_time_s`, `fall_rate`) and select on it.
2. **Balance before locomotion.** The dominant humanoid lesson — without fall
   termination and an upright prerequisite, RL discovers "fall toward the goal."
3. **Borrowed envs carry borrowed assumptions.** Reusing a gait-tracking env for a
   goal-reaching task inherited its no-terminate-on-fall behavior, which silently
   shaped the wrong policy. Audit the base env's `done`/reward semantics when you
   repurpose it.
4. **Refinement count is the budget that matters for Eureka.** Losing 4/12
   iterations to a containable validation gap measurably reduced the loop's chance
   to fix the reward. Cheap, realistic up-front checks pay for themselves.
