# Lesson: Changing the Goal & Re-training a Trained Policy

A recurring blog-worthy question on this project: **once we have a trained
policy, can we change the goal — mid-run or afterward — and re-train instead of
starting from scratch?** The answer is yes, with important nuances that come
straight from how the system is wired. This note captures them for the writeup.

---

## 1. What the LLM is actually told the goal is

Every reward-generation request carries two frozen-at-startup pieces:

- **Task instruction** (`DEFAULT_TASK_DESCRIPTION`, a module constant):
  > "Teach a Unitree H1 humanoid to walk from its start position (point A) to a
  > configurable target position (point B) on flat terrain. The robot must arrive
  > within the success radius of the Goal while staying upright and moving
  > efficiently. This is a point-to-point goal-reaching task, not a
  > velocity-tracking task."
- **Concrete goal line** (`_goal_description()`, filled from config):
  > "Reach the Goal at (x=5, y=0) on the ground plane and stop within 0.5 m of it
  > without falling."

The reward functions the LLM writes therefore reward **velocity/heading toward
the goal + an arrival bonus + a stop-near-goal penalty**, NOT raw walking
distance. Rewarding total distance would be wrong for a "reach B and stop" task
(it would encourage pacing past the target). This is the correct shape for
point-to-point goal-reaching, and it is what the LLM converged on by iteration
4-6 on its own.

---

## 2. Can we change the instruction MID-run?

**Not in the already-running process without a restart.** Both the instruction
(`DEFAULT_TASK_DESCRIPTION`, a module-level constant bound at import) and
`self.goal` (built once in `Orchestrator.__init__` from the startup config) are
**frozen in process memory**. Editing the file or config on disk does nothing to
a live process — there is no hot-reload path.

**But cleanly, at an iteration boundary, yes — via checkpoint resume (Req 16.2):**
1. Let the current iteration finish (consistent `loop_checkpoint`).
2. Stop the process.
3. Edit the instruction and/or `goal_position` in config.
4. Relaunch with the same `--run-id` / checkpoint-dir → it resumes at the next
   iteration, now generating rewards against the new goal while keeping prior
   `MetricsHistory` and Best_Policy.

### Caveats that make this an experiment-design choice, not a free tweak
- **Metrics history becomes inconsistent.** Iterations scored against "reach x=5"
  are misleading feedback for a different objective. For a *fundamental* task
  change you'd want to reset the history.
- **Eval gates are task-specific.** `distance_to_goal_m`, `reaches_goal`,
  `success_rate` are wired to goal-reaching. A non-goal-reaching instruction
  makes the scorecard — and Best_Policy selection — meaningless.

**Rule of thumb:**
- *Move the goal* (x=5 → x=8, change radius): low-friction, same metric shape,
  restart-resume is fine.
- *Change the whole task* (goal-reaching → max-distance walking): needs a **new
  run** — new instruction, new goal handling, new eval metrics/gates, fresh
  history.

---

## 3. Can we re-train a TRAINED model with a new goal?

Yes — this is fine-tuning / transfer learning, and it's a strong use case here
because **the policy is goal-conditioned**: its observation already includes the
robot-frame `vec_to_goal`, `distance`, and `heading`. A policy trained to reach
x=5 has learned "walk toward whatever the goal vector points at," not "walk to one
hard-coded spot."

### Level 1 — same task shape, new goal position (easy, supported today)
Moving the goal (x=5 → x=10) is largely **in-distribution**. The trained policy
may reach the new target **zero-shot**, and a short fine-tune sharpens it. Today
this needs only a config change + a new run; the goal-conditioning does the heavy
lifting. **This is the cleanest blog demo: take the Best_Policy, move the goal,
show it still arrives — generalization for free from goal-conditioning.**

### Level 2 — faster convergence via warm-start (small code gap to close)
True fine-tuning = continue optimizing the *same weights* rather than retraining
from random init. The pieces exist (`_load_policy` rebuilds the full Brax policy
from `params + <path>.netcfg.json`; the loop already resumes from checkpoints),
**but `PPORunner.train` currently builds a fresh PPO from random initialization
each iteration** — it does not yet thread "initialize network params from this
checkpoint." Brax's `ppo.train` supports warm-starting; the project just doesn't
wire it through.

Closing the gap is a contained change: thread an optional warm-start checkpoint
through `TrainConfig` → `_MjxTrainer` → `ppo_train.train` (e.g. a
`restore_checkpoint_path` / load-params-into-init seam). With it, a new-goal
fine-tune converges in far fewer steps than from scratch — the whole point of
transfer learning.

### Level 3 — genuinely new skill (new run, optionally warm-started)
Goal-reaching → walk-and-turn / climb / etc. is transfer to a new behavior:
new run, new reward framing, possibly warm-started from the walking policy.

---

## 4. Summary table

| Goal | Approach | Effort |
|---|---|---|
| New goal position, same task | Change `goal_position`, new run; policy is goal-conditioned (often near-zero-shot + quick refine) | trivial (config) |
| Faster convergence on new goal | Wire warm-start checkpoint into the trainer, fine-tune from trained weights | small code change |
| Curriculum (goal moves farther over iterations) | Warm-start + step the goal each iteration | small code + loop hook |
| Totally new skill | New run, optionally warm-started from the walking policy | medium |

---

## 5. The meta-lessons for the blog

1. **Goal-conditioning is what makes a policy reusable.** Because the goal enters
   through the observation (not baked into the weights as a fixed target), a
   single trained policy generalizes across goal positions and is a natural
   warm-start for new ones. Design the observation for reuse from day one.
2. **"Change the instruction" has two very different meanings.** Moving a
   parameter within the same task (goal position) is a config edit + resume.
   Changing the *objective* is a new experiment — because the feedback history and
   the eval scorecard are both task-specific. Don't conflate them.
3. **Frozen-at-startup config is a feature, not a bug.** Reproducibility depends
   on the instruction and goal being fixed for the duration of a run. Mid-run
   mutation would make a run un-interpretable; the checkpoint-resume boundary is
   the right place to introduce a change.
4. **Re-training ≠ from scratch.** With a goal-conditioned, checkpointed policy,
   "re-train with a new goal" should mean *fine-tune from the learned weights*.
   The only missing piece is a warm-start seam in the trainer — a small, additive
   change, not a redesign.
