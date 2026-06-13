# Lesson: bringing up MuJoCo MJX on the B200 (the two gates Isaac failed)

**Date:** 2026-06-12 (MuJoCo track bring-up, live on `i-EXAMPLE0000000001`)
**Where it came from:** standing up the MJX/Brax/JAX stack on the `p6-b200.48xlarge`
via SSM `AWS-RunShellScript`, after Isaac Sim was ruled out on this hardware.
**Blog section:** "Running MuJoCo MJX Simulation" → B200 bring-up + the Isaac-vs-MuJoCo
hardware story.

## TL;DR

The two risks that sank Isaac on the B200 — GPU compute compatibility and headless
rendering without RT cores — **both pass cleanly on MJX**, verified in ~minutes of GPU
time with cheap smoke gates before writing any trainer code. The one real friction was a
**version matrix pinch** in the MuJoCo Playground stack, resolved by pinning
`mujoco==3.4.0` / `mujoco-mjx==3.4.0`.

## What passed (the decisive results)

1. **MJX physics on the B200 (the gate Isaac Sim failed).** JAX sees all 8 B200s as
   `CudaDevice`, and a **4096-env vmap'd + jitted `mjx.step`** integrates rigid-body
   physics correctly (a free body falls 1.00→0.95 m over 20 steps). This is the exact
   execution model the training loop uses, and it runs on pure CUDA compute — no RT cores,
   the thing Isaac Sim's RTX renderer required and the B200 lacks.
   `scripts/smoke_test_mjx.py` → `MJX SMOKE: PASS`.

2. **Headless rendering with EGL (Req 10 risk).** `mujoco.Renderer` produces RGB frames
   headless via `MUJOCO_GL=egl` (libEGL_nvidia is present on the DLAMI). A 240×320×3 frame
   rendered with no display and no RT cores. `scripts/smoke_test_render.py` →
   `RENDER SMOKE: PASS`. **OSMesa is NOT installed** (PyOpenGL import fails) — EGL is the
   path; don't rely on the OSMesa fallback here.

3. **H1 env loads and steps.** `H1JoystickGaitTracking` loads from the Playground registry
   (menagerie auto-downloads on first use) and `env.step` runs. Real surface captured below.

## The version-matrix pinch (the one real friction)

PyPI `playground==0.1.0` (the only installable version on Python 3.10) calls
`mjx.make_data(..., nconmax=...)`, an argument **removed in mujoco-mjx 3.9**. But the
menagerie H1 model XML uses a `<contact>` schema element **rejected by mujoco 3.2.7**. So:

| mujoco / mjx | result |
|---|---|
| 3.9.0 (pip default) | `TypeError: make_data() got an unexpected keyword argument 'nconmax'` |
| 3.2.7 | `XML Error: Schema violation: unrecognized element 'contact'` |
| **3.4.0** | **works** ✓ |
| git-main playground | needs Python ≥ 3.11 (host has 3.10) — not worth a Python upgrade |

> **Lesson:** the Playground PyPI release lags the MuJoCo release line. On Python 3.10,
> pin the whole trio to the **3.4.x** era (`mujoco==3.4.0`, `mujoco-mjx==3.4.0`,
> `playground==0.1.0`) rather than taking `pip install mujoco` (which pulls 3.9 and breaks
> on `nconmax`). If you want git-main playground + mujoco 3.9, install Python 3.11 first.

## The real H1 surface (write the wrapper against THIS, not assumptions)

From `scripts/probe_h1_env.py` on the box (`H1JoystickGaitTracking`):

- **observation**: flat `jax.Array` shape `(113,)` float32 — NOT a dict. (The prompt/
  obs-space text must describe a 113-dim proprio+gait vector, then + our Goal_Observation.)
- **action**: `(19,)` joint-position targets. **dt = 0.02 s**.
- **`state.data`** (the `mjx.Data` the reward reads) exposes exactly the accessors the
  prompt promises:
  - `qpos` `(26,)` = 7 free-joint (3 pos + 4 quat) + 19 joints
  - `qvel` `(25,)` = 6 base (3 lin + 3 ang) + 19 joint
  - `xpos` `(21, 3)`, `actuator_force` `(19,)`, `cfrc_ext` `(21, 6)`
- **stock reward terms** (zero these for goal-reaching, Req 6): `tracking_lin_vel`,
  `tracking_ang_vel`, `lin_vel_z`, `ang_vel_xy`, `feet_air_time`, `feet_phase`, `foot_slip`,
  `pose`, `action_rate`.

Note the prompt's `qpos`/`qvel` index comments were already right (base-free-joint-first
layout); `actuator_force`/`cfrc_ext` confirmed present. The only correction vs the initial
prompt draft: obs is **113-dim flat**, not the ~69-dim Isaac layout — update the obs-space
description accordingly.

## Reproducible setup (captured in `scripts/b200_setup_mjx.sh`)

1. `apt-get install -y python3.10-venv` (DLAMI lacks it; needed for `python3 -m venv`).
2. venv at `/data/mjxvenv` (isolated from the system Python that serves vLLM on GPU 0).
3. `pip install "jax[cuda12]" "mujoco==3.4.0" "mujoco-mjx==3.4.0" brax playground`
   (CUDA-12 JAX wheels run fine on the CUDA-13 / 580.x driver — forward compatible).
4. Always run rendering with `MUJOCO_GL=egl`.

`warp` import warnings from MJX are harmless (optional collision backend; the default works).

## Follow-ups

- [ ] Pin `mujoco==3.4.0` / `mujoco-mjx==3.4.0` in `pyproject.toml`'s `gpu` extra and the
      Dockerfile so the matrix pinch can't recur.
- [ ] Update `prompts/*.txt` + `DEFAULT_OBS_SPACE_DESCRIPTION`: obs is 113-dim flat.
- [ ] Reference this lesson in the Blog (Req 21.5) as the MuJoCo half of the
      Isaac/RTX-vs-MuJoCo/B200 hardware comparison.

## Update — Phases 4 & 5 validated live (training + eval + render)

After the bring-up gates, the full pipeline ran end-to-end on the B200 over SSM:

- **Brax PPO trains the goal env** (`scripts/smoke_test_train.py`): generated JAX reward →
  goal env → `brax.training.agents.ppo.train` with `wrap_for_brax_training` → eval metrics
  carrying the `goal/*` components → 1.2 MB policy checkpoint. ~137 s for a tiny budget on one
  B200, JAX pinned to GPU 1 (`CUDA_VISIBLE_DEVICES=1 XLA_PYTHON_CLIENT_MEM_FRACTION=0.6`) so
  vLLM keeps GPU 0.
- **Evaluator rollout + headless demo render** (`scripts/smoke_test_eval.py`): MJX rollout →
  `EpisodeTrajectory` → framework-agnostic `compute_eval_metrics` (success_rate, distance,
  upright, fall_rate + all 3 staged gates) → JSON → a real `demo.mp4` rendered via the EGL
  `mujoco.Renderer`.

### Two Brax-specific bugs the live train gate caught (both fixed)

1. **`lax.scan` carry-structure invariant.** Brax's training wrapper scans the env step, which
   requires `state.metrics` to have *identical pytree structure* at reset and at step. Our goal
   reward adds `goal/*` metric keys in `step`; `reset` didn't have them → `TypeError: scan body
   carry input and carry output must have the same pytree structure ... symmetric difference
   {goal/...}`. **Fix:** seed the `goal/*` keys in `reset` by computing the goal reward once on
   the reset data (same keys + dtypes). Lesson: any env that adds metrics in `step` must add the
   same keys in `reset` for Brax.
2. **Unpicklable policy.** Brax returns `make_inference_fn`, a closure that `pickle` can't
   serialize (`Can't pickle local object 'make_inference_fn.<locals>.make_policy'`). **Fix:**
   `save_checkpoint` persists the policy *params* pytree (as numpy leaves), not the function;
   rebuild the inference fn from params + the network factory at eval time.

### Render note

`imageio.mimwrite` spawns ffmpeg via `os.fork()`, which warns under JAX's threadpool
(`os.fork() was called ... JAX is multithreaded ... may deadlock`). Harmless in the smoke, but
for the unattended run prefer writing frames with a fork-free encoder (e.g. `imageio` with the
`pyav`/`ffmpeg` plugin in-process) or render+encode in a subprocess started before JAX init.

### Follow-up

- [ ] Full policy reconstruction in `mjx_rollout._load_policy` (currently a zero-action fallback
      that lets the rollout/metric/render path run; wire brax `ppo_networks.make_inference_fn`
      with the saved params for a real trained-policy eval).
