"""MJX policy rollout + headless demo recorder for the Evaluator (Phase 5).

The Evaluator's metric math is framework-agnostic and consumes
:class:`EpisodeTrajectory` (host-side arrays). This module is the GPU-side
adapter that produces those trajectories from a trained Brax/MJX policy, and
records best/worst demo video via MuJoCo's headless ``Renderer`` (EGL on the
B200, verified in scripts/smoke_test_render.py).

DI seam: every JAX/MuJoCo import is lazy and inside methods, so importing this
module on the controller host pulls in no GPU stack — but in practice the
Evaluator only imports it via ``build_mjx_*`` on the GPU box.
"""

from __future__ import annotations

from typing import Any

from ..data_models import Goal
from .evaluator import EpisodeRollout, EpisodeTrajectory


class MjxPolicyRollout:  # pragma: no cover - requires the GPU runtime
    """Rolls a trained policy out in the goal-conditioned MJX env.

    Produces one :class:`EpisodeRollout` per requested episode, each carrying an
    :class:`EpisodeTrajectory` of host-side arrays (base xy, upright flags,
    torques, rewards, foot contacts) the Evaluator's pure metric math consumes.
    """

    def __init__(self, goal: Goal, eval_config: Any) -> None:
        self._goal = goal
        self._config = eval_config
        self._env_name = getattr(eval_config, "env_name", "H1JoystickGaitTracking")
        self._fall_threshold_s = float(getattr(eval_config, "fall_threshold_s", 3.0))

    def rollout(self, checkpoint: Any, goal: Goal, num_episodes: int):
        import jax
        import jax.numpy as jnp
        import numpy as np

        from ..envs.goal_env import build_goal_env

        env = build_goal_env(self._env_name, goal)
        policy = _load_policy(checkpoint, env)

        reset = jax.jit(jax.vmap(env.reset))
        step = jax.jit(jax.vmap(env.step))

        n = int(num_episodes)
        keys = jax.random.split(jax.random.PRNGKey(0), n)
        state = reset(keys)

        episode_length = int(getattr(self._config, "episode_length", 500))
        dt = float(getattr(env, "dt", 0.02))

        # Per-episode time series collected on host.
        xs: list[list[tuple[float, float]]] = [[] for _ in range(n)]
        upright: list[list[bool]] = [[] for _ in range(n)]
        torques: list[list[list[float]]] = [[] for _ in range(n)]
        rewards: list[list[float]] = [[] for _ in range(n)]

        for _t in range(episode_length):
            action = policy(state.obs)
            state = step(state, action)
            qpos = np.asarray(state.data.qpos)        # (n, nq)
            af = np.asarray(state.data.actuator_force)  # (n, nu)
            rew = np.asarray(state.reward)            # (n,)
            base_z = qpos[:, 2]
            qw = qpos[:, 3]
            up = (2.0 * qw * qw - 1.0 > 0.0) & (base_z > 0.5)
            for i in range(n):
                xs[i].append((float(qpos[i, 0]), float(qpos[i, 1])))
                upright[i].append(bool(up[i]))
                torques[i].append([float(v) for v in af[i]])
                rewards[i].append(float(rew[i]))

        out = []
        for i in range(n):
            traj = EpisodeTrajectory(
                positions_xy=xs[i],
                upright_flags=upright[i],
                dt=dt,
                torques=torques[i],
                rewards=rewards[i],
            )
            out.append(EpisodeRollout(
                trajectory=traj,
                replay={"episode": i, "goal": goal, "checkpoint": checkpoint},
            ))
        return out


class MjxDemoVideoRecorder:  # pragma: no cover - requires the GPU runtime
    """Re-renders one episode to an RGB video via MuJoCo's headless Renderer.

    Uses ``MUJOCO_GL=egl`` (verified on the B200). Records the demo cameras
    (chase + side) the Evaluator's CameraConfig supplies at a small env count;
    never instantiated during training and never a policy input (Req 10.4)."""

    def __init__(self, goal: Goal, eval_config: Any) -> None:
        self._goal = goal
        self._config = eval_config
        self._env_name = getattr(eval_config, "env_name", "H1JoystickGaitTracking")

    def record(self, *, rollout, cameras, label: str, output_dir: str):
        """Render the selected episode through the demo cameras (Req 10.3).

        Matches the ``DemoVideoProducer`` contract:
        ``record(rollout=, cameras=, label=, output_dir=) -> list[str]`` of the
        written video file paths (one per camera). Renders frames with MuJoCo's
        headless EGL Renderer, then encodes each video in a SEPARATE subprocess
        (``src.eval.encode_video``) so ffmpeg is never forked from inside the
        live JAX process — avoiding the os.fork()/JAX-threadpool deadlock
        (lesson §6.2). Re-rolls a single env to reproduce the episode."""
        import os
        import subprocess
        import sys
        import tempfile

        import jax
        import mujoco
        import numpy as np

        from ..envs.goal_env import build_goal_env

        os.makedirs(output_dir, exist_ok=True)
        goal = self._goal
        env = build_goal_env(self._env_name, goal)
        replay = getattr(rollout, "replay", {}) or {}
        episode = int(replay.get("episode", 0)) if isinstance(replay, dict) else 0
        ckpt = replay.get("checkpoint") if isinstance(replay, dict) else None
        policy = _load_policy(ckpt, env)

        reset = jax.jit(env.reset)
        step = jax.jit(env.step)
        mj_model = env.mj_model
        episode_length = int(getattr(self._config, "episode_length", 300))

        out_paths = []
        cam_list = cameras if isinstance(cameras, (list, tuple)) else [cameras]
        for cam_spec in cam_list:
            state = reset(jax.random.PRNGKey(episode))
            mj_data = mujoco.MjData(mj_model)
            renderer = mujoco.Renderer(
                mj_model,
                height=int(getattr(cam_spec, "height", 480)),
                width=int(getattr(cam_spec, "width", 640)),
            )
            cam = cam_spec.to_mujoco_camera()
            frames = []
            for _t in range(episode_length):
                action = policy(state.obs)
                state = step(state, action)
                mj_data.qpos[:] = np.asarray(state.data.qpos)
                mj_data.qvel[:] = np.asarray(state.data.qvel)
                mujoco.mj_forward(mj_model, mj_data)
                renderer.update_scene(mj_data, camera=cam)
                frames.append(renderer.render())
            renderer.close()

            fps = int(getattr(cam_spec, "fps", 30))
            name = getattr(cam_spec, "name", "cam")
            path = os.path.join(output_dir, f"{label}_{name}.mp4")
            # Hand frames to a separate-process encoder (fork+exec, no JAX).
            arr = np.asarray(frames, dtype=np.uint8)
            with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as fh:
                npy_path = fh.name
            np.save(npy_path, arr)
            try:
                proc = subprocess.run(
                    [sys.executable, "-m", "src.eval.encode_video", npy_path, path, str(fps)],
                    capture_output=True, text=True, timeout=300,
                    cwd=os.environ.get("PROJECT_ROOT", "/data/mujoco"),
                )
                if proc.returncode != 0 or not os.path.exists(path):
                    # Encoding is additive (Req 10): never fail the eval over a video.
                    continue
                out_paths.append(path)
            finally:
                try:
                    os.unlink(npy_path)
                except OSError:
                    pass
        return out_paths


def _load_policy(checkpoint: Any, env: Any):  # pragma: no cover - GPU host only
    """Rebuild the trained inference policy from a saved checkpoint.

    Accepts a directly-callable policy (test/eval injection) as-is. Otherwise
    reconstructs the Brax PPO policy from the params saved by
    ``_MjxTrainer.save_checkpoint`` (``brax.io.model.save_params``) plus the
    sidecar ``<path>.netcfg.json`` describing the network architecture, and
    returns a deterministic ``policy(obs) -> action`` callable via
    ``ppo_networks.make_inference_fn``. Falls back to a zero-action policy only
    if the checkpoint or its netcfg is missing/unreadable, so the rollout still
    runs rather than crashing."""
    if callable(checkpoint):
        return checkpoint

    import json
    import os

    import jax
    import jax.numpy as jnp

    path = checkpoint if isinstance(checkpoint, str) else getattr(checkpoint, "path", None)
    action_size = int(getattr(env, "action_size", 19))

    def _zero_policy(obs):
        lead = obs.shape[:-1] if hasattr(obs, "shape") else ()
        return jnp.zeros(tuple(lead) + (action_size,))

    if not path or not os.path.exists(path):
        return _zero_policy

    netcfg_path = path + ".netcfg.json"
    if not os.path.exists(netcfg_path):
        return _zero_policy

    try:
        from brax.io import model as bx_model
        from brax.training.acme import running_statistics, specs
        from brax.training.agents.ppo import networks as ppo_networks

        with open(netcfg_path) as fh:
            netcfg = json.load(fh)
        params = bx_model.load_params(path)

        obs_size = int(netcfg["observation_size"])
        act_size = int(netcfg["action_size"])
        normalize = bool(netcfg.get("normalize_observations", True))
        preprocess = (
            running_statistics.normalize if normalize else (lambda x, y: x)
        )
        ppo_nets = ppo_networks.make_ppo_networks(
            observation_size=obs_size,
            action_size=act_size,
            preprocess_observations_fn=preprocess,
            policy_hidden_layer_sizes=tuple(netcfg.get("policy_hidden_layer_sizes", (128,) * 4)),
            value_hidden_layer_sizes=tuple(netcfg.get("value_hidden_layer_sizes", (256,) * 5)),
        )
        make_inference = ppo_networks.make_inference_fn(ppo_nets)
        inference = make_inference(params, deterministic=True)

        def _policy(obs):
            # Brax inference_fn expects (obs, rng) and returns (action, extras).
            act, _ = inference(obs, jax.random.PRNGKey(0))
            return act

        return _policy
    except Exception:  # noqa: BLE001 - any reconstruction failure -> safe fallback
        return _zero_policy
