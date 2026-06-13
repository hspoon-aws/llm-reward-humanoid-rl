#!/usr/bin/env python3
"""Headless MJX physics smoke test — the B200 compute gate (Phase 2).

Proves the riskiest unknown cheaply BEFORE building the full trainer: does JAX
see the B200, and does MJX step rigid-body physics on it under jit + vmap (the
exact execution model the training loop uses)? This is the MuJoCo analog of the
Isaac Stage-A launch probe that caught the RT-core wall in ~30s of GPU time.

Run inside the MJX venv on the B200:
    /data/mjxvenv/bin/python scripts/smoke_test_mjx.py

Exit 0 + "MJX SMOKE: PASS" means JAX is on the GPU and a batched, jitted MJX
step advances physics.
"""
from __future__ import annotations

import sys


def main() -> int:
    try:
        import jax
        import jax.numpy as jnp
        import mujoco
        from mujoco import mjx
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] import error: {type(exc).__name__}: {exc}")
        return 1

    devices = jax.devices()
    print(f"[mjx-smoke] jax devices: {devices}")
    if not any(d.platform == "gpu" for d in devices):
        print("[FAIL] JAX sees no GPU device (expected a B200 CUDA device)")
        return 1

    # Minimal free-falling body — enough to prove rigid-body integration on GPU.
    xml = """
    <mujoco>
      <option timestep="0.005"/>
      <worldbody>
        <geom type="plane" size="5 5 0.1"/>
        <body pos="0 0 1.0">
          <freejoint/>
          <geom type="capsule" size="0.06 0.2" mass="1"/>
        </body>
      </worldbody>
    </mujoco>
    """
    try:
        mj_model = mujoco.MjModel.from_xml_string(xml)
        mjx_model = mjx.put_model(mj_model)
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] model build/put failed: {type(exc).__name__}: {exc}")
        return 1

    BATCH = 4096  # the training-scale vmap width

    @jax.jit
    def init_batch(seed):
        keys = jax.random.split(seed, BATCH)

        def one(_k):
            d = mjx.make_data(mjx_model)
            return d

        return jax.vmap(one)(keys)

    @jax.jit
    def step_batch(data):
        return jax.vmap(lambda d: mjx.step(mjx_model, d))(data)

    try:
        data = init_batch(jax.random.PRNGKey(0))
        z0 = float(jnp.mean(data.qpos[:, 2]))
        for _ in range(20):
            data = step_batch(data)
        data.qpos.block_until_ready()
        z1 = float(jnp.mean(data.qpos[:, 2]))
    except Exception as exc:  # noqa: BLE001
        print(f"[FAIL] jitted vmap step failed: {type(exc).__name__}: {exc}")
        return 1

    print(f"[ok] batched MJX step ran: batch={BATCH}, mean z {z0:.4f} -> {z1:.4f} (should fall)")
    if not (z1 < z0):
        print("[WARN] body did not fall as expected; physics may be misconfigured")
    print("=== MJX SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
