#!/usr/bin/env python3
"""Headless MuJoCo render smoke test — the B200 render-risk gate (Phase 5).

The one MJX surface that touches GL is demo-video rendering (Req 10). MJX
*physics* runs on CUDA compute and is fine on a B200, but MuJoCo's renderer
needs a working headless GL context (EGL or OSMesa). Data-center Blackwell has
no RT cores; this script proves whether the offscreen renderer nonetheless
works on this host BEFORE the loop depends on it — the same cheap ~30s gate
philosophy that caught the Isaac Sim RT-core wall.

Usage (on the GPU host, inside the JAX/MuJoCo container):
    MUJOCO_GL=egl python3 scripts/smoke_test_render.py
    # fallback to try if EGL fails:
    MUJOCO_GL=osmesa python3 scripts/smoke_test_render.py

Exit 0 + "RENDER SMOKE: PASS" means offscreen RGB frames render headless.
"""
from __future__ import annotations

import os
import sys


def main() -> int:
    gl = os.environ.get("MUJOCO_GL", "(unset — set MUJOCO_GL=egl or osmesa)")
    print(f"[render-smoke] MUJOCO_GL={gl}")
    try:
        import mujoco
        import numpy as np
    except ImportError as exc:
        print(f"[FAIL] mujoco not importable: {exc}")
        return 1

    # Minimal model: a single box on a plane is enough to prove the GL context.
    xml = """
    <mujoco>
      <worldbody>
        <light pos="0 0 3"/>
        <geom type="plane" size="2 2 0.1"/>
        <body pos="0 0 0.5"><freejoint/><geom type="box" size="0.2 0.2 0.2"/></body>
      </worldbody>
    </mujoco>
    """
    try:
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        with mujoco.Renderer(model, height=240, width=320) as renderer:
            renderer.update_scene(data)
            frame = renderer.render()
    except Exception as exc:  # noqa: BLE001 - this is the gate; report anything
        print(f"[FAIL] offscreen render failed: {type(exc).__name__}: {exc}")
        print("       Try the other backend (MUJOCO_GL=osmesa) or render eval on CPU.")
        return 1

    if frame is None or np.asarray(frame).size == 0:
        print("[FAIL] renderer returned an empty frame")
        return 1

    print(f"[ok] rendered RGB frame shape={np.asarray(frame).shape}")
    print("=== RENDER SMOKE: PASS ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
