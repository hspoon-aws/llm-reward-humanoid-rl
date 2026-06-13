#!/usr/bin/env python3
"""Stage A bring-up check — run INSIDE the Isaac Lab container.

Verifies, cheaply and without training:
  1. the Isaac Sim SimulationApp launches headless (AppLauncher)
  2. isaaclab / isaaclab_tasks import — ONLY valid AFTER the app launches,
     because the task registry pulls in omni.physics which the SimulationApp
     provides (importing isaaclab_tasks cold fails with
     "No module named 'omni.physics'")
  3. the H1 velocity task id is registered (the corrected, vendor-less name)

Prints a STAGE_A: PASS/FAIL line; exit 0 on pass.
"""
from __future__ import annotations

import sys

EXPECTED = "Isaac-Velocity-Flat-H1-v0"


def main() -> int:
    # (1) Launch the simulator headless BEFORE importing any task module.
    try:
        from isaaclab.app import AppLauncher
    except Exception as e:  # noqa: BLE001
        print(f"STAGE_A: FAIL import isaaclab.app.AppLauncher: {e}")
        return 1
    try:
        app_launcher = AppLauncher(headless=True)
        sim_app = app_launcher.app
    except Exception as e:  # noqa: BLE001
        print(f"STAGE_A: FAIL AppLauncher(headless=True): {e}")
        return 1

    # (2) Now the registry-bearing imports are valid (omni.physics exists).
    try:
        import gymnasium as gym
        import isaaclab  # noqa: F401
        import isaaclab_tasks  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print(f"STAGE_A: FAIL import isaaclab_tasks after app launch: {e}")
        try:
            sim_app.close()
        except Exception:
            pass
        return 1

    ids = sorted(e for e in gym.envs.registry if "H1" in e)
    print("H1_IDS:", ids)
    has_expected = EXPECTED in gym.envs.registry
    print(f"HAS_EXPECTED[{EXPECTED}]:", has_expected)

    rc = 0 if has_expected else 1
    print("STAGE_A: PASS" if rc == 0 else
          "STAGE_A: FAIL expected H1 id not registered; set training.task to one of H1_IDS")

    try:
        sim_app.close()
    except Exception:
        pass
    return rc


if __name__ == "__main__":
    sys.exit(main())
