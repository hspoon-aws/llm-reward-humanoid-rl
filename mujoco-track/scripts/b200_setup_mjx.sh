#!/usr/bin/env bash
# One-time MJX/Brax/JAX environment setup on the B200, in an isolated venv at
# /data/mjxvenv so it never disturbs the system Python that serves vLLM.
set -uo pipefail

VENV=/data/mjxvenv
echo "=== creating venv at $VENV ==="
python3 -m venv "$VENV" 2>&1 | tail -2 || { echo "venv create failed"; exit 1; }
. "$VENV/bin/activate"
python -m pip install -q --upgrade pip 2>&1 | tail -2

echo "=== installing jax[cuda12] + mujoco + mjx + brax + playground ==="
# CUDA-12 JAX wheels run on the CUDA-13 driver (forward compatible).
# Pin mujoco/mjx to 3.4.0: playground 0.1.0 (the PyPI release) still calls
# mjx.make_data(nconmax=...) which mujoco-mjx 3.9 removed, while the menagerie
# H1 XML needs >= 3.4. See docs/lesson-mjx-b200-bringup.md.
pip install -q "jax[cuda12]" "mujoco==3.4.0" "mujoco-mjx==3.4.0" brax playground 2>&1 | tail -15

echo "=== installed versions ==="
python - <<'PY'
import importlib.metadata as m
for pkg in ("jax", "jaxlib", "mujoco", "mujoco-mjx", "brax", "playground", "flax", "optax"):
    try:
        print(f"{pkg}=={m.version(pkg)}")
    except Exception as e:
        print(f"{pkg}: MISSING ({e})")
PY
echo "=== DONE setup. Always run rendering with MUJOCO_GL=egl ==="
