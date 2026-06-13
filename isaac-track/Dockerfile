# syntax=docker/dockerfile:1
#
# Humanoid-from-scratch runtime image.
#
# Base: official NVIDIA Isaac Lab image (bundles Isaac Sim + Isaac Lab +
# RSL-RL). On top of it we add vLLM for the self-hosted Qwen3-Coder model and
# this project's orchestration code.
#
# Target host: EC2 p6-b200.48xlarge (8x B200, sm_100 / Blackwell).
#   - vLLM pinned to a Blackwell-aware build; the base image already ships a
#     CUDA 12.8+ / PyTorch cu12x stack required for sm_100.
#   - vLLM serves on GPU 0; Isaac Lab PPO uses GPUs 1-7.
#
# Build (on the GPU host, which can see nvcr.io):
#   docker build -t humanoid-from-scratch:latest .
#
# NOTE: model weights are NOT baked into the image. They are hydrated at
# runtime from S3/EBS into /data/models/qwen3-coder-30b (see launch_vllm.sh).

ARG ISAACLAB_IMAGE=nvcr.io/nvidia/isaac-lab
ARG ISAACLAB_TAG=2.2.0
FROM ${ISAACLAB_IMAGE}:${ISAACLAB_TAG}

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HUMANOID_HOME=/opt/humanoid-from-scratch

# Isaac Lab images ship their own Python (the isaac-sim interpreter). Use it.
# `isaaclab.sh -p` proxies to that interpreter; for plain pip we resolve it.
SHELL ["/bin/bash", "-c"]

# --- system deps for video encoding (demo montage) and S3 sync ------------- #
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ffmpeg \
        awscli \
        tini \
    && rm -rf /var/lib/apt/lists/*

# --- python deps: vLLM (Blackwell build) + client/runtime libs ------------- #
# vLLM 0.11+ carries sm_100 kernels. Pin to a known-good Blackwell release and
# let it pull a compatible torch; --no-build-isolation avoids recompiling the
# CUDA stack already present in the base image.
RUN ${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh -p -m pip install \
        "vllm==0.11.0" \
        "huggingface_hub>=0.34" \
        "requests>=2.31" \
        "boto3>=1.34" \
        "tensorboard" \
    || python -m pip install \
        "vllm==0.11.0" \
        "huggingface_hub>=0.34" \
        "requests>=2.31" \
        "boto3>=1.34" \
        "tensorboard"

# --- CRITICAL: restore Isaac-Sim-compatible dep versions ------------------ #
# The vLLM install above upgrades several deps that Isaac Sim 5.0's bundled
# code is NOT compatible with, each causing a hard failure before any task runs:
#   * NumPy -> 2.x: pinocchio_pywrap/eigenpy (via isaaclab dex_retargeting) are
#     built for NumPy 1.x and SEGFAULT at import under 2.x.
#   * typing_extensions -> 4.12+: it MONKEYPATCHES stdlib typing._collect_parameters
#     to enforce PEP-696 default ordering, which isaaclab_tasks' generics violate
#     -> "Type parameter ~_T1 without a default follows type parameter with a
#     default" when the task registry imports, killing parse_env_cfg/gym.make.
#     Python 3.11's OWN _collect_parameters does NOT enforce this; pin
#     typing_extensions to 4.11.0 (last release before the backported patch) so
#     the stdlib behavior is used. (4.12.2 was tested and still fails.)
# Pin both back to Isaac-Sim-compatible versions in the Isaac Sim python.
# (Verified: numpy fix -> Stage A passes. typing_extensions==4.11.0 is the
# reasoned fix for the registry import crash; pending GPU re-validation.)
# See docs/lesson-isaac-lab-bringup.md.
RUN ${ISAACLAB_PATH:-/workspace/isaaclab}/isaaclab.sh -p -m pip install \
        "numpy==1.26.4" "typing_extensions==4.11.0" \
    || /workspace/isaaclab/_isaac_sim/python.sh -m pip install \
        "numpy==1.26.4" "typing_extensions==4.11.0" \
    || python -m pip install "numpy==1.26.4" "typing_extensions==4.11.0"

# --- project code ---------------------------------------------------------- #
WORKDIR ${HUMANOID_HOME}
COPY src/      ${HUMANOID_HOME}/src/
COPY prompts/  ${HUMANOID_HOME}/prompts/
COPY scripts/  ${HUMANOID_HOME}/scripts/
COPY config/   ${HUMANOID_HOME}/config/
RUN chmod +x ${HUMANOID_HOME}/scripts/*.sh || true

ENV PYTHONPATH=${HUMANOID_HOME}:${PYTHONPATH}

# tini reaps the vLLM + training subprocesses cleanly.
ENTRYPOINT ["/usr/bin/tini", "--"]
CMD ["bash"]
