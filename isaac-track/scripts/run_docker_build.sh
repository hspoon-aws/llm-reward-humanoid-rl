#!/usr/bin/env bash
# Host-side launcher for the Isaac Lab image build. Logs to /var/log/docker-build.log
# with a DOCKER_BUILD_DONE_<rc> marker. Backgroundable + pollable.
set -uo pipefail
LOG=/var/log/docker-build.log
cd /opt/humanoid
: > "${LOG}"
{
  docker build -t humanoid-from-scratch:latest .
  echo "DOCKER_BUILD_DONE_$?"
} >> "${LOG}" 2>&1
