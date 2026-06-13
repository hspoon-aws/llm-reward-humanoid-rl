#!/usr/bin/env bash
# Push a local file to the B200 via SSM by base64-encoding it into a RunShellScript.
# Usage: scripts/ssm_push.sh <local_path> <remote_path>
set -euo pipefail

LOCAL="${1:?usage: ssm_push.sh <local> <remote>}"
REMOTE="${2:?usage: ssm_push.sh <local> <remote>}"

B64="$(base64 < "$LOCAL" | tr -d '\n')"
# Write the decoder command; base64 content is safe (no shell metachars).
CMD="mkdir -p \"\$(dirname '$REMOTE')\" && echo '$B64' | base64 -d > '$REMOTE' && wc -c '$REMOTE'"
WAIT=120 "$(dirname "$0")/ssm_run.sh" "$CMD" 120
