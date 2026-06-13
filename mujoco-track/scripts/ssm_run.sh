#!/usr/bin/env bash
# Run a shell command on the B200 via SSM and print its output.
# Usage: scripts/ssm_run.sh '<command>'  [timeout_seconds]
#   or:  scripts/ssm_run.sh --file <local_script.sh> [timeout_seconds]
# Env: AWS_PROFILE (default your-aws-profile), INSTANCE (default the B200), REGION.
set -euo pipefail

PROFILE="${AWS_PROFILE:-your-aws-profile}"
REGION="${REGION:-us-west-2}"
INSTANCE="${INSTANCE:-i-EXAMPLE0000000001}"
WAIT="${WAIT:-600}"

if [[ "${1:-}" == "--file" ]]; then
  SCRIPT_PATH="${2:?usage: ssm_run.sh --file <script.sh>}"
  CMD="$(cat "$SCRIPT_PATH")"
  WAIT="${3:-$WAIT}"
else
  CMD="${1:?usage: ssm_run.sh '<command>' [timeout_s]}"
  WAIT="${2:-$WAIT}"
fi

# Build the parameters JSON safely with python (handles all quoting/newlines).
PARAMS_FILE="$(mktemp)"
trap 'rm -f "$PARAMS_FILE"' EXIT
CMD="$CMD" WAIT="$WAIT" python3 - "$PARAMS_FILE" <<'PY'
import json, os, sys
params = {"commands": [os.environ["CMD"]], "executionTimeout": [os.environ["WAIT"]]}
with open(sys.argv[1], "w") as fh:
    json.dump(params, fh)
PY

cid=$(AWS_PROFILE="$PROFILE" aws ssm send-command \
  --region "$REGION" \
  --instance-ids "$INSTANCE" \
  --document-name "AWS-RunShellScript" \
  --comment "mujoco-bringup" \
  --parameters "file://$PARAMS_FILE" \
  --query "Command.CommandId" --output text)

for _ in $(seq 1 "$WAIT"); do
  status=$(AWS_PROFILE="$PROFILE" aws ssm get-command-invocation \
    --region "$REGION" --command-id "$cid" --instance-id "$INSTANCE" \
    --query "Status" --output text 2>/dev/null || echo "Pending")
  case "$status" in
    Success|Failed|Cancelled|TimedOut) break ;;
  esac
  sleep 3
done

AWS_PROFILE="$PROFILE" aws ssm get-command-invocation \
  --region "$REGION" --command-id "$cid" --instance-id "$INSTANCE" \
  --query "{status:Status,code:ResponseCode}" --output json
echo "----- STDOUT -----"
AWS_PROFILE="$PROFILE" aws ssm get-command-invocation \
  --region "$REGION" --command-id "$cid" --instance-id "$INSTANCE" \
  --query "StandardOutputContent" --output text
echo "----- STDERR -----"
AWS_PROFILE="$PROFILE" aws ssm get-command-invocation \
  --region "$REGION" --command-id "$cid" --instance-id "$INSTANCE" \
  --query "StandardErrorContent" --output text
