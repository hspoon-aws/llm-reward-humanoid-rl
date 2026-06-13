"""Quick Bedrock connectivity check: invoke Claude Opus 4.8 via Converse."""
import sys
import boto3

region = "us-west-2"
model_id = "global.anthropic.claude-opus-4-8"
client = boto3.client("bedrock-runtime", region_name=region)
try:
    resp = client.converse(
        modelId=model_id,
        messages=[{"role": "user", "content": [{"text": "Reply with the word OK only."}]}],
        inferenceConfig={"maxTokens": 16},
    )
    text = "".join(b.get("text", "") for b in resp["output"]["message"]["content"])
    print("BEDROCK_OK:", repr(text.strip()))
    sys.exit(0)
except Exception as exc:  # noqa: BLE001
    print("BEDROCK_FAIL:", type(exc).__name__, str(exc)[:300])
    sys.exit(1)
