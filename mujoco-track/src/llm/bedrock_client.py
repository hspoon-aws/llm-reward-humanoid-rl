"""BedrockClient: a drop-in alternative to QwenClient that generates / refines
reward functions via Amazon Bedrock (Anthropic Claude) instead of the
self-hosted vLLM Qwen endpoint.

Why this exists
---------------
The self-hosted Qwen3-Coder runs are bottlenecked less by the LLM's raw
reasoning than by feedback richness + search budget (see
docs/lesson-llm-reward-design-bottleneck). To test whether a stronger model
helps the reward-refinement loop, this client swaps ONLY the transport: it
subclasses :class:`QwenClient` and overrides ``_chat`` to call Bedrock's
Messages API. All the framework-agnostic plumbing — prompt-template loading,
``compute_reward`` extraction, ``ast.parse`` validation, the refine-vs-initial
prompt selection — is inherited unchanged, so the reward contract the rest of
the loop depends on is identical.

Bedrock specifics
-----------------
* Claude Opus 4.x is INFERENCE_PROFILE-only, so ``model`` must be an inference
  profile id (e.g. ``us.anthropic.claude-opus-4-8``), not the bare foundation
  model id.
* Uses the Anthropic Messages API shape (``anthropic_version``, ``messages``,
  ``max_tokens``) via ``bedrock-runtime.invoke_model``.
* Auth/region come from the standard boto3 chain (instance role / env / profile);
  the instance role must allow ``bedrock:InvokeModel`` on the profile + the
  underlying foundation model.

No JAX / MJX dependency, so it runs and unit-tests on the controller host.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .qwen_client import (
    QwenClient,
    QwenClientConfig,
    RequestError,
    ServiceUnavailableError,
)

_ANTHROPIC_VERSION = "bedrock-2023-05-31"


@dataclass
class BedrockClientConfig(QwenClientConfig):
    """Bedrock-flavored config. Inherits the Qwen knobs (prompts_dir, retries,
    temperature, max_tokens, ...) and adds Bedrock routing.

    ``model`` is reused as the Bedrock inference-profile id (so callers and the
    base class treat it uniformly as "the model name")."""

    region: str = "us-west-2"
    # Default to the cross-region inference profile for Claude Opus 4.8.
    model: str = "us.anthropic.claude-opus-4-8"
    # Newer Claude models deprecate `temperature` in the invoke body and reject
    # it. Default False (omit it). Set True only for older models that accept it.
    send_temperature: bool = False


class BedrockClient(QwenClient):
    """QwenClient-compatible client backed by Amazon Bedrock (Claude)."""

    def __init__(
        self,
        config: Optional[BedrockClientConfig] = None,
        *,
        client: Any = None,
    ):
        # QwenClient.__init__ loads the prompt templates from config.prompts_dir.
        super().__init__(config or BedrockClientConfig())
        # Injectable for unit tests; lazily built otherwise so importing this
        # module never requires boto3/credentials.
        self._client = client

    # ----------------------------- transport ------------------------------ #
    def _bedrock(self):
        if self._client is None:
            import boto3  # local import: only needed when no client is injected

            self._client = boto3.client(
                "bedrock-runtime", region_name=getattr(self.config, "region", "us-west-2")
            )
        return self._client

    def _chat(self, prompt: str) -> str:
        """Single-turn completion via Bedrock Messages API, retrying transient
        failures. Mirrors QwenClient._chat's error contract so the Orchestrator's
        recovery edges (skip on RequestError, wait/resume on
        ServiceUnavailableError) behave identically.

        Honors the same cross-process request lock as QwenClient (a no-op unless
        ``request_lock_path`` is set), so a Bedrock run can still serialize
        against other loop processes if desired (rarely needed — Bedrock is a
        managed, concurrent endpoint, unlike the single self-hosted vLLM)."""
        with self._request_lock():
            return self._invoke_with_retries(prompt)

    def _invoke_with_retries(self, prompt: str) -> str:
        body = {
            "anthropic_version": _ANTHROPIC_VERSION,
            "max_tokens": int(self.config.max_tokens),
            "messages": [{"role": "user", "content": prompt}],
        }
        # Newer Claude models (e.g. Opus 4.8) DEPRECATE `temperature` in the
        # Bedrock invoke body and reject the request if it is present. Only send
        # it when it differs from the model default AND the model accepts it; the
        # safe, broadly-compatible choice is to omit it (let the model use its
        # default sampling). Kept configurable via ``send_temperature`` for older
        # models that still honor it.
        if getattr(self.config, "send_temperature", False):
            body["temperature"] = float(self.config.temperature)
        payload = json.dumps(body)
        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                client = self._bedrock()
                resp = client.invoke_model(modelId=self.config.model, body=payload)
                data = json.loads(resp["body"].read())
                return self._extract_text(data)
            except Exception as exc:  # noqa: BLE001 - classify below
                name = type(exc).__name__
                # Connection / endpoint-reachability problems -> let the
                # Orchestrator wait for the endpoint and resume (Req 16).
                if name in ("EndpointConnectionError", "ConnectTimeoutError"):
                    raise ServiceUnavailableError(
                        f"Bedrock endpoint unreachable: {exc}"
                    ) from exc
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_s * attempt)

        raise RequestError(
            f"Bedrock request failed after {self.config.max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _extract_text(data: dict) -> str:
        """Pull the assistant text out of a Bedrock Anthropic Messages response.

        Shape: ``{"content": [{"type": "text", "text": "..."}, ...], ...}``.
        Concatenates all text blocks so a multi-block reply is preserved."""
        content = data.get("content")
        if isinstance(content, list):
            parts = [
                block.get("text", "")
                for block in content
                if isinstance(block, dict) and block.get("type", "text") == "text"
            ]
            text = "".join(parts).strip()
            if text:
                return text
        raise RequestError(
            f"Bedrock response had no text content: keys={list(data.keys())}"
        )
