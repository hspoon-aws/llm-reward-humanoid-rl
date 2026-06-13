"""Qwen_Client: talk to the self-hosted Qwen3-Coder model over an
OpenAI-compatible HTTP API to generate / refine reward functions and to
analyze training failures.

Implements requirements 1, 2, 3, and 16 of the humanoid-locomotion spec:
  - generate reward code from a task + observation-space description
  - include Metrics_History when refining
  - extract and ast.parse-validate the compute_reward definition
  - retry transient request failures, raise a typed error when exhausted
  - distinguish "endpoint unreachable" (service unavailable) so the
    Orchestrator can wait and resume
  - load all prompts from external template files

This module deliberately has no Isaac Lab / torch dependency so it can run
and be unit-tested on the controller host without GPUs.
"""

from __future__ import annotations

import ast
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
# These keep their established public names (the offline tests and the
# orchestrator import them from here), but each is aligned with the shared
# error taxonomy in ``src/exceptions.py`` so that code catching a shared type
# (e.g. ValidationError) also catches the Qwen_Client variant. The shared
# module is the canonical home; these are thin, Qwen-scoped subclasses.
from ..exceptions import (
    HumanoidRLError as _HumanoidRLError,
    ExtractionError as _SharedExtractionError,
    RequestError as _SharedRequestError,
    ServiceUnavailableError as _SharedServiceUnavailableError,
    TemplateError as _SharedTemplateError,
    ValidationError as _SharedValidationError,
)


class QwenClientError(_HumanoidRLError):
    """Base class for all Qwen_Client errors (part of the shared taxonomy)."""


class TemplateError(_SharedTemplateError, QwenClientError):
    """A required prompt template is missing or unreadable (Req 3.3)."""


class ExtractionError(_SharedExtractionError, QwenClientError):
    """No compute_reward definition could be extracted (Req 1.4)."""


class CodeValidationError(_SharedValidationError, QwenClientError):
    """Extracted reward code is not parseable Python (Req 1.6).

    Aligned with the shared ``ValidationError`` so the Orchestrator can catch
    the canonical type; the Qwen-scoped name is retained for callers/tests.
    """


class RequestError(_SharedRequestError, QwenClientError):
    """The model request failed after exhausting retries (Req 1.5, 2.3)."""


class ServiceUnavailableError(_SharedServiceUnavailableError, QwenClientError):
    """The model endpoint could not be reached at all (Req 16.1)."""


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class QwenClientConfig:
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3-Coder-30B-A3B-Instruct"
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    max_retries: int = 6
    retry_backoff_s: float = 3.0
    request_timeout_s: float = 180.0
    temperature: float = 0.4
    max_tokens: int = 4096
    # Backend selection. "vllm" (default) talks to the self-hosted Qwen over the
    # OpenAI-compatible HTTP API at ``endpoint``. "bedrock" calls Amazon Bedrock
    # directly (no self-hosted vLLM needed), using the Converse API with
    # ``bedrock_model_id`` in ``bedrock_region``.
    provider: str = "vllm"
    bedrock_model_id: str = "global.anthropic.claude-opus-4-8"
    bedrock_region: str = "us-west-2"


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

# Balance-first guidance injected when an immediate-fall behavior is observed
# (Req 2.2, 13.1). Kept as a constant so the text is consistent across calls and
# composes with any caller-supplied free-text guidance.
_BALANCE_GUIDANCE = (
    "The robot falls almost immediately. Prioritize balance and uprightness: "
    "strengthen the upright/posture and alive terms and de-emphasize "
    "goal-directed speed until the robot can remain standing and make "
    "controlled progress toward the Goal."
)


class QwenClient:
    def __init__(self, config: Optional[QwenClientConfig] = None):
        self.config = config or QwenClientConfig()
        self._templates = self._load_templates(self.config.prompts_dir)

    # ----------------------------- templates ------------------------------ #
    @staticmethod
    def _load_templates(prompts_dir: Path) -> dict[str, str]:
        """Load the three prompt templates up front so a missing file is
        caught before the GPU sprint starts (Req 3.1, 3.3)."""
        names = {
            "initial": "initial_reward.txt",
            "refine": "refine_reward.txt",
            "analyze": "analyze_failure.txt",
        }
        templates: dict[str, str] = {}
        for key, filename in names.items():
            path = Path(prompts_dir) / filename
            try:
                templates[key] = path.read_text(encoding="utf-8")
            except OSError as exc:
                raise TemplateError(
                    f"Could not read required prompt template '{path}': {exc}"
                ) from exc
        return templates

    # ----------------------------- public API ----------------------------- #
    def generate_reward(
        self,
        task_description: str,
        obs_space: str,
        goal_description: str = "",
        metrics_history: Optional[str] = None,
        guidance: Optional[str] = None,
        validation_error: Optional[str] = None,
        runtime_error: Optional[str] = None,
        balance_priority: bool = False,
    ) -> str:
        """Request a reward function. Uses the refine template when any
        feedback (metrics history, guidance, balance priority, a prior
        validation error, or a prior runtime error) is available, otherwise the
        initial template. Returns the extracted, ast-validated source of
        compute_reward (Reqs 1.1-1.6, 2.2, 12.1).

        goal_description carries the point-to-point Goal (target position +
        success radius) and is substituted into the goal-reaching prompts.

        balance_priority injects balance-first guidance (Req 2.2, 13.1) when
        an immediate-fall behavior was observed; it composes with any
        caller-supplied free-text guidance.

        runtime_error carries the message from a reward that validated but
        raised during training (e.g. a tensor-shape mismatch); it is fed back so
        the model corrects the failing body (Req 12.1 spirit)."""
        use_refine = bool(
            metrics_history
            or guidance
            or validation_error
            or runtime_error
            or balance_priority
        )
        if use_refine:
            combined_guidance = "\n".join(
                part
                for part in (
                    (_BALANCE_GUIDANCE if balance_priority else None),
                    guidance,
                    (
                        f"The previous candidate was rejected: {validation_error}. "
                        "Return corrected, parseable Python."
                        if validation_error
                        else None
                    ),
                    (
                        "The previous reward validated but raised during "
                        f"training: {runtime_error}. A common cause is mixing "
                        "tensor shapes (e.g. subtracting the 2-D goal xy from "
                        "the 3-D root_pos_w, or indexing obs out of range). "
                        "Slice tensors to matching shapes (use root_pos_w[:, "
                        ":2] for ground-plane xy) and return per-env tensors of "
                        "shape (num_envs,). Return corrected Python."
                        if runtime_error
                        else None
                    ),
                )
                if part
            )
            prompt = self._templates["refine"].format(
                task_description=task_description,
                obs_space=obs_space,
                goal_description=goal_description or "(reach the configured Goal)",
                metrics_history=metrics_history or "(none yet)",
                guidance=combined_guidance or "(none)",
            )
        else:
            prompt = self._templates["initial"].format(
                task_description=task_description,
                obs_space=obs_space,
                goal_description=goal_description or "(reach the configured Goal)",
            )

        response_text = self._chat(prompt)
        code = self._extract_compute_reward(response_text)
        self._validate_python(code)
        return code

    def analyze_failure(self, metrics: str, behavior_description: str) -> str:
        """Ask the model to diagnose a failing policy and return the analysis
        text (Reqs 2.1-2.3). The template itself carries the balance-first
        guidance for immediate falls (Req 2.2)."""
        prompt = self._templates["analyze"].format(
            metrics=metrics,
            behavior_description=behavior_description,
        )
        return self._chat(prompt).strip()

    # ----------------------------- internals ------------------------------ #
    def _chat(self, prompt: str) -> str:
        """Single-turn chat completion, dispatched to the configured backend.

        ``provider == "bedrock"`` calls Amazon Bedrock directly; otherwise the
        default self-hosted vLLM OpenAI-compatible path is used. Both return the
        assistant message text; the prompt-building, code extraction, and
        validation around this call are provider-agnostic."""
        if self.config.provider.strip().lower() == "bedrock":
            return self._chat_bedrock(prompt)
        return self._chat_vllm(prompt)

    def _chat_vllm(self, prompt: str) -> str:
        """POST a single-turn chat completion to vLLM, retrying transient failures.

        Connection-level failures are surfaced as ServiceUnavailableError so
        the Orchestrator can wait for vLLM to come back (Req 16). HTTP / other
        request failures are retried up to max_retries, then raised as
        RequestError (Reqs 1.5, 2.3)."""
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = requests.post(
                    url, json=payload, timeout=self.config.request_timeout_s
                )
                resp.raise_for_status()
                data = resp.json()
                return data["choices"][0]["message"]["content"]
            except requests.ConnectionError as exc:
                # Endpoint unreachable -> let the orchestrator wait & resume.
                raise ServiceUnavailableError(
                    f"Language model endpoint unreachable at {url}: {exc}"
                ) from exc
            except (requests.RequestException, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_s * attempt)

        raise RequestError(
            f"Model request failed after {self.config.max_retries} attempts: {last_exc}"
        )

    def _chat_bedrock(self, prompt: str) -> str:
        """Single-turn completion via the Amazon Bedrock Converse API.

        Uses the provider-uniform Converse shape so the same call works across
        Bedrock models (here, Claude Opus 4.8). Auth/region come from the
        standard AWS credential chain + ``bedrock_region``. Transient throttling
        and 5xx errors are retried up to ``max_retries``; a credentials/endpoint
        failure is surfaced as ServiceUnavailableError so the Orchestrator can
        wait & resume (Req 16), matching the vLLM path's contract."""
        try:
            import boto3  # noqa: PLC0415
            from botocore.config import Config as _BotoConfig  # noqa: PLC0415
            from botocore.exceptions import (  # noqa: PLC0415
                BotoCoreError,
                ClientError,
                EndpointConnectionError,
                NoCredentialsError,
            )
        except ImportError as exc:  # pragma: no cover - boto3 is a runtime dep
            raise ServiceUnavailableError(
                "provider=bedrock requires boto3/botocore to be installed"
            ) from exc

        client = boto3.client(
            "bedrock-runtime",
            region_name=self.config.bedrock_region,
            config=_BotoConfig(
                read_timeout=self.config.request_timeout_s,
                retries={"max_attempts": 1},  # we own the retry loop below
            ),
        )
        request = dict(
            modelId=self.config.bedrock_model_id,
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"maxTokens": self.config.max_tokens},
        )
        # Newer Claude models (Opus 4.8+) deprecate the ``temperature`` knob and
        # reject the Converse call when it is present. Only send it for models
        # that still accept it (older Claude / Qwen-on-Bedrock / etc.).
        if not self._bedrock_omits_temperature(self.config.bedrock_model_id):
            request["inferenceConfig"]["temperature"] = self.config.temperature

        last_exc: Optional[Exception] = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                resp = client.converse(**request)
                # Converse returns output.message.content as a list of blocks;
                # concatenate the text blocks into the completion string.
                blocks = resp["output"]["message"]["content"]
                text = "".join(b.get("text", "") for b in blocks)
                if not text:
                    raise ValueError("Bedrock response carried no text content")
                return text
            except (NoCredentialsError, EndpointConnectionError) as exc:
                # Cannot reach Bedrock at all -> wait & resume, like vLLM down.
                raise ServiceUnavailableError(
                    f"Bedrock endpoint unreachable in {self.config.bedrock_region}: {exc}"
                ) from exc
            except (ClientError, BotoCoreError, KeyError, ValueError) as exc:
                last_exc = exc
                if attempt < self.config.max_retries:
                    time.sleep(self.config.retry_backoff_s * attempt)

        raise RequestError(
            f"Bedrock request failed after {self.config.max_retries} attempts: {last_exc}"
        )

    @staticmethod
    def _bedrock_omits_temperature(model_id: str) -> bool:
        """True if the Bedrock model rejects the ``temperature`` inference knob.

        Claude Opus 4.8 (and the 4.7+ line) deprecated ``temperature`` and fail
        the Converse call when it is supplied. Match those by model-id substring
        so the reward-generation call uses the model's default sampling instead.
        """
        mid = (model_id or "").lower()
        if "claude" not in mid:
            return False
        return any(tag in mid for tag in ("opus-4-8", "opus-4-7", "sonnet-4-8", "haiku-4-8"))

    @staticmethod
    def _extract_compute_reward(response_text: str) -> str:
        """Pull the compute_reward source out of the model response.

        Strategy: prefer fenced code blocks; among candidates, keep the first
        that actually defines compute_reward. Fall back to slicing from a bare
        `def compute_reward` to the end. Raise ExtractionError if nothing
        usable is found (Req 1.4)."""
        candidates = _CODE_BLOCK_RE.findall(response_text)
        for block in candidates:
            if "def compute_reward" in block:
                return _trim_to_function(block, "compute_reward")

        if "def compute_reward" in response_text:
            return _trim_to_function(response_text, "compute_reward")

        raise ExtractionError(
            "No 'compute_reward' definition found in the model response."
        )

    @staticmethod
    def _validate_python(code: str) -> None:
        """ast.parse the extracted code (Req 1.6)."""
        try:
            ast.parse(code)
        except SyntaxError as exc:
            raise CodeValidationError(
                f"Extracted reward code is not valid Python: {exc}"
            ) from exc


def _trim_to_function(text: str, func_name: str) -> str:
    """Return text starting at `def <func_name>` through the end of its
    indented body, dropping anything before the definition and any trailing
    prose that the model may have appended after a dedent."""
    lines = text.splitlines()
    start = next(
        (i for i, ln in enumerate(lines) if ln.lstrip().startswith(f"def {func_name}")),
        None,
    )
    if start is None:
        return text.strip()

    def_indent = len(lines[start]) - len(lines[start].lstrip())
    end = len(lines)
    for i in range(start + 1, len(lines)):
        line = lines[i]
        if not line.strip():
            continue
        indent = len(line) - len(line.lstrip())
        if indent <= def_indent:
            end = i
            break
    return "\n".join(lines[start:end]).strip()
