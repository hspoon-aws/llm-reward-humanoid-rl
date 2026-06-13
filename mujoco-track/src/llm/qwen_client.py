"""Qwen_Client: talk to the self-hosted Qwen3-Coder model over an
OpenAI-compatible HTTP API to generate / refine reward functions and to
analyze training failures.

Carried over UNCHANGED from the Isaac track — the client is transport + prompt
plumbing and is framework-agnostic (it deals in prompt-in / text-out). Only the
prompt-template *contents* differ between tracks (JAX vs PyTorch reward
contract); the loading/substitution/extraction/validation logic is identical.

Implements requirements 1, 2, 3, and 16:
  - generate reward code from a task + observation-space description
  - include Metrics_History when refining
  - extract and ast.parse-validate the compute_reward definition
  - retry transient request failures, raise a typed error when exhausted
  - distinguish "endpoint unreachable" (service unavailable) so the
    Orchestrator can wait and resume
  - load all prompts from external template files

No JAX / MJX dependency, so it runs and unit-tests on the controller host.
"""

from __future__ import annotations

import ast
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import requests

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
    """Extracted reward code is not parseable Python (Req 1.6)."""


class RequestError(_SharedRequestError, QwenClientError):
    """The model request failed after exhausting retries (Req 1.5, 2.3)."""


class ServiceUnavailableError(_SharedServiceUnavailableError, QwenClientError):
    """The model endpoint could not be reached at all (Req 16.1)."""


@dataclass
class QwenClientConfig:
    endpoint: str = "http://127.0.0.1:8000/v1"
    model: str = "Qwen3-Coder-30B-A3B-Instruct"
    prompts_dir: Path = field(default_factory=lambda: Path("prompts"))
    max_retries: int = 3
    retry_backoff_s: float = 2.0
    request_timeout_s: float = 120.0
    temperature: float = 0.4
    max_tokens: int = 4096
    # Optional cross-process serialization of the vLLM call. When set to a file
    # path, every ``_chat`` acquires an exclusive ``flock`` on it for the
    # duration of the request, so multiple independent loop processes sharing one
    # self-hosted vLLM take turns instead of contending and exhausting retries
    # (see docs/lesson-problems-and-resolutions.md §7D). None = no locking
    # (single-process runs need none). Generation is rare + fast, so the wait is
    # negligible.
    request_lock_path: Optional[str] = None
    lock_timeout_s: float = 300.0


_CODE_BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)

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
        balance_priority: bool = False,
        best_reward_code: Optional[str] = None,
    ) -> str:
        """Request a reward function. Uses the refine template when any feedback
        is available, otherwise the initial template. Returns the extracted,
        ast-validated source of compute_reward (Reqs 1.1-1.6, 2.2, 12.1).

        ``best_reward_code`` is the source of the best-so-far reward; the refine
        template shows it as the explicit starting point for INCREMENTAL
        improvement (so the model refines a known-good reward instead of
        rewriting blind each iteration)."""
        use_refine = bool(
            metrics_history or guidance or validation_error or balance_priority
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
                )
                if part
            )
            prompt = self._templates["refine"].format(
                task_description=task_description,
                obs_space=obs_space,
                goal_description=goal_description or "(reach the configured Goal)",
                metrics_history=metrics_history or "(none yet)",
                guidance=combined_guidance or "(none)",
                best_reward_code=best_reward_code
                or "(no prior reward yet — design a first balanced goal reward)",
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
        text (Reqs 2.1-2.3)."""
        prompt = self._templates["analyze"].format(
            metrics=metrics,
            behavior_description=behavior_description,
        )
        return self._chat(prompt).strip()

    # ----------------------------- internals ------------------------------ #
    def _chat(self, prompt: str) -> str:
        """POST a single-turn chat completion, retrying transient failures.

        Connection-level failures are surfaced as ServiceUnavailableError so the
        Orchestrator can wait for vLLM to come back (Req 16). HTTP / other
        request failures are retried up to max_retries, then RequestError
        (Reqs 1.5, 2.3).

        When ``request_lock_path`` is configured, the whole request is serialized
        across processes via an exclusive file lock so concurrent loops sharing
        one vLLM take turns rather than contending (§7D)."""
        url = f"{self.config.endpoint.rstrip('/')}/chat/completions"
        payload = {
            "model": self.config.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }
        with self._request_lock():
            return self._post_with_retries(url, payload)

    def _post_with_retries(self, url: str, payload: dict) -> str:
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

    @contextmanager
    def _request_lock(self):
        """Exclusive cross-process lock around the vLLM call (no-op if unset).

        Uses ``fcntl.flock`` on ``request_lock_path`` so independent loop
        processes serialize their generation requests. Best-effort: if locking
        is unavailable (e.g. non-POSIX) or the lock can't be acquired within
        ``lock_timeout_s``, it proceeds without the lock rather than blocking the
        loop forever."""
        path = self.config.request_lock_path
        if not path:
            yield
            return
        try:
            import fcntl
        except ImportError:  # pragma: no cover - non-POSIX
            yield
            return

        handle = open(path, "w")  # noqa: SIM115 - held for the lock lifetime
        acquired = False
        deadline = time.monotonic() + float(self.config.lock_timeout_s)
        try:
            while True:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                    acquired = True
                    break
                except OSError:
                    if time.monotonic() >= deadline:
                        break  # proceed unlocked rather than hang
                    time.sleep(0.5)
            yield
        finally:
            if acquired:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            handle.close()

    @staticmethod
    def _extract_compute_reward(response_text: str) -> str:
        """Pull the compute_reward source out of the model response (Req 1.4)."""
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
