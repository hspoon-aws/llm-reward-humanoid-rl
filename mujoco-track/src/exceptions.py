"""Shared error taxonomy for the humanoid-mujoco-llm-rl system.

Carried over UNCHANGED from the Isaac project — the error taxonomy is
framework-agnostic (it names failure modes, not Isaac/MuJoCo specifics). The
single canonical home for the error types used across components (Config
loader, Qwen_Client, Reward_Executor, Orchestrator, PPO_Runner, S3_Store).

  | Error                            | Raised by              | Meaning                                       |
  |----------------------------------|------------------------|-----------------------------------------------|
  | ConfigurationError / ConfigError | Config loader, Qwen    | Bad config value, GPU overlap, missing template|
  | ExtractionError                  | Qwen_Client            | No compute_reward in model response (Req 1.4) |
  | ValidationError                  | Qwen, Reward_Executor  | Code not parseable / missing compute_reward   |
  | RequestError                     | Qwen_Client            | Request failed after retries (Req 1.5, 2.3)   |
  | ServiceUnavailableError          | Qwen_Client            | Endpoint unreachable (Req 16.1)               |
  | TimeoutError                     | Reward_Executor        | Reward exec exceeded sandbox limit (Req 5.2)  |
  | ExecutionError                   | Reward_Executor        | Reward raised during execution (Req 5.3)      |
  | DivergenceError                  | Reward_Executor, PPO   | Non-finite reward/loss (Req 5.4, 14.1)        |
  | OutOfMemoryError                 | PPO_Runner             | GPU OOM during training (Req 15.1)            |
  | PersistError                     | S3_Store               | Artifact upload failed (Req 11.3)             |
"""

from __future__ import annotations

import builtins

__all__ = [
    "HumanoidRLError",
    "ConfigError",
    "ConfigurationError",
    "ExtractionError",
    "ValidationError",
    "TemplateError",
    "RequestError",
    "ServiceUnavailableError",
    "TimeoutError",
    "ExecutionError",
    "DivergenceError",
    "OutOfMemoryError",
    "PersistError",
]


class HumanoidRLError(Exception):
    """Base class for every error in the humanoid-mujoco-llm-rl system."""


class ConfigError(HumanoidRLError):
    """Bad config value, missing template, or training GPU set overlapping the
    reserved model GPU (Req 3.3, 18.4, 18.5)."""


ConfigurationError = ConfigError


class TemplateError(ConfigError):
    """A required prompt template is missing or unreadable (Req 3.3)."""


class ExtractionError(HumanoidRLError):
    """No ``compute_reward`` definition could be extracted (Req 1.4)."""


class ValidationError(HumanoidRLError):
    """Generated reward code is not parseable Python or lacks ``compute_reward``
    (Req 1.6, 4.2, 4.3)."""


class RequestError(HumanoidRLError):
    """A model request failed after exhausting retries (Req 1.5, 2.3)."""


class ServiceUnavailableError(HumanoidRLError):
    """The Language_Model endpoint could not be reached at all (Req 16.1)."""


class TimeoutError(HumanoidRLError, builtins.TimeoutError):  # noqa: A001 - intentional shadow
    """Reward execution exceeded the sandbox wall-clock limit (Req 5.2)."""


class ExecutionError(HumanoidRLError):
    """A reward function raised during sandboxed execution (Req 5.3)."""


class DivergenceError(HumanoidRLError):
    """A non-finite (NaN/inf) reward or training loss was detected
    (Req 5.4, 14.1)."""


class OutOfMemoryError(HumanoidRLError, builtins.MemoryError):  # noqa: A001 - intentional shadow
    """A GPU out-of-memory condition occurred during training (Req 15.1)."""


class PersistError(HumanoidRLError):
    """Persisting an artifact to the S3_Store failed (Req 11.3)."""
