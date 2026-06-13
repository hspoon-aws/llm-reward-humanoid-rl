"""Shared error taxonomy for the YOUR_REPO system.

This module is the single canonical home for the error types used across the
components (Config loader, Qwen_Client, Reward_Executor, Orchestrator,
PPO_Runner, S3_Store). It mirrors the "Error Taxonomy" table in the design
document so every component raises and catches the same types.

Design references (design.md → Error Handling → Error Taxonomy):

  | Error                                | Raised by              | Meaning                                            |
  |--------------------------------------|------------------------|----------------------------------------------------|
  | ConfigurationError / ConfigError     | Config loader, Qwen    | Bad config value, GPU overlap, missing template    |
  | ExtractionError                      | Qwen_Client            | No compute_reward in model response (Req 1.4)      |
  | ValidationError                      | Qwen, Reward_Executor  | Code not parseable / missing compute_reward        |
  | RequestError                         | Qwen_Client            | Request failed after retries (Req 1.5, 2.3)        |
  | ServiceUnavailableError              | Qwen_Client            | Endpoint unreachable (Req 16.1)                    |
  | TimeoutError                         | Reward_Executor        | Reward exec exceeded sandbox limit (Req 5.2)       |
  | ExecutionError                       | Reward_Executor        | Reward raised during execution (Req 5.3)           |
  | DivergenceError                      | Reward_Executor, PPO   | Non-finite reward/loss (Req 5.4, 14.1)             |
  | OutOfMemoryError                     | PPO_Runner             | CUDA OOM during training (Req 15.1)                |
  | PersistError                         | S3_Store               | Artifact upload failed (Req 11.3)                  |

Note on shadowed builtins
--------------------------
`TimeoutError` and `OutOfMemoryError` are also names of Python builtins. The
design's taxonomy intentionally uses these names, so this module defines
PROJECT-SPECIFIC subclasses that *extend* the corresponding builtins. Because
they subclass the builtins, code that does ``except TimeoutError`` against the
builtin will still catch these, while importers of this module get the
project-scoped semantics (a common ``HumanoidRLError`` base). Import them
explicitly from ``src.exceptions`` to get the project versions.
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


# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #
class HumanoidRLError(Exception):
    """Base class for every error in the YOUR_REPO system.

    Catch this to handle any project-specific failure generically. Individual
    components raise the specific subclasses below.
    """


# --------------------------------------------------------------------------- #
# Configuration / startup (fail-fast, before the loop)
# --------------------------------------------------------------------------- #
class ConfigError(HumanoidRLError):
    """A configuration value is bad, a required template is missing, or the
    training GPU set overlaps the reserved model GPU (Req 3.3, 18.4, 18.5).

    ``ConfigurationError`` is provided as an alias so both spellings used in
    the design map to the same type.
    """


# Alias required by the design taxonomy (ConfigurationError / ConfigError).
ConfigurationError = ConfigError


class TemplateError(ConfigError):
    """A required prompt template is missing or unreadable (Req 3.3).

    Subclasses ConfigError so a missing template is treated as a configuration
    failure, while preserving the dedicated name the Qwen_Client uses.
    """


# --------------------------------------------------------------------------- #
# Language-model interaction (Qwen_Client)
# --------------------------------------------------------------------------- #
class ExtractionError(HumanoidRLError):
    """No ``compute_reward`` definition could be extracted from the model
    response (Req 1.4)."""


class ValidationError(HumanoidRLError):
    """Generated reward code is not parseable Python or does not define a
    ``compute_reward`` function (Req 1.6, 4.2, 4.3)."""


class RequestError(HumanoidRLError):
    """A model request failed after exhausting the configured retries
    (Req 1.5, 2.3)."""


class ServiceUnavailableError(HumanoidRLError):
    """The Language_Model endpoint could not be reached at all (Req 16.1).

    Distinct from ``RequestError`` so the Orchestrator can wait for the
    endpoint to recover and resume from the last checkpoint.
    """


# --------------------------------------------------------------------------- #
# Reward execution (Reward_Executor) — shadows builtins by design
# --------------------------------------------------------------------------- #
class TimeoutError(HumanoidRLError, builtins.TimeoutError):  # noqa: A001 - intentional shadow
    """Reward execution exceeded the sandbox wall-clock limit (Req 5.2).

    Project-specific class that intentionally shares the name of the builtin
    ``TimeoutError``. It subclasses both ``HumanoidRLError`` (so it is part of
    the project taxonomy) and the builtin ``TimeoutError`` (so existing
    ``except TimeoutError`` handlers against the builtin still catch it).
    """


class ExecutionError(HumanoidRLError):
    """A reward function raised an exception during sandboxed execution; the
    captured failure is surfaced through this error (Req 5.3)."""


class DivergenceError(HumanoidRLError):
    """A non-finite (NaN/inf) reward or training loss was detected
    (Req 5.4, 14.1)."""


class OutOfMemoryError(HumanoidRLError, builtins.MemoryError):  # noqa: A001 - intentional shadow
    """A CUDA out-of-memory condition occurred during training (Req 15.1).

    Project-specific class that intentionally shares the name of the builtin
    ``OutOfMemoryError``. It subclasses both ``HumanoidRLError`` and the
    builtin ``MemoryError`` (the builtin base of ``OutOfMemoryError``) so it is
    part of the project taxonomy while remaining catchable as a memory error.
    """


# --------------------------------------------------------------------------- #
# Persistence (S3_Store)
# --------------------------------------------------------------------------- #
class PersistError(HumanoidRLError):
    """Persisting an artifact to the S3_Store failed; the caller retains the
    local copy and records the failure rather than aborting (Req 11.3)."""
