"""Config loader for the YOUR_REPO system (spec Req 18).

This module loads the single external run configuration (``config/run_config.yaml``
by default), applies documented defaults for any absent optional fields, validates
types and ranges, and produces a flat, typed :class:`Config` dataclass exposing the
fields the design's Data Models section lists.

Design references:
  - design.md -> Components and Interfaces -> Config Loader (`src/config.py`) -- Req 18
  - design.md -> Data Models -> ``@dataclass Config``
  - requirements.md -> Requirement 18 (load + validate run configuration)

Key behaviors:
  - Load from a single external source (Req 18.1).
  - Provide every required field, including the Goal position, Success_Radius, the
    staged-gate thresholds, GPU set, env counts, recovery knobs, selection metric,
    S3 location, LLM endpoint, and capture/blog fields (Req 18.2, 18.6).
  - Apply documented defaults where a value is absent (Req 18.3).
  - Reject wrong-type or out-of-range values with a descriptive ``ConfigError`` and
    do NOT start the loop (Req 18.4).
  - Reject any training GPU set that contains the reserved language-model GPU (GPU 0)
    (Req 18.5).

The :class:`Config` is intentionally pure data: no Isaac Lab / torch dependency, so it
can be loaded and unit-tested on the controller host without GPUs. ``load_config`` is a
pure function of the file at ``path`` (plus documented defaults), which makes it
straightforward to exercise with property-based tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Optional

import yaml

from .exceptions import ConfigError

__all__ = ["Config", "load_config", "DEFAULTS"]


# --------------------------------------------------------------------------- #
# Documented defaults (Req 18.3)
# --------------------------------------------------------------------------- #
# Each entry is the value applied when the corresponding field is absent from the
# external config source. These mirror the canonical ``config/run_config.yaml`` and
# the design's stated bounds. The reserved language-model GPU is always GPU 0.
LLM_GPU = 0

DEFAULTS: dict[str, Any] = {
    # loop / model
    "max_iterations": 12,
    "task_description": None,
    "qwen_max_retries": 6,
    "qwen_retry_backoff_s": 3.0,
    "qwen_request_timeout_s": 180.0,
    "llm_endpoint": "http://127.0.0.1:8000/v1",
    "llm_provider": "vllm",
    "bedrock_model_id": "global.anthropic.claude-opus-4-8",
    "bedrock_region": "us-west-2",
    # training
    "env_id": "Isaac-Velocity-Flat-H1-v0",
    "train_epochs": 1500,
    "num_steps_per_env": 24,
    "checkpoint_interval": 500,
    "training_gpus": [1, 2, 3, 4, 5, 6, 7],
    "num_envs": 4096,
    "oom_fallback_envs": 2048,
    "learning_rate": 1.0e-3,
    "warm_start": False,
    "lr_reduction_factor": 0.5,
    # sandbox / recovery
    "sandbox_time_limit_s": 5.0,
    "fall_threshold_s": 3.0,
    # goal-reaching task
    "goal_position": (5.0, 0.0),
    "success_radius_m": 0.5,
    "min_progress_distance_m": 1.0,
    "time_to_goal_threshold_s": 12.0,
    "path_efficiency_threshold": 0.6,
    "selection_metric": "success_rate",
    # persistence
    "s3_location": "s3://humanoid-from-scratch-123456789012/runs",
    # blog / capture (Req 18.6)
    "capture_interval": 100.0,
    "capture_resolution": (1280, 720),
    "capture_env_subset_size": 4,
    "blog_output_location": "runs",
    "blog_output_format": "markdown",
}

# Selection metric must name a numeric field of EvalMetrics that the Orchestrator
# can compare against the current Best_Policy (Req 19.1).
_VALID_SELECTION_METRICS = frozenset(
    {
        "distance_to_goal_m",
        "success_rate",
        "time_to_goal_s",
        "path_efficiency",
        "upright_time_s",
        "fall_rate",
        "avg_forward_speed_mps",
        "energy_efficiency",
        "gait_smoothness",
        "symmetry_score",
    }
)


@dataclass
class Config:
    """Flat, typed view of a validated run configuration (design.md -> Data Models).

    Every field is required at the dataclass level; ``load_config`` is responsible for
    filling absent values from :data:`DEFAULTS` before constructing the instance, so a
    constructed ``Config`` is always complete and validated.
    """

    # loop / model
    max_iterations: int
    qwen_max_retries: int
    llm_endpoint: str
    # training
    train_epochs: int
    checkpoint_interval: int
    training_gpus: list[int]            # validated: GPU 0 not in set (Req 18.5)
    num_envs: int
    oom_fallback_envs: int              # < num_envs (Req 15.1)
    lr_reduction_factor: float          # in (0, 1) (Req 14.2)
    env_id: str                         # Isaac Lab task id to train (Req 8.1)
    # sandbox / recovery
    sandbox_time_limit_s: float
    fall_threshold_s: float             # Req 13.1
    # goal-reaching task
    goal_position: tuple[float, float]  # point B (Req 18.2)
    success_radius_m: float             # Req 18.2
    min_progress_distance_m: float      # makes-progress gate (Req 17.1, 17.4)
    time_to_goal_threshold_s: float     # efficient-goal gate (Req 17.3, 17.4)
    path_efficiency_threshold: float    # efficient-goal gate (Req 17.3, 17.4)
    selection_metric: str               # field name within EvalMetrics (Req 19.1)
    # persistence
    s3_location: str
    # blog / capture (Req 18.6)
    capture_interval: float
    capture_resolution: tuple[int, int]
    capture_env_subset_size: int
    blog_output_location: str
    blog_output_format: str
    # PPO rollout horizon per env per epoch — the env-step budget multiplier
    # (total env-steps/iter = num_envs × num_steps_per_env × train_epochs).
    # Defaulted so existing constructions stay valid; load_config always supplies it.
    num_steps_per_env: int = 24
    # Qwen_Client request resilience (Req 16). Defaulted so existing constructions
    # (test fixtures) stay valid; load_config always supplies them from llm.*.
    qwen_retry_backoff_s: float = 3.0
    qwen_request_timeout_s: float = 180.0
    # PPO optimizer learning rate. Previously hard-coded to 1e-3 in the trainer
    # (never read from config); surfaced here so it is tunable from yaml.
    # Defaulted so existing constructions / test fixtures stay valid; load_config
    # always supplies it from training.learning_rate.
    learning_rate: float = 1.0e-3
    # Language-model backend selection (Req 1, 3, 16). "vllm" (default) uses the
    # self-hosted Qwen at ``llm_endpoint``; "bedrock" calls Amazon Bedrock
    # directly with ``bedrock_model_id`` in ``bedrock_region`` (no vLLM needed).
    # Defaulted so existing configs/tests stay valid; load_config supplies them.
    llm_provider: str = "vllm"
    bedrock_model_id: str = "global.anthropic.claude-opus-4-8"
    bedrock_region: str = "us-west-2"
    # Warm-start (continuous learning): when True, each Eureka iteration loads the
    # previous iteration's trained policy and continues training instead of
    # restarting PPO from random weights. Suited to a fixed goal with refined
    # rewards. Defaulted off (canonical from-scratch Eureka) for back-compat.
    warm_start: bool = False
    # Natural-language task description handed to the LLM reward designer (Req 1).
    # The primary plain-English control surface: describe the desired behavior in
    # words (e.g. "walk forward facing the goal with a natural upright gait;
    # never walk backward") and the LLM turns it into the reward. None => the
    # orchestrator's built-in DEFAULT_TASK_DESCRIPTION.
    task_description: Optional[str] = None


# --------------------------------------------------------------------------- #
# Coercion / validation helpers
# --------------------------------------------------------------------------- #
def _require_int(value: Any, field: str, *, minimum: int | None = None,
                 maximum: int | None = None) -> int:
    # bool is a subclass of int but is never an acceptable numeric config value.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(
            f"{field}: expected an integer, got {type(value).__name__} ({value!r})"
        )
    if minimum is not None and value < minimum:
        raise ConfigError(f"{field}: must be >= {minimum}, got {value}")
    if maximum is not None and value > maximum:
        raise ConfigError(f"{field}: must be <= {maximum}, got {value}")
    return value


def _require_float(value: Any, field: str, *, minimum: float | None = None,
                   maximum: float | None = None, exclusive_min: bool = False,
                   exclusive_max: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(
            f"{field}: expected a number, got {type(value).__name__} ({value!r})"
        )
    fvalue = float(value)
    if fvalue != fvalue or fvalue in (float("inf"), float("-inf")):
        raise ConfigError(f"{field}: must be finite, got {value!r}")
    if minimum is not None:
        if exclusive_min and not fvalue > minimum:
            raise ConfigError(f"{field}: must be > {minimum}, got {fvalue}")
        if not exclusive_min and fvalue < minimum:
            raise ConfigError(f"{field}: must be >= {minimum}, got {fvalue}")
    if maximum is not None:
        if exclusive_max and not fvalue < maximum:
            raise ConfigError(f"{field}: must be < {maximum}, got {fvalue}")
        if not exclusive_max and fvalue > maximum:
            raise ConfigError(f"{field}: must be <= {maximum}, got {fvalue}")
    return fvalue


def _require_str(value: Any, field: str, *, non_empty: bool = True) -> str:
    if not isinstance(value, str):
        raise ConfigError(
            f"{field}: expected a string, got {type(value).__name__} ({value!r})"
        )
    if non_empty and not value.strip():
        raise ConfigError(f"{field}: must be a non-empty string")
    return value


def _require_bool(value: Any, field: str) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(
            f"{field}: expected a boolean, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_xy(value: Any, field: str) -> tuple[float, float]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{field}: expected a 2-element [x, y] sequence, "
            f"got {type(value).__name__} ({value!r})"
        )
    if len(value) != 2:
        raise ConfigError(
            f"{field}: expected exactly 2 elements [x, y], got {len(value)}"
        )
    x = _require_float(value[0], f"{field}[0]")
    y = _require_float(value[1], f"{field}[1]")
    return (x, y)


def _require_resolution(value: Any, field: str) -> tuple[int, int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{field}: expected a 2-element [width, height] sequence, "
            f"got {type(value).__name__} ({value!r})"
        )
    if len(value) != 2:
        raise ConfigError(
            f"{field}: expected exactly 2 elements [width, height], got {len(value)}"
        )
    width = _require_int(value[0], f"{field}[width]", minimum=1)
    height = _require_int(value[1], f"{field}[height]", minimum=1)
    return (width, height)


def _require_gpu_list(value: Any, field: str) -> list[int]:
    if isinstance(value, (str, bytes)) or not isinstance(value, (list, tuple)):
        raise ConfigError(
            f"{field}: expected a list of GPU indices, "
            f"got {type(value).__name__} ({value!r})"
        )
    if len(value) == 0:
        raise ConfigError(f"{field}: at least one training GPU is required")
    gpus: list[int] = []
    for i, item in enumerate(value):
        gpus.append(_require_int(item, f"{field}[{i}]", minimum=0))
    if len(set(gpus)) != len(gpus):
        raise ConfigError(f"{field}: duplicate GPU indices are not allowed: {gpus}")
    return gpus


def _section(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    """Return a nested mapping section, or an empty mapping if absent.

    A present-but-wrong-type section is a config error (Req 18.4).
    """
    if name not in raw:
        return {}
    section = raw[name]
    if section is None:
        return {}
    if not isinstance(section, Mapping):
        raise ConfigError(
            f"{name}: expected a mapping section, got {type(section).__name__}"
        )
    return section


def _get(section: Mapping[str, Any], key: str, default_key: str) -> Any:
    """Fetch ``key`` from a section, falling back to the documented default."""
    if key in section and section[key] is not None:
        return section[key]
    return DEFAULTS[default_key]


def _endpoint_is_loopback(endpoint: str) -> bool:
    """True if the LLM endpoint points at this host (loopback).

    Used to decide whether GPU 0 is reserved for a co-resident vLLM (Req 18.5)
    or free for training (cross-host vLLM, e.g. a single-GPU Isaac host that
    reaches a separate vLLM box over the VPC). Loopback hosts: ``localhost``,
    ``127.0.0.0/8``, ``::1``. Anything else (a routable IP/hostname) is treated
    as cross-host. Unparseable endpoints default to loopback (the safe, strict
    interpretation that keeps the Req 18.5 guard on).
    """
    from urllib.parse import urlsplit

    candidate = endpoint if "://" in endpoint else f"//{endpoint}"
    try:
        host = (urlsplit(candidate).hostname or "").lower()
    except (ValueError, TypeError):
        return True
    if not host:
        return True
    return host in {"localhost", "::1"} or host.startswith("127.")


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Config:
    """Load, default, and validate the run configuration at ``path``.

    Maps the nested YAML layout (``llm``, ``goal``, ``sandbox``, ``training``,
    ``evaluation``, ``s3``, ``capture``, ``blog``, ``video`` plus top-level
    ``max_iterations`` / ``selection_metric``) into a flat :class:`Config`.

    Applies documented defaults for absent optional fields (Req 18.3), validates
    types and ranges (Req 18.4), and rejects any training GPU set that contains the
    reserved language-model GPU 0 (Req 18.5). On any problem it raises
    :class:`~src.exceptions.ConfigError` so the loop never starts on bad config.
    """
    config_path = Path(path)
    try:
        text = config_path.read_text(encoding="utf-8")
    except FileNotFoundError as exc:
        raise ConfigError(f"config file not found: {path}") from exc
    except OSError as exc:
        raise ConfigError(f"config file could not be read: {path} ({exc})") from exc

    try:
        loaded = yaml.safe_load(text)
    except yaml.YAMLError as exc:
        raise ConfigError(f"config file is not valid YAML: {path} ({exc})") from exc

    if loaded is None:
        loaded = {}
    if not isinstance(loaded, Mapping):
        raise ConfigError(
            f"config root must be a mapping, got {type(loaded).__name__}"
        )

    llm = _section(loaded, "llm")
    goal = _section(loaded, "goal")
    sandbox = _section(loaded, "sandbox")
    training = _section(loaded, "training")
    evaluation = _section(loaded, "evaluation")
    s3 = _section(loaded, "s3")
    capture = _section(loaded, "capture")
    blog = _section(loaded, "blog")

    # --- loop / model ----------------------------------------------------- #
    max_iterations = _require_int(
        _get(loaded, "max_iterations", "max_iterations"),
        "max_iterations", minimum=1,
    )
    # Natural-language task description (the plain-English control surface). A
    # top-level optional string; when present it overrides the built-in default
    # the Orchestrator hands to the reward designer. Absent/null => default.
    task_description_raw = loaded.get("task_description", DEFAULTS["task_description"])
    if task_description_raw is None:
        task_description: Optional[str] = None
    else:
        task_description = _require_str(task_description_raw, "task_description")
    qwen_max_retries = _require_int(
        _get(llm, "max_retries", "qwen_max_retries"),
        "llm.max_retries", minimum=0,
    )
    qwen_retry_backoff_s = _require_float(
        _get(llm, "retry_backoff_s", "qwen_retry_backoff_s"),
        "llm.retry_backoff_s", minimum=0.0,
    )
    qwen_request_timeout_s = _require_float(
        _get(llm, "request_timeout_s", "qwen_request_timeout_s"),
        "llm.request_timeout_s", minimum=0.0, exclusive_min=True,
    )
    llm_endpoint = _require_str(
        _get(llm, "endpoint", "llm_endpoint"), "llm.endpoint",
    )
    llm_provider = _require_str(
        _get(llm, "provider", "llm_provider"), "llm.provider",
    ).strip().lower()
    if llm_provider not in ("vllm", "bedrock"):
        raise ConfigError(
            f"llm.provider: must be 'vllm' or 'bedrock', got {llm_provider!r}"
        )
    bedrock_model_id = _require_str(
        _get(llm, "bedrock_model_id", "bedrock_model_id"), "llm.bedrock_model_id",
    )
    bedrock_region = _require_str(
        _get(llm, "bedrock_region", "bedrock_region"), "llm.bedrock_region",
    )

    # --- training --------------------------------------------------------- #
    env_id = _require_str(
        _get(training, "task", "env_id"), "training.task",
    )
    train_epochs = _require_int(
        _get(training, "epochs", "train_epochs"),
        "training.epochs", minimum=1,
    )
    num_steps_per_env = _require_int(
        _get(training, "num_steps_per_env", "num_steps_per_env"),
        "training.num_steps_per_env", minimum=1,
    )
    checkpoint_interval = _require_int(
        _get(training, "checkpoint_interval", "checkpoint_interval"),
        "training.checkpoint_interval", minimum=1,
    )
    training_gpus = _require_gpu_list(
        _get(training, "training_gpus", "training_gpus"),
        "training.training_gpus",
    )
    num_envs = _require_int(
        _get(training, "num_envs", "num_envs"),
        "training.num_envs", minimum=1,
    )
    oom_fallback_envs = _require_int(
        _get(training, "oom_fallback_num_envs", "oom_fallback_envs"),
        "training.oom_fallback_num_envs", minimum=1,
    )
    lr_reduction_factor = _require_float(
        _get(training, "lr_reduction_factor", "lr_reduction_factor"),
        "training.lr_reduction_factor",
        minimum=0.0, maximum=1.0, exclusive_min=True, exclusive_max=True,
    )
    learning_rate = _require_float(
        _get(training, "learning_rate", "learning_rate"),
        "training.learning_rate", minimum=0.0, exclusive_min=True,
    )
    warm_start = _require_bool(
        _get(training, "warm_start", "warm_start"), "training.warm_start",
    )

    # --- sandbox / recovery ----------------------------------------------- #
    sandbox_time_limit_s = _require_float(
        _get(sandbox, "time_limit_s", "sandbox_time_limit_s"),
        "sandbox.time_limit_s", minimum=0.0, exclusive_min=True,
    )
    fall_threshold_s = _require_float(
        _get(evaluation, "fall_threshold_s", "fall_threshold_s"),
        "evaluation.fall_threshold_s", minimum=0.0, exclusive_min=True,
    )

    # --- goal-reaching task ----------------------------------------------- #
    goal_position = _require_xy(
        _get(goal, "goal_position", "goal_position"), "goal.goal_position",
    )
    success_radius_m = _require_float(
        _get(goal, "success_radius_m", "success_radius_m"),
        "goal.success_radius_m", minimum=0.0, exclusive_min=True,
    )
    min_progress_distance_m = _require_float(
        _get(evaluation, "min_progress_distance_m", "min_progress_distance_m"),
        "evaluation.min_progress_distance_m", minimum=0.0, exclusive_min=True,
    )
    time_to_goal_threshold_s = _require_float(
        _get(evaluation, "time_to_goal_threshold_s", "time_to_goal_threshold_s"),
        "evaluation.time_to_goal_threshold_s", minimum=0.0, exclusive_min=True,
    )
    path_efficiency_threshold = _require_float(
        _get(evaluation, "path_efficiency_threshold", "path_efficiency_threshold"),
        "evaluation.path_efficiency_threshold",
        minimum=0.0, maximum=1.0, exclusive_min=True,
    )
    selection_metric = _require_str(
        _get(loaded, "selection_metric", "selection_metric"), "selection_metric",
    )
    if selection_metric not in _VALID_SELECTION_METRICS:
        raise ConfigError(
            f"selection_metric: must be one of "
            f"{sorted(_VALID_SELECTION_METRICS)}, got {selection_metric!r}"
        )

    # --- persistence ------------------------------------------------------ #
    s3_location = _resolve_s3_location(s3)

    # --- blog / capture (Req 18.6) ---------------------------------------- #
    capture_interval = _require_float(
        _get(capture, "capture_interval_epochs", "capture_interval"),
        "capture.capture_interval_epochs", minimum=0.0, exclusive_min=True,
    )
    capture_resolution = _require_resolution(
        _capture_resolution_value(capture), "capture.[width, height]",
    )
    capture_env_subset_size = _require_int(
        _get(capture, "env_subset_size", "capture_env_subset_size"),
        "capture.env_subset_size", minimum=1,
    )
    blog_output_location = _require_str(
        _get(blog, "output_location", "blog_output_location"),
        "blog.output_location",
    )
    blog_output_format = _require_str(
        _get(blog, "output_format", "blog_output_format"), "blog.output_format",
    )

    # --- cross-field validation ------------------------------------------- #
    # GPU 0 is reserved for the Language_Model ONLY when vLLM is co-resident on
    # this host (a loopback endpoint). When the LLM endpoint is cross-host (e.g.
    # a separate vLLM box reached over the VPC), GPU 0 is free for training — as
    # on a single-GPU Isaac host (g6/L4) whose only device is 0. So the Req 18.5
    # exclusion is enforced only for loopback endpoints.
    # A "bedrock" provider is never co-resident (it is a managed AWS service),
    # so GPU 0 is always free for training under bedrock, regardless of endpoint.
    _llm_is_local = llm_provider != "bedrock" and _endpoint_is_loopback(llm_endpoint)
    if _llm_is_local and LLM_GPU in training_gpus:
        raise ConfigError(
            f"training.training_gpus must not include GPU {LLM_GPU}, which is "
            f"reserved for the co-resident language model (Req 18.5); got "
            f"{training_gpus}. (GPU 0 is allowed when llm.endpoint is cross-host.)"
        )
    # If the config names an explicit llm_gpu, it must also be absent from the
    # set — but again only when the LLM is co-resident (loopback endpoint).
    if _llm_is_local and "llm_gpu" in training:
        llm_gpu = _require_int(training["llm_gpu"], "training.llm_gpu", minimum=0)
        if llm_gpu in training_gpus:
            raise ConfigError(
                f"training.training_gpus must not include the reserved llm_gpu "
                f"({llm_gpu}); got {training_gpus}"
            )

    # OOM fallback env count must be a genuine reduction (Req 15.1).
    if oom_fallback_envs > num_envs:
        raise ConfigError(
            f"training.oom_fallback_num_envs ({oom_fallback_envs}) must be <= "
            f"training.num_envs ({num_envs})"
        )

    return Config(
        max_iterations=max_iterations,
        task_description=task_description,
        qwen_max_retries=qwen_max_retries,
        qwen_retry_backoff_s=qwen_retry_backoff_s,
        qwen_request_timeout_s=qwen_request_timeout_s,
        llm_endpoint=llm_endpoint,
        llm_provider=llm_provider,
        bedrock_model_id=bedrock_model_id,
        bedrock_region=bedrock_region,
        train_epochs=train_epochs,
        num_steps_per_env=num_steps_per_env,
        checkpoint_interval=checkpoint_interval,
        training_gpus=training_gpus,
        num_envs=num_envs,
        oom_fallback_envs=oom_fallback_envs,
        lr_reduction_factor=lr_reduction_factor,
        learning_rate=learning_rate,
        warm_start=warm_start,
        env_id=env_id,
        sandbox_time_limit_s=sandbox_time_limit_s,
        fall_threshold_s=fall_threshold_s,
        goal_position=goal_position,
        success_radius_m=success_radius_m,
        min_progress_distance_m=min_progress_distance_m,
        time_to_goal_threshold_s=time_to_goal_threshold_s,
        path_efficiency_threshold=path_efficiency_threshold,
        selection_metric=selection_metric,
        s3_location=s3_location,
        capture_interval=capture_interval,
        capture_resolution=capture_resolution,
        capture_env_subset_size=capture_env_subset_size,
        blog_output_location=blog_output_location,
        blog_output_format=blog_output_format,
    )


def _resolve_s3_location(s3: Mapping[str, Any]) -> str:
    """Build the S3 location string from the ``s3`` section.

    Accepts either an explicit ``location`` URI or a ``bucket`` (+ optional
    ``prefix``) pair; falls back to the documented default when absent.
    """
    if "location" in s3 and s3["location"] is not None:
        return _require_str(s3["location"], "s3.location")
    if "bucket" in s3 and s3["bucket"] is not None:
        bucket = _require_str(s3["bucket"], "s3.bucket")
        location = f"s3://{bucket}"
        if "prefix" in s3 and s3["prefix"] is not None:
            prefix = _require_str(s3["prefix"], "s3.prefix").strip("/")
            if prefix:
                location = f"{location}/{prefix}"
        return location
    return DEFAULTS["s3_location"]


def _capture_resolution_value(capture: Mapping[str, Any]) -> Any:
    """Resolve capture resolution from ``width``/``height`` keys or the default.

    If neither width nor height is supplied, return the documented default tuple.
    A partial specification (only one of the two) is treated as the default for the
    missing dimension so a present value is still honored.
    """
    if "width" not in capture and "height" not in capture:
        return DEFAULTS["capture_resolution"]
    default_w, default_h = DEFAULTS["capture_resolution"]
    width = capture.get("width", default_w)
    height = capture.get("height", default_h)
    return (width, height)
