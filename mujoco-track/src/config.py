"""Config loader for the humanoid-mujoco-llm-rl system (spec Req 18).

Ported from the Isaac track's ``src/config.py`` with the MuJoCo-track deltas:
  - ``env_name`` (Playground registry id, default ``H1JoystickGaitTracking``)
    replaces the Isaac ``env_id`` / ``training.task``;
  - ``xla_python_client_mem_fraction`` caps JAX GPU memory preallocation so vLLM
    retains GPU-0 headroom (Req 8.2);
  - the default S3 prefix is ``runs-mujoco`` so the two tracks don't collide.

Loads the single external run configuration, applies documented defaults for
absent fields, validates types/ranges, rejects a training GPU set containing the
reserved language-model GPU (GPU 0), and returns a flat, typed :class:`Config`.
Pure data: no JAX/MJX dependency, so it loads and unit-tests off-GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from .exceptions import ConfigError

__all__ = ["Config", "load_config", "DEFAULTS"]


LLM_GPU = 0

DEFAULTS: dict[str, Any] = {
    # loop / model
    "max_iterations": 12,
    "qwen_max_retries": 3,
    "llm_endpoint": "http://127.0.0.1:8000/v1",
    # LLM backend selection. provider: "vllm" (self-hosted Qwen, default) or
    # "bedrock" (Amazon Bedrock Claude). model/region are provider-specific.
    "llm_provider": "vllm",
    "llm_model": "Qwen3-Coder-30B-A3B-Instruct",
    "llm_region": "us-west-2",
    # training (MuJoCo MJX + Brax)
    "env_name": "H1JoystickGaitTracking",
    "train_epochs": 1500,
    "checkpoint_interval": 500,
    "training_gpus": [1, 2, 3, 4, 5, 6, 7],
    "num_envs": 4096,
    "oom_fallback_envs": 2048,
    "lr_reduction_factor": 0.5,
    "xla_python_client_mem_fraction": 0.6,
    # PPO knobs (previously hard-coded; now config-tunable). episode_length and
    # learning_rate were read via getattr with these same defaults, so existing
    # configs are unaffected.
    "learning_rate": 1.0e-3,
    "episode_length": 1000,
    "num_evals": 10,
    # balance-before-locomotion shaping (run v4 finding; defaults = legacy
    # behaviour, so existing configs/tests are unchanged).
    "fall_terminate": False,
    "fall_height_m": 0.5,
    "standing_reward": 0.0,
    # warm-start: seed each iteration's PPO from the best policy so far
    # (fine-tune across iterations) instead of random init. Default off = legacy.
    "warm_start": False,
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
    "s3_location": "s3://humanoid-from-scratch-123456789012/runs-mujoco",
    # blog / capture (Req 18.6)
    "capture_interval": 100.0,
    "capture_resolution": (1280, 720),
    "capture_env_subset_size": 4,
    "blog_output_location": "runs-mujoco",
    "blog_output_format": "markdown",
}

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
    """Flat, typed view of a validated run configuration (design.md -> Data Models)."""

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
    env_name: str                       # Playground registry id (Req 8.1, 18.5 delta)
    xla_python_client_mem_fraction: float  # JAX GPU mem cap (Req 8.2 delta)
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

    # PPO knobs + balance shaping. Defaulted so existing direct Config(...)
    # constructions (tests, callers) keep working; load_config always supplies
    # them from DEFAULTS-backed values.
    learning_rate: float = 1.0e-3       # base PPO learning rate
    episode_length: int = 1000          # PPO rollout horizon (steps)
    num_evals: int = 10                 # PPO eval points per training run
    fall_terminate: bool = False        # end episode when torso < fall_height_m
    fall_height_m: float = 0.5          # torso z below this = fallen
    standing_reward: float = 0.0        # per-step reward floor for staying upright
    warm_start: bool = False            # fine-tune each iter from best policy so far
    # LLM backend selection (defaulted -> existing constructions unaffected).
    llm_provider: str = "vllm"          # "vllm" (self-hosted Qwen) or "bedrock"
    llm_model: str = "Qwen3-Coder-30B-A3B-Instruct"
    llm_region: str = "us-west-2"       # Bedrock region (provider="bedrock")


# --------------------------------------------------------------------------- #
# Coercion / validation helpers
# --------------------------------------------------------------------------- #
def _require_int(value: Any, field: str, *, minimum: int | None = None,
                 maximum: int | None = None) -> int:
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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def load_config(path: str) -> Config:
    """Load, default, and validate the run configuration at ``path``.

    Maps the nested YAML layout (``llm``, ``goal``, ``sandbox``, ``training``,
    ``evaluation``, ``s3``, ``capture``, ``blog``, ``video`` plus top-level
    ``max_iterations`` / ``selection_metric``) into a flat :class:`Config`.

    Applies documented defaults (Req 18.3), validates types/ranges (Req 18.4),
    and rejects any training GPU set that contains the reserved language-model
    GPU 0 (Req 18.5). On any problem raises :class:`ConfigError` so the loop never
    starts on bad config.
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
    qwen_max_retries = _require_int(
        _get(llm, "max_retries", "qwen_max_retries"),
        "llm.max_retries", minimum=0,
    )
    llm_endpoint = _require_str(
        _get(llm, "endpoint", "llm_endpoint"), "llm.endpoint",
    )
    llm_provider = str(_get(llm, "provider", "llm_provider")).strip().lower()
    if llm_provider not in ("vllm", "bedrock"):
        raise ConfigError(
            f"llm.provider must be 'vllm' or 'bedrock', got {llm_provider!r}"
        )
    llm_model = _require_str(_get(llm, "model", "llm_model"), "llm.model")
    llm_region = _require_str(_get(llm, "region", "llm_region"), "llm.region")

    # --- training --------------------------------------------------------- #
    env_name = _require_str(
        _get(training, "env_name", "env_name"), "training.env_name",
    )
    train_epochs = _require_int(
        _get(training, "epochs", "train_epochs"),
        "training.epochs", minimum=1,
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
    xla_python_client_mem_fraction = _require_float(
        _get(training, "xla_python_client_mem_fraction",
             "xla_python_client_mem_fraction"),
        "training.xla_python_client_mem_fraction",
        minimum=0.0, maximum=1.0, exclusive_min=True,
    )
    learning_rate = _require_float(
        _get(training, "learning_rate", "learning_rate"),
        "training.learning_rate", minimum=0.0, exclusive_min=True,
    )
    episode_length = _require_int(
        _get(training, "episode_length", "episode_length"),
        "training.episode_length", minimum=1,
    )
    num_evals = _require_int(
        _get(training, "num_evals", "num_evals"),
        "training.num_evals", minimum=1,
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
    fall_terminate = bool(_get(training, "fall_terminate", "fall_terminate"))
    fall_height_m = _require_float(
        _get(training, "fall_height_m", "fall_height_m"),
        "training.fall_height_m", minimum=0.0, exclusive_min=True,
    )
    standing_reward = _require_float(
        _get(training, "standing_reward", "standing_reward"),
        "training.standing_reward", minimum=0.0,
    )
    warm_start = bool(_get(training, "warm_start", "warm_start"))

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
    # GPU 0 is reserved for a LOCAL Language_Model (self-hosted vLLM on the box;
    # Req 18.5). When the LLM is a managed off-box endpoint (provider="bedrock"),
    # no GPU is reserved, so training may use all GPUs including 0.
    local_llm = llm_provider == "vllm"
    if local_llm and LLM_GPU in training_gpus:
        raise ConfigError(
            f"training.training_gpus must not include GPU {LLM_GPU}, which is "
            f"reserved for the local language model (Req 18.5); got {training_gpus}. "
            f"Set llm.provider=bedrock to free GPU 0 for training."
        )
    if local_llm and "llm_gpu" in training:
        llm_gpu = _require_int(training["llm_gpu"], "training.llm_gpu", minimum=0)
        if llm_gpu in training_gpus:
            raise ConfigError(
                f"training.training_gpus must not include the reserved llm_gpu "
                f"({llm_gpu}); got {training_gpus}"
            )

    if oom_fallback_envs > num_envs:
        raise ConfigError(
            f"training.oom_fallback_num_envs ({oom_fallback_envs}) must be <= "
            f"training.num_envs ({num_envs})"
        )

    return Config(
        max_iterations=max_iterations,
        qwen_max_retries=qwen_max_retries,
        llm_endpoint=llm_endpoint,
        llm_provider=llm_provider,
        llm_model=llm_model,
        llm_region=llm_region,
        train_epochs=train_epochs,
        checkpoint_interval=checkpoint_interval,
        training_gpus=training_gpus,
        num_envs=num_envs,
        oom_fallback_envs=oom_fallback_envs,
        lr_reduction_factor=lr_reduction_factor,
        env_name=env_name,
        xla_python_client_mem_fraction=xla_python_client_mem_fraction,
        learning_rate=learning_rate,
        episode_length=episode_length,
        num_evals=num_evals,
        fall_terminate=fall_terminate,
        fall_height_m=fall_height_m,
        standing_reward=standing_reward,
        warm_start=warm_start,
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
    """Resolve capture resolution from ``width``/``height`` keys or the default."""
    if "width" not in capture and "height" not in capture:
        return DEFAULTS["capture_resolution"]
    default_w, default_h = DEFAULTS["capture_resolution"]
    width = capture.get("width", default_w)
    height = capture.get("height", default_h)
    return (width, height)
