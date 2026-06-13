"""Goal-reaching evaluation metric data models (spec Req 9, 17).

This module defines the JSON-serializable result types produced by the
Evaluator:

  - :class:`CapabilityGates` -- the staged goal-reaching gates (Req 17).
  - :class:`EvalMetrics` -- the full set of goal-reaching measurements plus the
    nested gates (Req 9), with exact ``to_json`` / ``from_json`` round-trip
    (Req 9.7).

Design references:
  - design.md -> Data Models -> ``@dataclass EvalMetrics`` / ``CapabilityGates``
  - requirements.md -> Requirement 9 (goal-reaching evaluation metrics)
  - requirements.md -> Requirement 17 (staged goal-reaching capability gating)

These models are intentionally pure Python: no Isaac Sim / torch import, so they
can be constructed, serialized, persisted to S3, and unit-tested on the
controller host without GPUs. The metric *computation* (running a policy and
filling these fields) is a separate Phase 2 task (11.2/11.3); this task owns the
data shape and the JSON round-trip only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from typing import Any, Mapping

from ..exceptions import ValidationError

__all__ = ["CapabilityGates", "EvalMetrics"]


# --------------------------------------------------------------------------- #
# Staged goal-reaching gates (Req 17)
# --------------------------------------------------------------------------- #
@dataclass
class CapabilityGates:
    """Staged goal-reaching capability gates (Req 17).

    The gates are *staged-monotonic*: a higher gate implies the lower ones
    (``efficient_goal`` => ``reaches_goal`` => ``makes_progress``). This class is
    a pure data carrier; the Evaluator (Task 11.3) is responsible for computing
    the boolean values from Config-sourced thresholds.
    """

    makes_progress: bool  # net displacement toward Goal > min_progress_distance (Req 17.1)
    reaches_goal: bool    # within Success_Radius without falling                (Req 17.2)
    efficient_goal: bool  # reaches_goal AND time-to-goal <= threshold AND
    #                       path_efficiency >= threshold                          (Req 17.3)

    def to_dict(self) -> dict[str, Any]:
        """Return a plain ``dict`` view (used by the JSON serializer)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CapabilityGates":
        """Build a :class:`CapabilityGates` from a mapping, validating shape/types."""
        return cls(
            makes_progress=_require_bool(data, "makes_progress"),
            reaches_goal=_require_bool(data, "reaches_goal"),
            efficient_goal=_require_bool(data, "efficient_goal"),
        )


# --------------------------------------------------------------------------- #
# Full goal-reaching metrics (Req 9)
# --------------------------------------------------------------------------- #
@dataclass
class EvalMetrics:
    """Goal-reaching measurements for an evaluated policy (Req 9).

    Field semantics (and the invariants the Evaluator must uphold, documented
    here so the data shape is self-describing):

      - ``distance_to_goal_m``  -- distance to Goal at episode end, meters,
        non-negative (Req 9.1).
      - ``success``             -- within Success_Radius without falling (Req 9.2).
      - ``success_rate``        -- mean success indicator across episodes, in
        ``[0, 1]`` (Req 9.2).
      - ``time_to_goal_s``      -- seconds to reach the Goal; present only for
        successful episodes, otherwise ``None`` (Req 9.3).
      - ``path_efficiency``     -- straight-line A->Goal distance / actual path
        length, in ``(0, 1]`` (Req 9.4).
      - ``upright_time_s``      -- seconds upright, non-negative and
        ``<=`` episode length (Req 9.5).
      - ``fall_rate``           -- mean fall indicator across episodes, in
        ``[0, 1]`` (Req 9.5).

    Optional secondary quality metrics (Req 9.6), each ``None`` when the
    secondary-metrics feature is disabled:

      - ``avg_forward_speed_mps`` -- average forward speed along the path.
      - ``energy_efficiency``     -- reward per squared torque.
      - ``gait_smoothness``       -- gait smoothness score.
      - ``symmetry_score``        -- left/right gait symmetry score.

    ``gates`` carries the staged goal gates (Req 17). ``to_json`` / ``from_json``
    provide an exact round-trip (Req 9.7).
    """

    # primary goal-reaching metrics (Req 9.1-9.5)
    distance_to_goal_m: float
    success: bool
    success_rate: float
    time_to_goal_s: float | None
    path_efficiency: float
    upright_time_s: float
    fall_rate: float
    # staged goal gates (Req 17)
    gates: CapabilityGates
    # optional secondary quality metrics (Req 9.6)
    avg_forward_speed_mps: float | None = None
    energy_efficiency: float | None = None
    gait_smoothness: float | None = None
    symmetry_score: float | None = None

    # ----------------------------------------------------------------- #
    # Serialization (Req 9.7)
    # ----------------------------------------------------------------- #
    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-ready ``dict`` with the nested gates expanded."""
        data: dict[str, Any] = {}
        for f in fields(self):
            value = getattr(self, f.name)
            if f.name == "gates":
                data[f.name] = value.to_dict()
            else:
                data[f.name] = value
        return data

    def to_json(self, *, indent: int | None = None) -> str:
        """Serialize to a JSON string (Req 9.7).

        ``sort_keys`` is enabled so the output is stable/deterministic, which
        makes artifacts diffable and the round-trip easy to assert on.
        """
        return json.dumps(self.to_dict(), sort_keys=True, indent=indent)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "EvalMetrics":
        """Build an :class:`EvalMetrics` from a mapping, validating shape/types."""
        if not isinstance(data, Mapping):
            raise ValidationError(
                f"EvalMetrics: expected a JSON object, got {type(data).__name__}"
            )

        gates_raw = data.get("gates")
        if gates_raw is None:
            raise ValidationError("EvalMetrics: missing required field 'gates'")
        if not isinstance(gates_raw, Mapping):
            raise ValidationError(
                f"EvalMetrics.gates: expected an object, got {type(gates_raw).__name__}"
            )

        return cls(
            distance_to_goal_m=_require_number(data, "distance_to_goal_m"),
            success=_require_bool(data, "success"),
            success_rate=_require_number(data, "success_rate"),
            time_to_goal_s=_optional_number(data, "time_to_goal_s"),
            path_efficiency=_require_number(data, "path_efficiency"),
            upright_time_s=_require_number(data, "upright_time_s"),
            fall_rate=_require_number(data, "fall_rate"),
            gates=CapabilityGates.from_dict(gates_raw),
            avg_forward_speed_mps=_optional_number(data, "avg_forward_speed_mps"),
            energy_efficiency=_optional_number(data, "energy_efficiency"),
            gait_smoothness=_optional_number(data, "gait_smoothness"),
            symmetry_score=_optional_number(data, "symmetry_score"),
        )

    @classmethod
    def from_json(cls, s: str) -> "EvalMetrics":
        """Deserialize from a JSON string, inverting :meth:`to_json` (Req 9.7).

        Raises :class:`~src.exceptions.ValidationError` when the payload is not
        valid JSON or does not match the expected shape/types.
        """
        try:
            data = json.loads(s)
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"EvalMetrics: invalid JSON ({exc})") from exc
        return cls.from_dict(data)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _require_bool(data: Mapping[str, Any], field: str) -> bool:
    if field not in data:
        raise ValidationError(f"missing required field {field!r}")
    value = data[field]
    if not isinstance(value, bool):
        raise ValidationError(
            f"{field}: expected a boolean, got {type(value).__name__} ({value!r})"
        )
    return value


def _require_number(data: Mapping[str, Any], field: str) -> float:
    if field not in data:
        raise ValidationError(f"missing required field {field!r}")
    return _coerce_number(data[field], field)


def _optional_number(data: Mapping[str, Any], field: str) -> float | None:
    value = data.get(field)
    if value is None:
        return None
    return _coerce_number(value, field)


def _coerce_number(value: Any, field: str) -> float:
    # bool is a subclass of int but is never an acceptable numeric metric value.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationError(
            f"{field}: expected a number, got {type(value).__name__} ({value!r})"
        )
    return float(value)
