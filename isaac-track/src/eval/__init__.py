"""Evaluation subpackage for the YOUR_REPO system.

Hosts the goal-reaching metric data models (:mod:`src.eval.metrics`) and, in
later Phase 2 tasks, the Evaluator that computes them from a trained policy.

The metric data models are intentionally pure Python (no Isaac Sim / torch
dependency) so they can be serialized, persisted, and unit-tested on the
controller host without GPUs.
"""

from __future__ import annotations

from .evaluator import (
    EpisodeMetrics,
    EpisodeTrajectory,
    compute_episode_metrics,
    compute_eval_metrics,
)
from .metrics import CapabilityGates, EvalMetrics

__all__ = [
    "EvalMetrics",
    "CapabilityGates",
    "EpisodeTrajectory",
    "EpisodeMetrics",
    "compute_episode_metrics",
    "compute_eval_metrics",
]
