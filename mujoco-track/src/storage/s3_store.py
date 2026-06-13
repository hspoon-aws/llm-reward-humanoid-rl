"""S3_Store: boto3-backed artifact persistence for the YOUR_REPO
system (spec Req 11, 19, 20, 21).

This module wraps Amazon S3 for the Orchestrator. It is responsible for two
distinct concerns the design assigns to the S3_Store:

  1. **Iteration-identifying path logic (Req 11.2).** Every artifact for an
     iteration lives under a path that embeds that iteration's index, e.g.
     ``s3://<bucket>/<prefix>/iteration-05/...``. Distinct iteration indices
     MUST map to distinct paths (path injectivity — Property 16). The
     Best_Policy is written to a *stable, well-known* path independent of any
     iteration (Req 19.3), and the Blog is written to the configured Blog
     location/format (Req 21.3).

  2. **Fail-soft persistence (Req 11.3).** A run is an unattended 24-hour GPU
     sprint; a transient S3 failure must NOT abort the loop. On any upload
     failure the local copy is retained (this layer never deletes local files)
     and a :class:`PersistResult` recording the failure is returned rather than
     raising fatally. ``src.exceptions.PersistError`` is provided for the rare
     caller that wants a typed error, but the primary contract here is the
     returned result, not a raised exception.

Design references:
  - design.md -> Components and Interfaces -> S3_Store (`src/storage/s3_store.py`)
  - design.md -> Data Models (PersistResult, CheckpointRef, IterationArtifacts,
    TrainingCapture, Blog, LoopCheckpoint, BestPolicyRef)
  - design.md -> Correctness Properties -> Property 16 (artifact path injectivity),
    Property 17 (persist-failure local retention)
  - requirements.md -> Requirements 11.1, 11.2, 11.3, 19.3, 20.4, 21.3

Testability
-----------
The boto3 client is **injectable**: pass any object exposing
``put_object(Bucket=, Key=, Body=, **kwargs)`` and ``get_object(Bucket=, Key=)``.
This lets the property tests (Tasks 5.2, 5.3) exercise path logic and
persist-failure handling against an in-memory fake with no real AWS calls and
no boto3 import. ``boto3`` is only imported lazily when no client is injected.

Several richer data models referenced by the design (``LoopCheckpoint``,
``BestPolicyRef``, ``MetricsHistory``) are owned by other components/tasks. To
keep this module self-contained and testable now, the ``put_*`` methods accept
**either** the lightweight dataclasses defined here **or** duck-typed objects /
plain mappings with the same shape, and ``save_loop_checkpoint`` serializes any
dataclass / mapping / ``to_dict``-able state to JSON.
"""

from __future__ import annotations

import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import Any, Mapping

__all__ = [
    "PersistResult",
    "CheckpointRef",
    "IterationArtifacts",
    "TrainingCapture",
    "Blog",
    "S3Store",
]


# --------------------------------------------------------------------------- #
# Result + lightweight data models
# --------------------------------------------------------------------------- #
@dataclass
class PersistResult:
    """Outcome of a persistence attempt (Req 11.3).

    ``ok`` is the primary signal. On failure the caller still holds the local
    artifact (``local_path``); ``retained_local`` makes that retention explicit
    and ``error`` carries a descriptive message. ``path`` is the destination S3
    URI (a prefix for multi-file artifacts) when known.
    """

    ok: bool
    path: str | None = None
    error: str | None = None
    local_path: str | None = None
    retained_local: bool = False

    @classmethod
    def success(cls, path: str) -> "PersistResult":
        return cls(ok=True, path=path)

    @classmethod
    def failure(cls, error: str, *, path: str | None = None,
                local_path: str | None = None) -> "PersistResult":
        return cls(
            ok=False,
            path=path,
            error=error,
            local_path=local_path,
            retained_local=local_path is not None,
        )


@dataclass
class CheckpointRef:
    """Reference to a policy checkpoint produced by the PPO_Runner.

    Minimal local definition (the authoritative type binds in a later task);
    ``path`` is the local filesystem path to the checkpoint artifact.
    """

    path: str
    iteration_index: int | None = None


@dataclass
class IterationArtifacts:
    """The set of per-iteration artifacts to persist (Req 11.1).

    ``files`` maps an artifact *name* (its key suffix within the iteration
    path, e.g. ``"reward.py"``, ``"metrics.json"``, ``"train.log"``) to a local
    filesystem path. A mapping or any object exposing ``files`` is also accepted
    by :meth:`S3Store.put_iteration_artifacts`.
    """

    files: dict[str, str] = field(default_factory=dict)


@dataclass
class TrainingCapture:
    """A periodic training-process capture (Req 20)."""

    iteration_index: int
    training_epoch: int
    wall_clock_s: float = 0.0
    screenshot_path: str | None = None
    clip_path: str | None = None


@dataclass
class Blog:
    """Assembled learning-blog deliverable (Req 21)."""

    files: dict[str, str] = field(default_factory=dict)
    output_format: str = "markdown"


# --------------------------------------------------------------------------- #
# Path helpers (pure; Req 11.2 / Property 16)
# --------------------------------------------------------------------------- #
def _format_index(index: int) -> str:
    """Format an iteration index into an unambiguous, injective token.

    Zero-padding to a minimum width keeps lexicographic ordering pleasant for
    small runs while remaining injective for *all* distinct integers: Python's
    integer formatting maps distinct ints to distinct strings (padding only adds
    leading zeros up to the minimum width and never collapses two values), so
    ``iteration-<token>`` is injective in ``index``. See Property 16.
    """
    return f"{int(index):02d}"


def _split_s3_uri(uri: str) -> tuple[str, str]:
    """Split an ``s3://bucket/prefix`` URI into ``(bucket, prefix)``.

    The returned prefix has no leading/trailing slashes. A bare ``s3://bucket``
    yields an empty prefix.
    """
    raw = uri.strip()
    if raw.startswith("s3://"):
        raw = raw[len("s3://"):]
    raw = raw.lstrip("/")
    if not raw:
        raise ValueError(f"invalid S3 location (no bucket): {uri!r}")
    bucket, _, prefix = raw.partition("/")
    if not bucket:
        raise ValueError(f"invalid S3 location (no bucket): {uri!r}")
    return bucket, prefix.strip("/")


def _join_key(*parts: str) -> str:
    """Join non-empty key segments with single slashes (no leading slash)."""
    cleaned = [p.strip("/") for p in parts if p is not None and p.strip("/") != ""]
    return "/".join(cleaned)


# --------------------------------------------------------------------------- #
# S3_Store
# --------------------------------------------------------------------------- #
class S3Store:
    """boto3-backed artifact store with iteration path logic and fail-soft puts.

    Parameters
    ----------
    s3_location:
        Base destination URI, ``s3://<bucket>/<prefix>`` (as produced by the
        Config loader's ``s3_location``). The ``<prefix>`` may be empty.
    run_id:
        Optional per-run identifier inserted between the base prefix and the
        per-iteration segment, yielding the design's scheme
        ``s3://<bucket>/<prefix>/<run-id>/iteration-NN/...`` (Req 11.2). It is
        constant for the life of the store, so iteration-path injectivity over
        the iteration index is unaffected. When ``None`` the run-id segment is
        omitted (``s3://<bucket>/<prefix>/iteration-NN/...``).
    client:
        Optional injected S3 client exposing ``put_object``/``get_object``. When
        omitted, a real boto3 ``s3`` client is created lazily on first use.
    region:
        AWS region for the lazily-created boto3 client (ignored if ``client`` is
        injected).
    project_tag:
        Optional tag value applied as ``project=<project_tag>`` to created
        objects (the run_config's ``project_tag``). Omitted from the request
        when ``None``.
    best_policy_prefix:
        Stable, well-known sub-path for the Best_Policy (Req 19.3). Defaults to
        ``"best_policy"`` under the base prefix — independent of any iteration.
    blog_location:
        Optional override S3 URI for the assembled Blog (Req 21.3). Defaults to
        ``<base>/blog``.
    loop_checkpoint_key:
        Sub-path (relative to the base prefix) for the resumable loop checkpoint
        (Req 16.2). Defaults to ``"loop_checkpoint.json"``.
    """

    def __init__(
        self,
        s3_location: str,
        *,
        run_id: str | None = None,
        client: Any | None = None,
        region: str | None = None,
        project_tag: str | None = None,
        best_policy_prefix: str = "best_policy",
        blog_location: str | None = None,
        loop_checkpoint_key: str = "loop_checkpoint.json",
    ) -> None:
        self._bucket, base_prefix = _split_s3_uri(s3_location)
        self._run_id = run_id.strip("/") if run_id else None
        # The per-run segment lives under the base prefix and above the
        # per-iteration segment (design scheme: <prefix>/<run-id>/iteration-NN).
        self._prefix = _join_key(base_prefix, self._run_id) if self._run_id else base_prefix
        self._client = client
        self._region = region
        self._project_tag = project_tag
        self._best_policy_prefix = best_policy_prefix.strip("/")
        self._loop_checkpoint_key = loop_checkpoint_key.strip("/")
        if blog_location is not None:
            self._blog_bucket, self._blog_prefix = _split_s3_uri(blog_location)
        else:
            self._blog_bucket = self._bucket
            self._blog_prefix = _join_key(self._prefix, "blog")

    # ------------------------------------------------------------------ #
    # Path logic (pure, public for testing — Req 11.2 / Property 16)
    # ------------------------------------------------------------------ #
    @property
    def bucket(self) -> str:
        return self._bucket

    @property
    def base_prefix(self) -> str:
        return self._prefix

    @property
    def run_id(self) -> str | None:
        return self._run_id

    def iteration_key_prefix(self, index: int) -> str:
        """Return the iteration-identifying *key prefix* (no bucket, no scheme).

        Embeds ``index`` and is injective in ``index`` (Property 16).
        """
        return _join_key(self._prefix, f"iteration-{_format_index(index)}")

    def iteration_path(self, index: int) -> str:
        """Return the full ``s3://`` URI for an iteration (Req 11.2)."""
        return self._uri(self._bucket, self.iteration_key_prefix(index))

    def best_policy_key_prefix(self) -> str:
        """Stable, well-known key prefix for the Best_Policy (Req 19.3)."""
        return _join_key(self._prefix, self._best_policy_prefix)

    def best_policy_path(self) -> str:
        return self._uri(self._bucket, self.best_policy_key_prefix())

    # ------------------------------------------------------------------ #
    # Public persistence API (design.md S3_Store interface)
    # ------------------------------------------------------------------ #
    def put_iteration_artifacts(
        self, index: int, artifacts: "IterationArtifacts | Mapping[str, str] | Any"
    ) -> PersistResult:
        """Persist an iteration's artifacts under its iteration path (Req 11.1, 11.2).

        On any per-file failure the local copies are retained and an aggregate
        failure :class:`PersistResult` is returned rather than raising (Req 11.3).
        """
        files = self._artifact_files(artifacts)
        dest_prefix = self.iteration_key_prefix(index)
        dest_uri = self._uri(self._bucket, dest_prefix)
        if not files:
            return PersistResult.success(dest_uri)
        return self._upload_files(files, dest_prefix, dest_uri)

    def put_training_capture(
        self, index: int, capture: "TrainingCapture | Mapping[str, Any] | Any"
    ) -> PersistResult:
        """Persist a Training_Capture under the iteration path (Req 20.4).

        The capture's training epoch is embedded in each object key so captures
        order deterministically for the Blog narrative (Req 20.2).
        """
        epoch = _get(capture, "training_epoch", 0)
        screenshot = _get(capture, "screenshot_path", None)
        clip = _get(capture, "clip_path", None)
        dest_prefix = _join_key(self.iteration_key_prefix(index), "captures")
        dest_uri = self._uri(self._bucket, dest_prefix)

        files: dict[str, str] = {}
        if screenshot:
            files[f"epoch-{int(epoch)}-{os.path.basename(screenshot)}"] = screenshot
        if clip:
            files[f"epoch-{int(epoch)}-{os.path.basename(clip)}"] = clip
        if not files:
            return PersistResult.success(dest_uri)
        return self._upload_files(files, dest_prefix, dest_uri)

    def put_best_policy(
        self, checkpoint: "CheckpointRef | Mapping[str, Any] | str | Any"
    ) -> PersistResult:
        """Write the Best_Policy to a stable, well-known path (Req 19.3)."""
        local_path = checkpoint if isinstance(checkpoint, str) else _get(checkpoint, "path", None)
        if not local_path:
            return PersistResult.failure(
                "best policy checkpoint has no local path", path=self.best_policy_path()
            )
        dest_prefix = self.best_policy_key_prefix()
        key = _join_key(dest_prefix, os.path.basename(local_path))
        return self._upload_one(local_path, key)

    def put_blog(self, blog: "Blog | Mapping[str, str] | Any") -> PersistResult:
        """Persist the assembled Blog + assets at the configured location (Req 21.3)."""
        files = self._artifact_files(blog)
        dest_uri = self._uri(self._blog_bucket, self._blog_prefix)
        if not files:
            return PersistResult.success(dest_uri)
        return self._upload_files(
            files, self._blog_prefix, dest_uri, bucket=self._blog_bucket
        )

    def save_loop_checkpoint(self, state: Any) -> None:
        """Persist the resumable loop checkpoint as JSON (Req 16.2).

        Fail-soft: a transient failure here must not abort the unattended loop,
        so this method never raises on an upload error (the in-memory loop state
        is unchanged and the next iteration will retry).
        """
        key = _join_key(self._prefix, self._loop_checkpoint_key)
        try:
            payload = json.dumps(_to_serializable(state), sort_keys=True).encode("utf-8")
        except (TypeError, ValueError):
            # Non-serializable state: nothing we can persist; do not crash the loop.
            return
        try:
            self._put_bytes(key, payload)
        except Exception:  # noqa: BLE001 - fail-soft persistence (Req 11.3 spirit)
            return

    def load_loop_checkpoint(self) -> dict[str, Any] | None:
        """Load the resumable loop checkpoint, or ``None`` if absent (Req 16.2).

        Returns the parsed JSON object. A richer typed ``LoopCheckpoint`` is
        reconstructed by the Orchestrator in a later task. Any read failure
        (missing key, transport error) is treated as "no checkpoint" so resume
        degrades gracefully to a fresh start.
        """
        key = _join_key(self._prefix, self._loop_checkpoint_key)
        try:
            client = self._client_or_default()
            response = client.get_object(Bucket=self._bucket, Key=key)
        except Exception:  # noqa: BLE001 - absent/unreadable -> no checkpoint
            return None
        try:
            body = response["Body"].read() if hasattr(response.get("Body"), "read") else response["Body"]
            if isinstance(body, (bytes, bytearray)):
                body = body.decode("utf-8")
            return json.loads(body)
        except Exception:  # noqa: BLE001 - malformed checkpoint -> ignore
            return None

    # ------------------------------------------------------------------ #
    # Internals
    # ------------------------------------------------------------------ #
    def _uri(self, bucket: str, key: str) -> str:
        return f"s3://{bucket}/{key}" if key else f"s3://{bucket}"

    def _client_or_default(self) -> Any:
        """Return the injected client, or lazily build a real boto3 S3 client."""
        if self._client is None:
            import boto3  # local import: only required when no client is injected

            self._client = boto3.client("s3", region_name=self._region)
        return self._client

    @staticmethod
    def _artifact_files(artifacts: Any) -> dict[str, str]:
        """Coerce an artifacts container into a ``{name: local_path}`` mapping."""
        if artifacts is None:
            return {}
        files = getattr(artifacts, "files", None)
        if files is None and isinstance(artifacts, Mapping):
            files = artifacts
        if files is None:
            return {}
        return {str(name): str(path) for name, path in dict(files).items()}

    def _upload_files(
        self,
        files: Mapping[str, str],
        dest_prefix: str,
        dest_uri: str,
        *,
        bucket: str | None = None,
    ) -> PersistResult:
        """Upload several local files under ``dest_prefix``; aggregate the result.

        Every file is attempted even if an earlier one fails; local copies are
        always retained. Returns one aggregate :class:`PersistResult` whose
        ``ok`` is true only when all uploads succeed (Req 11.3).
        """
        errors: list[str] = []
        for name, local_path in files.items():
            key = _join_key(dest_prefix, name)
            result = self._upload_one(local_path, key, bucket=bucket)
            if not result.ok:
                errors.append(f"{name}: {result.error}")
        if errors:
            return PersistResult.failure(
                "; ".join(errors), path=dest_uri, local_path=dest_prefix
            )
        return PersistResult.success(dest_uri)

    def _upload_one(
        self, local_path: str, key: str, *, bucket: str | None = None
    ) -> PersistResult:
        """Read a local file and upload it; never raise (Req 11.3)."""
        target_bucket = bucket or self._bucket
        dest_uri = self._uri(target_bucket, key)
        try:
            with open(local_path, "rb") as handle:
                data = handle.read()
        except OSError as exc:
            return PersistResult.failure(
                f"cannot read local artifact: {exc}", path=dest_uri, local_path=local_path
            )
        try:
            self._put_bytes(key, data, bucket=target_bucket)
        except Exception as exc:  # noqa: BLE001 - fail-soft: retain local (Req 11.3)
            return PersistResult.failure(
                f"upload failed: {exc}", path=dest_uri, local_path=local_path
            )
        return PersistResult.success(dest_uri)

    def _put_bytes(self, key: str, data: bytes, *, bucket: str | None = None) -> None:
        """Low-level ``put_object`` with optional project tagging."""
        client = self._client_or_default()
        kwargs: dict[str, Any] = {
            "Bucket": bucket or self._bucket,
            "Key": key,
            "Body": data,
        }
        if self._project_tag:
            kwargs["Tagging"] = f"project={self._project_tag}"
        client.put_object(**kwargs)


# --------------------------------------------------------------------------- #
# Serialization helpers
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any) -> Any:
    """Attribute-or-key accessor for duck-typed inputs (object or mapping)."""
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _to_serializable(state: Any) -> Any:
    """Best-effort conversion of loop state into JSON-serializable data."""
    if dataclasses.is_dataclass(state) and not isinstance(state, type):
        return dataclasses.asdict(state)
    if hasattr(state, "to_dict") and callable(state.to_dict):
        return state.to_dict()
    if isinstance(state, Mapping):
        return {str(k): _to_serializable(v) for k, v in state.items()}
    if isinstance(state, (list, tuple)):
        return [_to_serializable(v) for v in state]
    return state
