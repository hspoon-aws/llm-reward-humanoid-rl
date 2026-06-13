"""Learning-blog assembly (spec Req 21).

A second first-class deliverable of this project is a *learning Blog* that
documents two things the run teaches: how to self-host the Qwen3-Coder language
model via vLLM, and how to run physical-AI simulation in NVIDIA Isaac Lab (the
H1 humanoid goal-reaching environment and the Eureka-style LLM→Reward→RL loop).

This module assembles that Blog when a run completes (Req 21.1) from the
artifacts the loop produced:

  - per-iteration narrative covering the Language_Model self-hosting setup, the
    Isaac Lab simulation setup, the reward evolution across Iterations, and the
    progression of the humanoid's goal-reaching behavior (Req 21.1);
  - references to the captured Training_Captures and demo videos, the
    Metrics_History, and the per-Iteration reward-code snapshots (Req 21.2);
  - optional Language_Model-drafted narrative requested from the Qwen_Client
    using the Metrics_History and reward-code snapshots when LLM narrative is
    enabled (Req 21.4).

The assembled Blog is persisted to the configured Blog output location and in
the configured output format via the existing :class:`src.storage.s3_store.S3Store`
(Req 21.3).

Design references:
  - design.md -> Components and Interfaces -> Orchestrator ("assembles the Blog
    from captured assets and per-iteration narrative ... persists it to the
    configured Blog location/format (Req 21.1-21.4)")
  - design.md -> Qwen_Client.draft_blog_narrative (Req 21.4)
  - requirements.md -> Requirement 21

Design notes
------------
This is **pure-Python assembly** with no Isaac Sim / vLLM dependency: it turns
already-captured artifacts into a document and hands the bytes to the S3_Store.
Inputs are accepted as the lightweight dataclasses defined here **or** as any
duck-typed object / mapping exposing the same fields (mirroring the S3_Store's
tolerant input handling), so the Orchestrator's own ``IterationRecord``,
``MetricsHistory``, and ``TrainingCapture`` types (owned by other tasks) drop in
without a hard import dependency.

LLM-drafted narrative (Req 21.4) is treated as an *enhancement*: if the
Qwen_Client call fails the assembler falls back to deterministic templated
narrative rather than aborting the Blog, consistent with the unattended-run
fail-soft philosophy.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from src.storage.s3_store import Blog, PersistResult, S3Store

__all__ = [
    "BlogAsset",
    "AssembledBlog",
    "BlogAssembler",
    "assemble_blog",
]


# --------------------------------------------------------------------------- #
# Data models
# --------------------------------------------------------------------------- #
@dataclass
class BlogAsset:
    """A captured or generated artifact referenced by the Blog (Req 21.2).

    ``name`` is the human/file label used in the Blog body and as the key
    suffix when the asset is uploaded alongside the Blog. ``local_path`` is the
    on-disk path to the asset (used both to reference it and to upload it).
    ``kind`` categorizes the asset for section grouping (e.g. ``"screenshot"``,
    ``"clip"``, ``"video"``, ``"reward"``, ``"metrics"``).
    """

    name: str
    local_path: str
    kind: str = "asset"
    iteration_index: int | None = None
    caption: str | None = None


@dataclass
class AssembledBlog:
    """The in-memory result of assembling the Blog before persistence.

    ``document`` is the rendered Blog text (markdown or html). ``filename`` is
    the document's name within the Blog output location. ``assets`` are the
    referenced Blog_Assets to upload alongside the document. ``blog`` is the
    :class:`src.storage.s3_store.Blog` handed to ``S3Store.put_blog`` (Req 21.3).
    """

    document: str
    filename: str
    output_format: str
    assets: list[BlogAsset] = field(default_factory=list)
    blog: Blog | None = None


# --------------------------------------------------------------------------- #
# Duck-typed accessors (object-or-mapping), mirroring s3_store._get
# --------------------------------------------------------------------------- #
def _get(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _records_of(metrics_history: Any) -> list[Any]:
    """Extract the iteration records from a MetricsHistory-like input.

    Accepts an object exposing ``records``, a mapping with a ``records`` key, or
    a bare sequence of records.
    """
    if metrics_history is None:
        return []
    records = _get(metrics_history, "records", None)
    if records is None and isinstance(metrics_history, Sequence) and not isinstance(
        metrics_history, (str, bytes)
    ):
        records = metrics_history
    return list(records) if records else []


def _render_history_text(metrics_history: Any) -> str:
    """Render Metrics_History as text for the Blog (Req 21.2).

    Prefers the MetricsHistory's own ``render_for_prompt`` when available so the
    Blog reuses the exact textual feedback the loop fed to the model; otherwise
    falls back to a per-record summary.
    """
    render = _get(metrics_history, "render_for_prompt", None)
    if callable(render):
        try:
            text = render()
            if text:
                return str(text)
        except Exception:  # noqa: BLE001 - never let rendering abort the Blog
            pass
    lines: list[str] = []
    for rec in _records_of(metrics_history):
        idx = _get(rec, "index", "?")
        metrics = _get(rec, "metrics", None)
        summary = _summarize_metrics(metrics)
        lines.append(f"- Iteration {idx}: {summary}")
    return "\n".join(lines) if lines else "_No metrics history was recorded._"


def _summarize_metrics(metrics: Any) -> str:
    """One-line, defensive summary of an EvalMetrics-like value."""
    if metrics is None:
        return "no metrics"
    success_rate = _get(metrics, "success_rate", None)
    distance = _get(metrics, "distance_to_goal_m", None)
    fall_rate = _get(metrics, "fall_rate", None)
    parts: list[str] = []
    if success_rate is not None:
        parts.append(f"success_rate={success_rate}")
    if distance is not None:
        parts.append(f"distance_to_goal_m={distance}")
    if fall_rate is not None:
        parts.append(f"fall_rate={fall_rate}")
    return ", ".join(parts) if parts else "metrics recorded"


# --------------------------------------------------------------------------- #
# Blog assembler
# --------------------------------------------------------------------------- #
class BlogAssembler:
    """Assemble and persist the learning Blog (Req 21).

    Parameters
    ----------
    output_format:
        The configured Blog output format (``Config.blog_output_format``), e.g.
        ``"markdown"`` or ``"html"`` (Req 18.6, 21.3).
    llm_narrative_enabled:
        Whether to request narrative sections from the Qwen_Client (Req 21.4).
    qwen_client:
        Optional Qwen_Client exposing ``draft_blog_narrative(metrics_history,
        reward_snapshots) -> str``. Required only when ``llm_narrative_enabled``.
    title:
        Blog title.
    document_name:
        Base name of the rendered Blog document (extension is derived from the
        output format).
    """

    _LLM_SETUP_DEFAULT = (
        "The Language_Model is **Qwen3-Coder-30B-A3B**, self-hosted with "
        "**vLLM** on a single GPU (GPU 0) and exposed over an OpenAI-compatible "
        "HTTP endpoint. GPU 0 is reserved exclusively for serving; the training "
        "GPUs are kept disjoint from it so reward generation and PPO training "
        "co-exist on one host."
    )
    _ISAAC_SETUP_DEFAULT = (
        "Physical-AI simulation runs in **NVIDIA Isaac Lab** on the manager-based "
        "Unitree H1 flat-terrain task, reframed from velocity-command tracking "
        "into **point-to-point goal-reaching**: the robot starts at point A and "
        "must reach a configurable Goal (point B) within a success radius while "
        "staying upright. A goal-conditioned observation is appended to the "
        "proprioceptive policy input, and the LLM-generated reward shapes the "
        "policy via the Eureka-style LLM→Reward→RL loop."
    )

    def __init__(
        self,
        *,
        output_format: str = "markdown",
        llm_narrative_enabled: bool = False,
        qwen_client: Any | None = None,
        title: str = "Teaching a Humanoid to Walk: A Self-Hosted LLM + Isaac Lab Learning Log",
        document_name: str = "blog",
    ) -> None:
        self._output_format = (output_format or "markdown").strip().lower()
        self._llm_narrative_enabled = bool(llm_narrative_enabled)
        self._qwen_client = qwen_client
        self._title = title
        self._document_name = document_name

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def assemble(
        self,
        *,
        iteration_records: Sequence[Any] | None = None,
        metrics_history: Any | None = None,
        training_captures: Sequence[Any] | None = None,
        demo_videos: Sequence[Any] | None = None,
        llm_setup_notes: str | None = None,
        isaac_setup_notes: str | None = None,
        output_dir: str | None = None,
    ) -> AssembledBlog:
        """Assemble the Blog document + asset references (Req 21.1, 21.2, 21.4).

        Iteration records may be passed explicitly or sourced from
        ``metrics_history``. The rendered document is written to ``output_dir``
        (a temp dir is created when omitted) so the :class:`S3Store` can read
        and upload it; the returned :class:`AssembledBlog` carries the
        :class:`Blog` ready for :meth:`persist`.
        """
        records = list(iteration_records) if iteration_records is not None else _records_of(
            metrics_history
        )
        captures = list(training_captures or [])
        videos = list(demo_videos or [])

        reward_snapshots = self._reward_snapshots(records)

        narrative = self._draft_narrative(metrics_history, reward_snapshots)

        sections: list[str] = []
        sections.append(self._render_heading(self._title, level=1))
        sections.append(
            "_An automatically assembled learning log from an unattended "
            "Eureka-style LLM→Reward→RL run._"
        )

        # Req 21.1 — LLM self-hosting + Isaac Lab setup narrative.
        sections.append(self._render_heading("Self-Hosting the Language Model", level=2))
        sections.append(llm_setup_notes or self._LLM_SETUP_DEFAULT)
        sections.append(self._render_heading("Running Isaac Lab Simulation", level=2))
        sections.append(isaac_setup_notes or self._ISAAC_SETUP_DEFAULT)

        # Req 21.4 — optional LLM-drafted narrative.
        if narrative:
            sections.append(self._render_heading("Narrative", level=2))
            sections.append(narrative)

        # Req 21.1, 21.2 — reward evolution + goal-reaching progression across
        # iterations, referencing per-iteration reward-code snapshots.
        sections.append(self._render_heading("Reward Evolution Across Iterations", level=2))
        sections.append(self._render_iterations(records))

        # Req 21.2 — Metrics_History reference.
        sections.append(self._render_heading("Goal-Reaching Progression (Metrics History)", level=2))
        sections.append(_render_history_text(metrics_history) if metrics_history is not None
                        else self._render_history_from_records(records))

        # Req 21.2 — Training_Captures reference.
        sections.append(self._render_heading("Training Captures", level=2))
        capture_assets = self._capture_assets(captures)
        sections.append(self._render_asset_list(capture_assets, empty="_No training captures._"))

        # Req 21.2 — demo videos reference.
        sections.append(self._render_heading("Demo Videos", level=2))
        video_assets = self._video_assets(videos)
        sections.append(self._render_asset_list(video_assets, empty="_No demo videos._"))

        document = "\n\n".join(s for s in sections if s)

        assets: list[BlogAsset] = []
        assets.extend(capture_assets)
        assets.extend(video_assets)

        filename = f"{self._document_name}.{self._extension()}"
        local_doc_path = self._write_document(document, filename, output_dir)

        blog = self._build_blog(local_doc_path, filename, assets)

        return AssembledBlog(
            document=document,
            filename=filename,
            output_format=self._output_format,
            assets=assets,
            blog=blog,
        )

    def persist(self, store: S3Store, assembled: AssembledBlog) -> PersistResult:
        """Persist the assembled Blog to the configured location/format (Req 21.3)."""
        if assembled.blog is None:
            raise ValueError("AssembledBlog has no Blog to persist; call assemble() first")
        return store.put_blog(assembled.blog)

    def assemble_and_persist(
        self,
        store: S3Store,
        *,
        iteration_records: Sequence[Any] | None = None,
        metrics_history: Any | None = None,
        training_captures: Sequence[Any] | None = None,
        demo_videos: Sequence[Any] | None = None,
        llm_setup_notes: str | None = None,
        isaac_setup_notes: str | None = None,
        output_dir: str | None = None,
    ) -> tuple[AssembledBlog, PersistResult]:
        """Convenience: assemble (Req 21.1-21.2, 21.4) then persist (Req 21.3)."""
        assembled = self.assemble(
            iteration_records=iteration_records,
            metrics_history=metrics_history,
            training_captures=training_captures,
            demo_videos=demo_videos,
            llm_setup_notes=llm_setup_notes,
            isaac_setup_notes=isaac_setup_notes,
            output_dir=output_dir,
        )
        result = self.persist(store, assembled)
        return assembled, result

    # ------------------------------------------------------------------ #
    # Narrative (Req 21.4)
    # ------------------------------------------------------------------ #
    def _draft_narrative(self, metrics_history: Any, reward_snapshots: list[str]) -> str | None:
        """Request narrative from the Qwen_Client when enabled (Req 21.4).

        Fail-soft: if narrative is disabled, no client is available, or the call
        fails, return ``None`` so the deterministic templated sections still
        produce a complete Blog.
        """
        if not self._llm_narrative_enabled or self._qwen_client is None:
            return None
        draft = getattr(self._qwen_client, "draft_blog_narrative", None)
        if not callable(draft):
            return None
        try:
            text = draft(metrics_history, reward_snapshots)
        except Exception:  # noqa: BLE001 - narrative is an enhancement, never fatal
            return None
        return str(text) if text else None

    # ------------------------------------------------------------------ #
    # Rendering helpers
    # ------------------------------------------------------------------ #
    @staticmethod
    def _reward_snapshots(records: Sequence[Any]) -> list[str]:
        """Collect per-iteration reward-code snapshots (Req 21.2, 21.4 input)."""
        snapshots: list[str] = []
        for rec in records:
            code = _get(rec, "reward_code", None)
            if code:
                snapshots.append(str(code))
        return snapshots

    def _render_iterations(self, records: Sequence[Any]) -> str:
        """Render the per-iteration reward evolution with code snapshots (Req 21.2)."""
        if not records:
            return "_No iterations were recorded._"
        blocks: list[str] = []
        for rec in records:
            idx = _get(rec, "index", "?")
            status = _get(rec, "status", "completed")
            behavior = _get(rec, "behavior_description", "") or ""
            metrics = _get(rec, "metrics", None)
            code = _get(rec, "reward_code", None)

            block = [self._render_heading(f"Iteration {idx} ({status})", level=3)]
            if behavior:
                block.append(str(behavior))
            block.append(f"Metrics: {_summarize_metrics(metrics)}")
            if code:
                block.append(self._render_code(str(code)))
            else:
                block.append("_No reward-code snapshot for this iteration._")
            blocks.append("\n\n".join(block))
        return "\n\n".join(blocks)

    def _render_history_from_records(self, records: Sequence[Any]) -> str:
        if not records:
            return "_No metrics history was recorded._"
        lines = [
            f"- Iteration {_get(rec, 'index', '?')}: {_summarize_metrics(_get(rec, 'metrics', None))}"
            for rec in records
        ]
        return "\n".join(lines)

    def _capture_assets(self, captures: Sequence[Any]) -> list[BlogAsset]:
        """Turn Training_Capture-like inputs into referenceable Blog_Assets (Req 21.2)."""
        assets: list[BlogAsset] = []
        for cap in captures:
            idx = _get(cap, "iteration_index", None)
            epoch = _get(cap, "training_epoch", None)
            screenshot = _get(cap, "screenshot_path", None)
            clip = _get(cap, "clip_path", None)
            label = f"iter{idx}-epoch{epoch}" if idx is not None else "capture"
            if screenshot:
                assets.append(
                    BlogAsset(
                        name=f"{label}-{os.path.basename(str(screenshot))}",
                        local_path=str(screenshot),
                        kind="screenshot",
                        iteration_index=idx,
                        caption=f"Training capture (iteration {idx}, epoch {epoch})",
                    )
                )
            if clip:
                assets.append(
                    BlogAsset(
                        name=f"{label}-{os.path.basename(str(clip))}",
                        local_path=str(clip),
                        kind="clip",
                        iteration_index=idx,
                        caption=f"Training clip (iteration {idx}, epoch {epoch})",
                    )
                )
        return assets

    def _video_assets(self, videos: Sequence[Any]) -> list[BlogAsset]:
        """Turn demo-video inputs (asset, mapping, or bare path) into Blog_Assets (Req 21.2)."""
        assets: list[BlogAsset] = []
        for video in videos:
            if isinstance(video, BlogAsset):
                assets.append(video)
                continue
            if isinstance(video, str):
                path = video
                idx = None
                caption = None
            else:
                path = _get(video, "local_path", None) or _get(video, "path", None)
                idx = _get(video, "iteration_index", None)
                caption = _get(video, "caption", None)
            if not path:
                continue
            assets.append(
                BlogAsset(
                    name=os.path.basename(str(path)),
                    local_path=str(path),
                    kind="video",
                    iteration_index=idx,
                    caption=caption or "Demo video",
                )
            )
        return assets

    def _render_asset_list(self, assets: Sequence[BlogAsset], *, empty: str) -> str:
        if not assets:
            return empty
        if self._output_format == "html":
            items = "\n".join(
                f'  <li><a href="{a.name}">{a.caption or a.name}</a></li>' for a in assets
            )
            return f"<ul>\n{items}\n</ul>"
        return "\n".join(f"- [{a.caption or a.name}]({a.name})" for a in assets)

    def _render_heading(self, text: str, *, level: int) -> str:
        if self._output_format == "html":
            return f"<h{level}>{text}</h{level}>"
        return f"{'#' * level} {text}"

    def _render_code(self, code: str) -> str:
        if self._output_format == "html":
            escaped = code.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            return f"<pre><code>{escaped}</code></pre>"
        return f"```python\n{code}\n```"

    def _extension(self) -> str:
        return {"markdown": "md", "md": "md", "html": "html"}.get(self._output_format, "md")

    # ------------------------------------------------------------------ #
    # Persistence wiring
    # ------------------------------------------------------------------ #
    @staticmethod
    def _write_document(document: str, filename: str, output_dir: str | None) -> str:
        """Write the rendered Blog document to disk so the S3_Store can read it."""
        if output_dir is None:
            import tempfile

            output_dir = tempfile.mkdtemp(prefix="blog-")
        os.makedirs(output_dir, exist_ok=True)
        local_path = os.path.join(output_dir, filename)
        with open(local_path, "w", encoding="utf-8") as handle:
            handle.write(document)
        return local_path

    def _build_blog(
        self, local_doc_path: str, filename: str, assets: Sequence[BlogAsset]
    ) -> Blog:
        """Build the :class:`Blog` (document + assets) for ``S3Store.put_blog`` (Req 21.3)."""
        files: dict[str, str] = {filename: local_doc_path}
        for asset in assets:
            # Asset names are unique by construction (label + basename); keep the
            # first occurrence if a duplicate ever appears.
            files.setdefault(asset.name, asset.local_path)
        return Blog(files=files, output_format=self._output_format)


# --------------------------------------------------------------------------- #
# Functional entry point
# --------------------------------------------------------------------------- #
def assemble_blog(
    *,
    output_format: str = "markdown",
    llm_narrative_enabled: bool = False,
    qwen_client: Any | None = None,
    iteration_records: Sequence[Any] | None = None,
    metrics_history: Any | None = None,
    training_captures: Sequence[Any] | None = None,
    demo_videos: Sequence[Any] | None = None,
    llm_setup_notes: str | None = None,
    isaac_setup_notes: str | None = None,
    output_dir: str | None = None,
) -> AssembledBlog:
    """Assemble a Blog in one call (Req 21.1, 21.2, 21.4).

    Convenience wrapper around :class:`BlogAssembler` for callers that only need
    the assembled document and assets (persistence is a separate step via
    :meth:`BlogAssembler.persist`).
    """
    assembler = BlogAssembler(
        output_format=output_format,
        llm_narrative_enabled=llm_narrative_enabled,
        qwen_client=qwen_client,
    )
    return assembler.assemble(
        iteration_records=iteration_records,
        metrics_history=metrics_history,
        training_captures=training_captures,
        demo_videos=demo_videos,
        llm_setup_notes=llm_setup_notes,
        isaac_setup_notes=isaac_setup_notes,
        output_dir=output_dir,
    )
