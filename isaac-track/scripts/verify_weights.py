#!/usr/bin/env python3
"""Task 0.3 — post-restore weight verification.

Verifies a restored model directory before vLLM is launched, so a corrupt or
incomplete EBS restore is caught loudly instead of surfacing as a confusing
vLLM load error at 21:00.

Two modes:
  --write-manifest   compute a manifest (file list, sizes, sha256) and write it
                     to <model-dir>/.weights_manifest.json. Run this ONCE on the
                     known-good seed instance before snapshotting.
  (default) verify   recompute and compare against the committed manifest; if no
                     manifest is present, fall back to structural checks
                     (required files present, sizes > 0, sharded weights complete).

Exit code 0 = OK, non-zero = verification failed (fail loudly).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

MANIFEST_NAME = ".weights_manifest.json"

# Files a HF/vLLM model dir must have to load. Weight files are checked
# separately because they may be sharded (*.safetensors) or single-file.
REQUIRED_FILES = ["config.json", "tokenizer_config.json"]
WEIGHT_GLOBS = ["*.safetensors", "*.bin"]


def sha256_of(path: Path, chunk: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def iter_files(model_dir: Path):
    for p in sorted(model_dir.rglob("*")):
        if p.is_file() and p.name != MANIFEST_NAME:
            yield p


def build_manifest(model_dir: Path) -> dict:
    files = {}
    for p in iter_files(model_dir):
        rel = str(p.relative_to(model_dir))
        files[rel] = {"size": p.stat().st_size, "sha256": sha256_of(p)}
    return {"file_count": len(files), "files": files}


def fail(msg: str) -> "NoReturn":  # type: ignore[name-defined]
    print(f"VERIFY FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def structural_checks(model_dir: Path) -> None:
    for req in REQUIRED_FILES:
        fp = model_dir / req
        if not fp.is_file():
            fail(f"required file missing: {req}")
        if fp.stat().st_size == 0:
            fail(f"required file is empty: {req}")

    weight_files = [p for g in WEIGHT_GLOBS for p in model_dir.glob(g)]
    if not weight_files:
        fail("no weight files (*.safetensors / *.bin) found")
    for wf in weight_files:
        if wf.stat().st_size == 0:
            fail(f"weight file is empty: {wf.name}")

    # If safetensors are sharded, the index must reference every shard present.
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        data = json.loads(index.read_text())
        referenced = set(data.get("weight_map", {}).values())
        present = {p.name for p in model_dir.glob("*.safetensors")}
        missing = referenced - present
        if missing:
            fail(f"sharded weights incomplete; missing: {sorted(missing)}")
    print(f"structural check OK: {len(weight_files)} weight file(s), required files present")


def verify_against_manifest(model_dir: Path, manifest: dict) -> None:
    expected = manifest["files"]
    actual_files = {str(p.relative_to(model_dir)): p for p in iter_files(model_dir)}

    missing = set(expected) - set(actual_files)
    if missing:
        fail(f"{len(missing)} file(s) missing vs manifest: {sorted(missing)[:10]}")

    extra = set(actual_files) - set(expected)
    if extra:
        # Extra files are non-fatal but reported.
        print(f"WARNING: {len(extra)} file(s) not in manifest: {sorted(extra)[:10]}")

    for rel, meta in expected.items():
        p = actual_files[rel]
        size = p.stat().st_size
        if size != meta["size"]:
            fail(f"size mismatch {rel}: expected {meta['size']}, got {size}")
        digest = sha256_of(p)
        if digest != meta["sha256"]:
            fail(f"checksum mismatch {rel}: expected {meta['sha256'][:12]}…, got {digest[:12]}…")

    print(f"checksum verify OK: {len(expected)} files match manifest")


def main() -> None:
    ap = argparse.ArgumentParser(description="Verify restored model weights.")
    ap.add_argument("--model-dir", required=True, type=Path)
    ap.add_argument("--write-manifest", action="store_true",
                    help="compute and write the manifest (run on known-good seed instance)")
    args = ap.parse_args()

    model_dir: Path = args.model_dir
    if not model_dir.is_dir():
        fail(f"model dir does not exist: {model_dir}")

    if args.write_manifest:
        manifest = build_manifest(model_dir)
        out = model_dir / MANIFEST_NAME
        out.write_text(json.dumps(manifest, indent=2))
        print(f"wrote manifest: {out} ({manifest['file_count']} files)")
        return

    manifest_path = model_dir / MANIFEST_NAME
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        verify_against_manifest(model_dir, manifest)
    else:
        print(f"no {MANIFEST_NAME} found; running structural checks only")
        structural_checks(model_dir)

    print("VERIFY OK")


if __name__ == "__main__":
    main()
