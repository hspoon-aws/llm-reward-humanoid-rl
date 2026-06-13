#!/usr/bin/env python3
"""Standalone video encoder — runs in its OWN process (no JAX imported).

The demo recorder collects rendered RGB frames during the JAX rollout, saves
them to a .npy, and invokes THIS script via subprocess (fork+exec). Encoding
here never touches JAX's threadpool, so it avoids the `os.fork()` deadlock that
`imageio-ffmpeg` triggers when it forks the ffmpeg child from inside a live,
multithreaded JAX process (see docs/lesson-problems-and-resolutions.md §6.2).

Usage:
    python -m src.eval.encode_video <frames.npy> <out.mp4> <fps>

Frames are padded to even/macro-block-friendly dimensions so ffmpeg never warns
or refuses (libx264 needs even dims; many players prefer multiples of 16).
"""
from __future__ import annotations

import sys


def _pad_to_multiple(frames, multiple: int = 16):
    import numpy as np

    n, h, w, c = frames.shape
    new_h = ((h + multiple - 1) // multiple) * multiple
    new_w = ((w + multiple - 1) // multiple) * multiple
    if new_h == h and new_w == w:
        return frames
    padded = np.zeros((n, new_h, new_w, c), dtype=frames.dtype)
    padded[:, :h, :w, :] = frames
    return padded


def main(argv) -> int:
    if len(argv) != 3:
        print("usage: encode_video.py <frames.npy> <out.mp4> <fps>", file=sys.stderr)
        return 2
    npy_path, out_path, fps_s = argv
    fps = int(float(fps_s))

    import numpy as np
    import imageio

    frames = np.load(npy_path)
    frames = _pad_to_multiple(frames, 16)
    # macro_block_size=1 + even dims (guaranteed by padding) avoids the resize warning.
    imageio.mimwrite(out_path, list(frames), fps=fps, macro_block_size=1)
    print(f"encoded {len(frames)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
