"""Preprocess fastMRI single-coil .h5 volumes into 128x128 magnitude .npz shards.

Reads the ``.h5`` files in ``data/raw/`` and, for each, takes the provided
``reconstruction_esc`` dataset — the *emulated single-coil* reconstruction,
which is already the real-valued magnitude image (320x320 float32, the standard
fastMRI target) — centre-crops it to a square, resizes to 128x128, and writes
one ``.npz`` per volume with key ``slices`` of shape (n_slices, 128, 128).

We use ``reconstruction_esc`` (not the raw ``kspace``) for the magnitude path,
per CLAUDE.md Contract 3 / the data rules: ``kspace`` is the *native* size
(e.g. 640x368) with readout oversampling, so ``ifft2c(kspace)`` does not equal
``reconstruction_esc`` — the standard target is the cropped esc image. The raw
``kspace`` is reserved for the complex (2-channel) stretch path.

    python data/preprocess.py --size 128

No patient data is committed; this only reads files you downloaded yourself.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from skimage.transform import resize

HERE = Path(__file__).resolve().parent
RAW_DIR = HERE / "raw"
OUT_DIR = HERE / "processed"

SOURCE_KEY = "reconstruction_esc"


def _centre_crop_square(img: np.ndarray) -> np.ndarray:
    """Centre-crop a 2D image to its largest centred square (no-op if square)."""
    h, w = img.shape
    s = min(h, w)
    top, left = (h - s) // 2, (w - s) // 2
    return img[top : top + s, left : left + s]


def _to_slices(esc: np.ndarray, size: int) -> np.ndarray:
    """(n, H, W) real magnitude volume -> (n, size, size) float32, square-cropped."""
    esc = np.abs(np.asarray(esc, dtype=np.float32))  # |.|: no-op for magnitude, safe
    out = np.empty((esc.shape[0], size, size), dtype=np.float32)
    for i, sl in enumerate(esc):
        out[i] = resize(
            _centre_crop_square(sl), (size, size), anti_aliasing=True, preserve_range=True
        )
    return out


def preprocess_volume(vol: Path, out_dir: Path, size: int = 128) -> int:
    """Write ``out_dir/<vol stem>.npz`` from one ``.h5`` volume; return its slice count.

    Returns 0 (and writes nothing) if the volume has no ``reconstruction_esc``,
    e.g. the fastMRI *test* set, which ships only undersampled k-space.
    """
    try:
        with h5py.File(vol, "r") as f:
            if SOURCE_KEY not in f:
                print(f"skip {vol.name}: no '{SOURCE_KEY}' dataset")
                return 0
            esc = f[SOURCE_KEY][()]  # (n_slices, 320, 320) float32 magnitude
    except OSError as exc:
        # A truncated download is the usual cause: the fastMRI links are
        # time-limited, so a stream that dies part-way leaves a file that looks
        # fine in `ls` and cannot be opened. Skip it rather than losing the
        # whole run -- but the caller reports it, because a silently missing
        # volume is a silently smaller training set, and that shows up later
        # as a worse prior.
        print(f"skip {vol.name}: cannot read ({exc.__class__.__name__}) -- "
              f"likely a truncated download; re-download this volume")
        return -1
    slices = _to_slices(esc, size)
    # write-then-rename, so an interrupted run never leaves a truncated shard
    tmp = out_dir / f".{vol.stem}.npz.tmp"
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, slices=slices)
    tmp.replace(out_dir / f"{vol.stem}.npz")
    return slices.shape[0]


def preprocess(size: int = 128, raw_dir: Path = RAW_DIR, out_dir: Path = OUT_DIR) -> None:
    vols = sorted(raw_dir.glob("*.h5"))
    if not vols:
        raise SystemExit(
            f"No .h5 files in {raw_dir}. Run `python data/download_subset.py` first."
        )
    out_dir.mkdir(parents=True, exist_ok=True)

    unreadable = []
    for vol in vols:
        n = preprocess_volume(vol, out_dir, size)
        if n == -1:
            unreadable.append(vol.name)
        elif n:
            print(f"{vol.name}: {n} slices -> {out_dir / (vol.stem + '.npz')}")

    shards = sorted(out_dir.glob("*.npz"))
    total = sum(np.load(p)["slices"].shape[0] for p in shards)
    print(f"done: {total} slices from {len(shards)} volume(s) in {out_dir}")
    if unreadable:
        print(f"WARNING: {len(unreadable)} of {len(vols)} raw volume(s) could not "
              f"be read and were skipped; re-download them:")
        for name in unreadable:
            print(f"  - {name}")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--size", type=int, default=128)
    p.add_argument("--raw", type=Path, default=RAW_DIR, help="directory of .h5 volumes")
    p.add_argument("--out", type=Path, default=OUT_DIR, help="where to write .npz shards")
    args = p.parse_args()
    preprocess(args.size, args.raw, args.out)


if __name__ == "__main__":
    main()
