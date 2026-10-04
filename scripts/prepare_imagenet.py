#!/usr/bin/env python3
"""Convert the ImageNet-1k parquet shards into training-friendly arrays.

Two layouts are produced, depending on the source resolution:

* ``--layout pixels`` decodes every JPEG once and stores uint8 arrays of shape
  (N, H, W, 3). Used for the 64x64 copy (15.7 GB for the training split).
* ``--layout jpeg`` keeps the compressed bytes, concatenated into one file with
  an offset table, so data-loader workers can decode random images through a
  shared memory map. Used for the 256x256 copy.

Both layouts store int16 labels and the original file names. A fixed
"minival" split of 10 training images per class is chosen from the sorted
file names with a seeded generator, so the same images are held out at every
resolution.
"""

import argparse
import glob
import io
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

MINIVAL_PER_CLASS = 10
MINIVAL_SEED = 0
NUM_CLASSES = 1000


def _row_groups(files):
    """Yield (file, row_group, first_row) for every row group, in order."""
    start = 0
    for f in files:
        meta = pq.ParquetFile(f).metadata
        for g in range(meta.num_row_groups):
            yield f, g, start
            start += meta.row_group(g).num_rows


def _read_group(job):
    path, group, _ = job
    table = pq.ParquetFile(path).read_row_group(group)
    images = table.column("image").to_pylist()
    labels = np.asarray(table.column("label").to_pylist(), dtype=np.int16)
    names = [im["path"] for im in images]
    data = [im["bytes"] for im in images]
    return data, labels, names


def _decode_group(job):
    data, labels, names = _read_group(job)
    pixels = np.stack([
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB")) for b in data
    ])
    return pixels, labels, names


def _pool(workers):
    # Parquet readers start threads; forking after that can deadlock the
    # children, so workers are started fresh.
    return mp.get_context("spawn").Pool(workers)


def convert_split(files, out_dir, split, layout, workers):
    jobs = list(_row_groups(files))
    total = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
    out_dir.mkdir(parents=True, exist_ok=True)
    labels = np.empty(total, dtype=np.int16)
    names = []

    if layout == "pixels":
        first = _decode_group(jobs[0])[0]
        shape = (total,) + first.shape[1:]
        pixels = np.lib.format.open_memmap(
            out_dir / f"{split}_images.npy", mode="w+", dtype=np.uint8,
            shape=shape)
        with _pool(workers) as pool:
            for (px, lb, nm), (_, _, start) in zip(
                    pool.imap(_decode_group, jobs, chunksize=4), jobs):
                pixels[start:start + len(lb)] = px
                labels[start:start + len(lb)] = lb
                names.extend(nm)
        pixels.flush()
        del pixels
    else:
        offsets = np.zeros(total + 1, dtype=np.int64)
        with open(out_dir / f"{split}_jpeg.bin", "wb") as blob, \
                _pool(workers) as pool:
            position = 0
            for (data, lb, nm), (_, _, start) in zip(
                    pool.imap(_read_group, jobs, chunksize=4), jobs):
                for i, b in enumerate(data):
                    blob.write(b)
                    position += len(b)
                    offsets[start + i + 1] = position
                labels[start:start + len(lb)] = lb
                names.extend(nm)
        np.save(out_dir / f"{split}_offsets.npy", offsets)

    np.save(out_dir / f"{split}_labels.npy", labels)
    np.save(out_dir / f"{split}_names.npy", np.asarray(names))
    return labels, names


def minival_indices(labels, names):
    """10 images per class, chosen by file name so all resolutions agree."""
    rng = np.random.default_rng(MINIVAL_SEED)
    names = np.asarray(names)
    chosen = []
    for c in range(NUM_CLASSES):
        idx = np.flatnonzero(labels == c)
        idx = idx[np.argsort(names[idx])]
        chosen.append(np.sort(rng.choice(idx, MINIVAL_PER_CLASS,
                                         replace=False)))
    return np.sort(np.concatenate(chosen))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True,
                        help="directory with train-*.parquet and "
                             "validation-*.parquet")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--layout", choices=("pixels", "jpeg"),
                        required=True)
    parser.add_argument("--workers", type=int,
                        default=min(64, os.cpu_count()))
    args = parser.parse_args()

    for split, pattern in (("train", "train-*.parquet"),
                           ("val", "validation-*.parquet")):
        files = sorted(glob.glob(str(args.source / pattern)))
        if not files:
            raise FileNotFoundError(f"no {pattern} in {args.source}")
        labels, names = convert_split(files, args.out, split, args.layout,
                                      args.workers)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        print(f"{split}: {len(labels)} images, per-class "
              f"min {counts.min()} max {counts.max()}", flush=True)
        if split == "train":
            held = minival_indices(labels, names)
            keep = np.setdiff1d(np.arange(len(labels)), held)
            np.save(args.out / "minival_indices.npy", held)
            np.save(args.out / "train_indices.npy", keep)
            print(f"minival: {len(held)} images; train: {len(keep)}",
                  flush=True)


if __name__ == "__main__":
    main()
