"""ImageNet input pipeline shared by both arms.

The global order of training images is fixed by the seed alone: epoch e uses
a seeded permutation of the training indices, and step t takes the next
``global_batch`` positions of the concatenated epochs. Each process loads
only its contiguous share of every global batch, so the images seen at a
step do not depend on how many hosts or chips run the job, and the two arms
of a comparison see identical batches for the same seed.

Two storage layouts are supported (see scripts/prepare_imagenet.py):
uint8 pixel arrays (64x64 copy) and concatenated JPEG bytes with an offset
table (256x256 copy), decoded by a pool of worker processes.
"""

import io
import multiprocessing as mp
import queue
import threading
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
from PIL import Image


class PixelSource:
    """Decoded uint8 images stored as an (N, H, W, 3) .npy file."""

    def __init__(self, root, split):
        root = Path(root)
        self.images = np.load(root / f"{split}_images.npy", mmap_mode="r")
        self.labels = np.load(root / f"{split}_labels.npy").astype(np.int32)
        self.image_shape = self.images.shape[1:]

    def __len__(self):
        return len(self.labels)

    def load(self, indices):
        order = np.argsort(indices)
        out = np.empty((len(indices),) + self.image_shape, np.uint8)
        out[order] = self.images[indices[order]]
        return out, self.labels[indices]

    def close(self):
        pass


_JPEG = {}


def _init_jpeg_worker(root, split):
    root = Path(root)
    _JPEG["blob"] = np.memmap(root / f"{split}_jpeg.bin", np.uint8, "r")
    _JPEG["offsets"] = np.load(root / f"{split}_offsets.npy")


def _decode_into(job):
    """Decode JPEGs straight into a shared-memory batch buffer."""
    name, shape, first, indices = job
    if _JPEG.get("shm_name") != name:
        _JPEG["shm"] = shared_memory.SharedMemory(name=name)
        _JPEG["shm_name"] = name
    out = np.ndarray(shape, np.uint8, buffer=_JPEG["shm"].buf)
    blob, offsets = _JPEG["blob"], _JPEG["offsets"]
    for row, i in enumerate(indices, start=first):
        data = blob[offsets[i]:offsets[i + 1]].tobytes()
        out[row] = np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))
    return len(indices)


class JpegSource:
    """JPEG bytes decoded on demand by a process pool (256x256 copy).

    Workers write decoded pixels into one shared-memory buffer, so only
    indices cross the process boundary.
    """

    def __init__(self, root, split, workers=32, image_shape=(256, 256, 3),
                 max_batch=1024):
        root = Path(root)
        self.labels = np.load(root / f"{split}_labels.npy").astype(np.int32)
        self.image_shape = tuple(image_shape)
        self.workers = workers
        self.shape = (max_batch,) + self.image_shape
        self.shm = shared_memory.SharedMemory(
            create=True, size=int(np.prod(self.shape)))
        self.buffer = np.ndarray(self.shape, np.uint8, buffer=self.shm.buf)
        self.lock = threading.Lock()
        self.pool = mp.get_context("spawn").Pool(
            workers, initializer=_init_jpeg_worker,
            initargs=(str(root), split))

    def __len__(self):
        return len(self.labels)

    def load(self, indices):
        indices = np.asarray(indices)
        if len(indices) > self.shape[0]:
            raise ValueError("batch larger than the shared buffer")
        bounds = np.linspace(0, len(indices), self.workers + 1).astype(int)
        jobs = [(self.shm.name, self.shape, int(a), indices[a:b])
                for a, b in zip(bounds[:-1], bounds[1:]) if b > a]
        with self.lock:
            self.pool.map(_decode_into, jobs)
            images = self.buffer[:len(indices)].copy()
        return images, self.labels[indices]

    def close(self):
        self.pool.terminate()
        self.shm.close()
        self.shm.unlink()


def open_source(root, split, **kwargs):
    root = Path(root)
    if (root / f"{split}_images.npy").exists():
        return PixelSource(root, split)
    return JpegSource(root, split, **kwargs)


class EpochSampler:
    """Seeded, topology-independent order over a fixed index set."""

    def __init__(self, indices, seed):
        self.indices = np.asarray(indices)
        self.seed = seed
        self._cache = {}

    def _epoch(self, e):
        if e not in self._cache:
            rng = np.random.default_rng([self.seed, e])
            self._cache = {e: self.indices[rng.permutation(len(self.indices))]}
        return self._cache[e]

    def positions(self, start, count):
        n = len(self.indices)
        out = []
        while count > 0:
            e, off = divmod(start, n)
            take = min(count, n - off)
            out.append(self._epoch(e)[off:off + take])
            start += take
            count -= take
        return np.concatenate(out)


class TrainLoader:
    """Background prefetch of this process's share of each global batch."""

    def __init__(self, source, sampler, global_batch, process_index=0,
                 process_count=1, start_step=0, prefetch=4):
        if global_batch % process_count:
            raise ValueError("global batch must divide across processes")
        self.source = source
        self.sampler = sampler
        self.global_batch = global_batch
        self.local = global_batch // process_count
        self.offset = process_index * self.local
        self.queue = queue.Queue(maxsize=prefetch)
        self.step = start_step
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        step = self.step
        while not self._stop.is_set():
            idx = self.sampler.positions(step * self.global_batch
                                         + self.offset, self.local)
            item = self.source.load(idx)
            while not self._stop.is_set():
                try:
                    self.queue.put(item, timeout=1.0)
                    break
                except queue.Full:
                    continue
            step += 1

    def next(self):
        return self.queue.get()

    def close(self):
        self._stop.set()


def eval_batches(source, indices, global_batch, process_index=0,
                 process_count=1):
    """Yield (images, labels, valid_mask) shares covering `indices` once.

    The last batch is padded by repeating index 0 and masked out.
    """
    indices = np.asarray(indices)
    local = global_batch // process_count
    for start in range(0, len(indices), global_batch):
        chunk = indices[start:start + global_batch]
        valid = np.ones(global_batch, bool)
        if len(chunk) < global_batch:
            valid[len(chunk):] = False
            chunk = np.concatenate([chunk, np.full(global_batch - len(chunk),
                                                   indices[0])])
        sl = slice(process_index * local, (process_index + 1) * local)
        images, labels = source.load(chunk[sl])
        yield images, labels, valid[sl]
