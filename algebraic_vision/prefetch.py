"""Prefetch with failure propagation and clean decoder shutdown."""

import queue
from algebraic_vision.data import TrainLoader


class CheckedLoader(TrainLoader):
    def _run(self):
        try:
            super()._run()
        except Exception as error:
            while not self._stop.is_set():
                try:
                    self.queue.put(error, timeout=0.2)
                    break
                except queue.Full:
                    continue

    def next(self):
        item = self.queue.get()
        if isinstance(item, Exception):
            raise RuntimeError("input prefetch failed") from item
        return item

    def close(self):
        super().close()
        self.thread.join(timeout=10)
        if self.thread.is_alive():
            raise RuntimeError("input worker did not shut down")
