"""
In-process caches (#132). Only safe, reusable values: query embeddings (keyed by model + exact text) and
patient chunk matrices (keyed by patient id + a fingerprint that changes whenever that patient's chunks change).
Nothing is shared across patients: every patient-derived key contains the patient id. Per process, in memory,
never persisted.
"""
import threading
from collections import OrderedDict
from typing import Any, Callable, Hashable


class LRU:
    def __init__(self, maxsize: int):
        self.maxsize, self._d, self._lock = maxsize, OrderedDict(), threading.Lock()
        self.hits = self.misses = 0

    def get_or_set(self, key: Hashable, make: Callable[[], Any]) -> Any:
        with self._lock:
            if key in self._d:
                self._d.move_to_end(key)
                self.hits += 1
                return self._d[key]
        value = make()
        with self._lock:
            self.misses += 1
            self._d[key] = value
            self._d.move_to_end(key)
            while len(self._d) > self.maxsize:
                self._d.popitem(last=False)
        return value

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self.hits = self.misses = 0


QUERY_EMBEDDINGS = LRU(2048)
PATIENT_CHUNKS = LRU(256)
