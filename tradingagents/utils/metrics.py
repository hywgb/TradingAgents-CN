#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict
from typing import Dict, Tuple
from threading import Lock

class _Metrics:
    def __init__(self):
        self.counters: Dict[Tuple[str, Tuple[Tuple[str,str],...]], int] = defaultdict(int)
        self._lock = Lock()
    def inc(self, name: str, labels: Dict[str,str] | None = None, value: int = 1):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self.counters[key] += value
    def snapshot(self) -> Dict[str,int]:
        with self._lock:
            return {f"{name}{{{','.join([k+'='+v for k,v in labels])}}}": v for (name, labels), v in self.counters.items()}

metrics = _Metrics()