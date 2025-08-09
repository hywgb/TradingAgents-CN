#!/usr/bin/env python3
from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, Tuple, Deque, List
from threading import Lock
import math

class _Metrics:
    def __init__(self):
        self.counters: Dict[Tuple[str, Tuple[Tuple[str,str],...]], int] = defaultdict(int)
        self.hists: Dict[Tuple[str, Tuple[Tuple[str,str],...]], Deque[float]] = {}
        self._lock = Lock()
    def inc(self, name: str, labels: Dict[str,str] | None = None, value: int = 1):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            self.counters[key] += value
    def hist(self, name: str, labels: Dict[str,str] | None, value: float, maxlen: int = 500):
        key = (name, tuple(sorted((labels or {}).items())))
        with self._lock:
            if key not in self.hists:
                self.hists[key] = deque(maxlen=maxlen)
            self.hists[key].append(value)
    def _percentile(self, arr: List[float], q: float) -> float:
        if not arr:
            return 0.0
        arr_sorted = sorted(arr)
        k = (len(arr_sorted)-1) * q
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return arr_sorted[int(k)]
        d0 = arr_sorted[int(f)] * (c-k)
        d1 = arr_sorted[int(c)] * (k-f)
        return d0 + d1
    def snapshot(self) -> Dict[str,object]:
        with self._lock:
            counters = {f"{name}{{{','.join([k+'='+v for k,v in labels])}}}": v for (name, labels), v in self.counters.items()}
            hists = {}
            for (name, labels), dq in self.hists.items():
                arr = list(dq)
                hists[f"{name}{{{','.join([k+'='+v for k,v in labels])}}}"] = {
                    'count': len(arr),
                    'p50': self._percentile(arr, 0.50),
                    'p95': self._percentile(arr, 0.95),
                    'min': min(arr) if arr else 0.0,
                    'max': max(arr) if arr else 0.0,
                }
            return {'counters': counters, 'hists': hists}

metrics = _Metrics()