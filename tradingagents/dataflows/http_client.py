#!/usr/bin/env python3
"""
统一HTTP客户端（httpx）
- 连接池、超时、重试（429/5xx）、指数退避
- 可选速率限制（简单基于时间窗）
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict, Any

import httpx

DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)

class RateLimiter:
    def __init__(self, rate_per_sec: float = 5.0):
        self.min_interval = 1.0 / max(rate_per_sec, 0.0001)
        self._last = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self):
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last = time.monotonic()

class HttpClient:
    def __init__(self, base_headers: Optional[Dict[str, str]] = None, rate_per_sec: float = 5.0):
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=base_headers or {}, http2=True)
        self.rate_limiter = RateLimiter(rate_per_sec)

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, max_attempts: int = 5) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            await self.rate_limiter.acquire()
            try:
                resp = await self.client.get(url, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if attempt >= max_attempts:
                        return resp
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                return resp
            except (httpx.ConnectError, httpx.ReadTimeout):
                if attempt >= max_attempts:
                    raise
                await asyncio.sleep(min(60, 2 ** attempt))

    async def aclose(self):
        await self.client.aclose()

# 单例
_shared_client: Optional[HttpClient] = None

async def get_http_client() -> HttpClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = HttpClient()
    return _shared_client