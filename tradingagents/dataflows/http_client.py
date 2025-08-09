#!/usr/bin/env python3
"""
统一HTTP客户端（httpx）
- 连接池、超时、重试（429/5xx）、指数退避
- 可选速率限制（按host配置）
- 简单指标打点（调用次数/重试次数/状态码）
"""
from __future__ import annotations

import asyncio
import time
from typing import Optional, Dict

import httpx
import os

try:
    from tradingagents.utils.metrics import metrics
except Exception:
    metrics = None

DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)
DEFAULT_UA = "TradingAgents-CN/1.0 (+httpx)"

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
    def __init__(self, base_headers: Optional[Dict[str, str]] = None, default_rate_per_sec: float = 5.0, per_host_rate: Optional[Dict[str, float]] = None):
        headers = {"User-Agent": DEFAULT_UA}
        if base_headers:
            headers.update(base_headers)
        self.client = httpx.AsyncClient(timeout=DEFAULT_TIMEOUT, headers=headers, http2=True, trust_env=True)
        self.default_limiter = RateLimiter(default_rate_per_sec)
        self.host_limiters: Dict[str, RateLimiter] = {}
        if per_host_rate:
            for host, r in per_host_rate.items():
                self.host_limiters[host] = RateLimiter(r)

    async def _limit(self, url: str):
        try:
            host = httpx.URL(url).host
        except Exception:
            host = None
        limiter = self.host_limiters.get(host) if host else None
        if limiter is None:
            limiter = self.default_limiter
        await limiter.acquire()

    async def get(self, url: str, headers: Optional[Dict[str, str]] = None, max_attempts: int = 5) -> httpx.Response:
        attempt = 0
        while True:
            attempt += 1
            await self._limit(url)
            try:
                if metrics:
                    metrics.inc("http_requests_total", {"method": "GET"})
                resp = await self.client.get(url, headers=headers)
                if resp.status_code in (429, 500, 502, 503, 504):
                    if metrics:
                        metrics.inc("http_retries_total", {"code": str(resp.status_code)})
                    if attempt >= max_attempts:
                        return resp
                    await asyncio.sleep(min(60, 2 ** attempt))
                    continue
                if metrics:
                    metrics.inc("http_status_total", {"code": str(resp.status_code)})
                return resp
            except (httpx.ConnectError, httpx.ReadTimeout):
                if metrics:
                    metrics.inc("http_errors_total", {"type": "network"})
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
        # 可按需配置 per-host 速率（可通过环境变量 JSON 覆盖）
        per_host = {
            'www.google.com': 1.0,
            'newsapi.org': 2.0,
            'finnhub.io': 2.0,
            'gnews.io': 2.0,
            'api.eastmoney.com': 2.0,
            'push2.eastmoney.com': 5.0,
        }
        try:
            import json
            cfg = os.getenv('HTTP_PER_HOST_RATE_JSON')
            if cfg:
                per_host.update(json.loads(cfg))
        except Exception:
            pass
        _shared_client = HttpClient(per_host_rate=per_host)
    return _shared_client