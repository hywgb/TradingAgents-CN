#!/usr/bin/env python3
from __future__ import annotations

import json
import base64
from io import BytesIO
from typing import Any, Optional

import numpy as np

from tradingagents.config.database_manager import get_redis_client, is_redis_available
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

DEFAULT_TTL_SECONDS = 6 * 60 * 60  # 6 hours
EMBEDDING_TTL_SECONDS = 7 * 24 * 60 * 60  # 7 days


def _get_redis():
    try:
        if is_redis_available():
            return get_redis_client()
    except Exception as e:
        logger.debug(f"[cache] Redis not available: {e}")
    return None


def cache_get(key: str) -> Optional[str]:
    client = _get_redis()
    if not client:
        return None
    try:
        val = client.get(key)
        return val.decode('utf-8') if isinstance(val, (bytes, bytearray)) else val
    except Exception as e:
        logger.debug(f"[cache] get failed: {e}")
        return None


def cache_set(key: str, value: str, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    client = _get_redis()
    if not client:
        return False
    try:
        client.setex(key, ttl, value)
        return True
    except Exception as e:
        logger.debug(f"[cache] set failed: {e}")
        return False


def cache_get_json(key: str) -> Optional[Any]:
    raw = cache_get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def cache_set_json(key: str, obj: Any, ttl: int = DEFAULT_TTL_SECONDS) -> bool:
    try:
        return cache_set(key, json.dumps(obj, ensure_ascii=False), ttl)
    except Exception:
        return False


def ndarray_to_b64(arr: np.ndarray) -> str:
    buf = BytesIO()
    # 保存 dtype/shape 信息
    np.save(buf, arr, allow_pickle=False)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def b64_to_ndarray(data: str) -> np.ndarray:
    raw = base64.b64decode(data.encode('ascii'))
    buf = BytesIO(raw)
    return np.load(buf, allow_pickle=False)


def emb_get(key: str) -> Optional[np.ndarray]:
    raw = cache_get(key)
    if raw is None:
        return None
    try:
        return b64_to_ndarray(raw)
    except Exception as e:
        logger.debug(f"[cache] emb decode failed: {e}")
        return None


def emb_set(key: str, arr: np.ndarray, ttl: int = EMBEDDING_TTL_SECONDS) -> bool:
    try:
        s = ndarray_to_b64(arr)
        return cache_set(key, s, ttl)
    except Exception as e:
        logger.debug(f"[cache] emb encode failed: {e}")
        return False