#!/usr/bin/env python3
"""
A股量化模块（轻量版骨架）
- 数据：统一从 dataflows.interface 提供的统一接口/或 Tushare 拉取并落地到本地CSV/Mongo（可选）
- 特征：示例性 Alpha 因子（动量、均线偏离、成交量变化、波动率）
- 回测：简要的截面选股 + 等权持仓收益计算骨架
- 性能：优先使用 pandas，后续可切换 Polars/cuDF（通过 feature flag）
"""
from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np

from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

# Feature flags
USE_POLARS = os.getenv('TA_USE_POLARS', 'false').lower() in ('1','true','yes')
USE_GPU_DF = os.getenv('TA_USE_CUDF', 'false').lower() in ('1','true','yes')

try:
    import polars as pl  # type: ignore
except Exception:
    pl = None

try:
    import cudf  # type: ignore
except Exception:
    cudf = None

@dataclass
class Bar:
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: float


def load_daily_bars(symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
    """从统一接口或本地缓存加载日线数据（开高低收量）
    返回 pandas DataFrame: [date, open, high, low, close, volume]
    """
    try:
        from .interface import get_china_stock_data_tushare
        csv = get_china_stock_data_tushare(symbol, start_date, end_date)
        # 若统一接口返回字符串报告，此处仅作占位，真实系统应返回结构化数据
        # 这里兼容：如果返回DataFrame字符串或CSV格式，尝试解析
        if isinstance(csv, str) and '日期' in csv and ',' in csv:
            from io import StringIO
            df = pd.read_csv(StringIO(csv))
        else:
            # 回退：构造空DF
            df = pd.DataFrame(columns=['date','open','high','low','close','volume'])
    except Exception as e:
        logger.warning(f"[Quant] 加载数据失败，返回空DF: {e}")
        df = pd.DataFrame(columns=['date','open','high','low','close','volume'])

    # 规整字段
    rename_map = {
        '日期':'date','开盘':'open','最高':'high','最低':'low','收盘':'close','成交量':'volume','volume':'volume'
    }
    for c in list(df.columns):
        if c in rename_map:
            df.rename(columns={c: rename_map[c]}, inplace=True)
    keep = [c for c in ['date','open','high','low','close','volume'] if c in df.columns]
    df = df[keep].copy()
    # 类型
    if 'date' in df:
        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
    for c in ['open','high','low','close','volume']:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors='coerce')
    df.dropna(subset=['close'], inplace=True)
    df.sort_values('date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def compute_alpha_factors(df: pd.DataFrame) -> pd.DataFrame:
    """示例 Alpha 因子计算（向量化）"""
    if df.empty:
        return df
    out = df.copy()
    close = out['close']
    volume = out['volume'] if 'volume' in out else pd.Series(np.zeros(len(out)))
    # 动量因子（20日收益）
    out['mom_20'] = close.pct_change(20)
    # 均线偏离（close 与 20日均线差值/均线）
    ma20 = close.rolling(20, min_periods=5).mean()
    out['ma20_dev'] = (close - ma20) / (ma20.replace(0, np.nan))
    # 成交量变化（5日平均与20日平均比值）
    vol5 = volume.rolling(5, min_periods=3).mean()
    vol20 = volume.rolling(20, min_periods=5).mean()
    out['vol_ratio'] = (vol5 / vol20.replace(0, np.nan))
    # 波动率（20日对数收益标准差）
    logret = np.log(close.replace(0, np.nan)).diff()
    out['vol_20'] = logret.rolling(20, min_periods=5).std()
    return out


def simple_cross_section_select(factors: pd.DataFrame, top_k: int = 20) -> List[int]:
    """按因子打分选股的简单骨架（单标的时返回最近一日信号）"""
    if factors.empty:
        return []
    # 单标的：用 mom_20 + ma20_dev - vol_20 作为简单评分
    s = (
        factors['mom_20'].fillna(0.0) +
        factors['ma20_dev'].fillna(0.0) -
        factors['vol_20'].fillna(0.0)
    )
    # 取最近一日是否为正作为信号
    last = s.iloc[-1]
    return [1 if last > 0 else 0]


def backtest_equal_weight(dates: List[str], signals: List[int], returns: List[float]) -> Dict[str, float]:
    """等权持仓回测：单标的示例。未来可扩展为多标的横截面回测。"""
    if not dates or not signals or not returns:
        return {'cum_return': 0.0, 'trade_days': 0}
    cum = 1.0
    for sig, r in zip(signals, returns):
        if sig == 1 and not math.isnan(r):
            cum *= (1.0 + r)
    return {'cum_return': cum - 1.0, 'trade_days': len(dates)}


def run_quant_pipeline(symbol: str, start_date: str, end_date: str) -> Dict[str, object]:
    """量化流程：加载数据 -> 计算因子 -> 生成简单信号 -> 回测结果"""
    df = load_daily_bars(symbol, start_date, end_date)
    if df.empty:
        return {'symbol': symbol, 'factors': df, 'signals': [], 'backtest': {}}
    fac = compute_alpha_factors(df)
    # 次日收益（简单近似）：close 的日收益
    next_ret = fac['close'].pct_change().shift(-1)
    signal = simple_cross_section_select(fac, top_k=20)
    # 与 dates 对齐（示例仅取最后一天信号）
    returns = [next_ret.iloc[-1] if not next_ret.empty else np.nan]
    dates = [fac['date'].iloc[-1]]
    bt = backtest_equal_weight(dates, signal, returns)
    return {'symbol': symbol, 'factors': fac, 'signals': signal, 'backtest': bt}