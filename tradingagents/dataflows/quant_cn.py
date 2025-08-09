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


def backtest_equal_weight(dates: List[str], signals: List[int], returns: List[float], commission_bps: int = 0, buy_hold_ret: Optional[float] = None) -> Dict[str, float]:
    """等权持仓回测：单标的示例。未来可扩展为多标的横截面回测。"""
    if not dates or not signals or not returns:
        return {'cum_return': 0.0, 'trade_days': 0, 'buy_hold_cum_return': buy_hold_ret or 0.0}
    cum = 1.0
    for sig, r in zip(signals, returns):
        if sig == 1 and not math.isnan(r):
            # 扣除单边交易成本（bps）
            cost = commission_bps / 10000.0 if commission_bps else 0.0
            cum *= max(0.0, (1.0 + r - cost))
    return {'cum_return': cum - 1.0, 'trade_days': len(dates), 'buy_hold_cum_return': buy_hold_ret or 0.0}


def run_quant_pipeline(symbol: str, start_date: str, end_date: str, commission_bps: int = 0) -> Dict[str, object]:
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
    # 基准：区间内买入并持有
    buy_hold = float((fac['close'].pct_change().add(1).cumprod().iloc[-1] - 1.0) if len(fac) > 1 else 0.0)
    bt = backtest_equal_weight(dates, signal, returns, commission_bps=commission_bps, buy_hold_ret=buy_hold)
    return {'symbol': symbol, 'factors': fac, 'signals': signal, 'backtest': bt}


def cross_section_backtest(universe: List[str], start_date: str, end_date: str, commission_bps: int = 0) -> Dict[str, object]:
    """极简横截面回测：
    - 对每个标的计算因子与当日评分（mom_20 + ma20_dev - vol_20）
    - 取样本期最后一天，选择 Top-1 作为持仓
    - 下一交易日收益作为组合收益，扣除交易成本
    """
    results: Dict[str, Dict[str, object]] = {}
    scores: List[Tuple[str, float]] = []
    for sym in universe:
        df = load_daily_bars(sym, start_date, end_date)
        if df.empty or len(df) < 30:
            continue
        fac = compute_alpha_factors(df)
        s = (
            fac['mom_20'].fillna(0.0) +
            fac['ma20_dev'].fillna(0.0) -
            fac['vol_20'].fillna(0.0)
        )
        scores.append((sym, float(s.iloc[-1])))
        results[sym] = {'factors': fac}
    if not scores:
        return {'universe': universe, 'selected': [], 'portfolio_return': 0.0}
    # 选取Top-1
    scores.sort(key=lambda x: x[1], reverse=True)
    selected = [scores[0][0]]
    # 计算下一日收益
    rets = []
    for sym in selected:
        fac = results[sym]['factors']
        next_ret = fac['close'].pct_change().shift(-1)
        r = float(next_ret.iloc[-1]) if not next_ret.empty else 0.0
        cost = commission_bps / 10000.0 if commission_bps else 0.0
        rets.append(max(0.0, r - cost))
    port_ret = float(np.mean(rets)) if rets else 0.0
    return {'universe': universe, 'selected': selected, 'portfolio_return': port_ret, 'scores': scores[:5]}


def _resample_dates(dates: pd.Series, freq: str) -> List[str]:
    """Resample trading dates to period endpoints according to freq (e.g., 'W','M').
    Returns a sorted list of end dates in '%Y-%m-%d' format that are present in dates.
    """
    if dates.empty:
        return []
    d = pd.to_datetime(dates)
    # Build a DataFrame for resampling alignment
    tmp = pd.DataFrame({'one': 1}, index=d)
    # Use period end by resampling then forward/back alignment
    if freq.upper().startswith('W'):
        rs = tmp.resample('W-FRI').sum()
    elif freq.upper().startswith('M'):
        rs = tmp.resample('M').sum()
    else:
        # Default weekly Friday
        rs = tmp.resample('W-FRI').sum()
    # Map resampled index back to nearest trading date <= period end
    period_ends: List[str] = []
    d_sorted = pd.Series(d.sort_values().unique())
    for end in rs.index:
        end_date = pd.Timestamp(end).tz_localize(None).to_pydatetime().date()
        # find last trading day <= end_date
        candidates = d_sorted[d_sorted <= pd.Timestamp(end_date)]
        if len(candidates) == 0:
            continue
        period_ends.append(pd.Timestamp(candidates.iloc[-1]).strftime('%Y-%m-%d'))
    # De-duplicate and sort
    period_ends = sorted(list(dict.fromkeys(period_ends)))
    return period_ends


def _score_latest(fac: pd.DataFrame) -> float:
    """Compute simple score on the latest row of factors."""
    if fac.empty:
        return float('nan')
    latest = fac.iloc[-1]
    return float(
        (latest.get('mom_20', 0.0) or 0.0)
        + (latest.get('ma20_dev', 0.0) or 0.0)
        - (latest.get('vol_20', 0.0) or 0.0)
    )


def _max_drawdown(series: List[float]) -> float:
    if not series:
        return 0.0
    peak = -1e9
    mdd = 0.0
    for v in series:
        peak = max(peak, v)
        mdd = max(mdd, (peak - v) / max(peak, 1e-9))
    return float(mdd)


def _load_benchmark_close(start_date: str, end_date: str) -> Optional[pd.Series]:
    """尝试加载沪深300指数收盘，若不可用返回None。此处用tushare统一接口占位。"""
    try:
        # 使用000300.SH或399300.SZ等作基准，统一接口当前返回文本，做简化占位：返回None
        return None
    except Exception:
        return None


def rolling_cross_section_backtest(
    universe: List[str],
    start_date: str,
    end_date: str,
    freq: str = 'W',
    top_k: int = 3,
    commission_bps: int = 0,
) -> Dict[str, object]:
    """
    滚动横截面回测（简版）：
    - 以freq为调仓频率（'W'周/'M'月），每期在样本期末日用简单分数选择Top-K
    - 下一交易日开盘到收盘（日收益近似）作为当期收益，组合等权
    - 扣除单边交易成本（bps）

    返回:
    {
      'params': {...},
      'dates': [rebalance_dates...],
      'selected_each_period': [[symbols...], ...],
      'period_returns': [r1, r2, ...],
      'cum_curve': [{'date': d, 'cum': v}, ...],
      'summary': {'cum_return': x, 'avg_period_ret': y, 'periods': n}
    }
    """
    top_k = max(1, int(top_k))
    cost = commission_bps / 10000.0 if commission_bps else 0.0

    # Load and cache factors per symbol once
    sym_to_fac: Dict[str, pd.DataFrame] = {}
    for sym in universe:
        try:
            df = load_daily_bars(sym, start_date, end_date)
            if df.empty or len(df) < 40:
                continue
            fac = compute_alpha_factors(df)
            sym_to_fac[sym] = fac
        except Exception as e:
            logger.warning(f"[Quant] 加载因子失败 {sym}: {e}")

    if not sym_to_fac:
        return {'params': {'freq': freq, 'top_k': top_k, 'commission_bps': commission_bps},
                'dates': [], 'selected_each_period': [], 'period_returns': [],
                'cum_curve': [], 'summary': {'cum_return': 0.0, 'avg_period_ret': 0.0, 'periods': 0}}

    # Determine rebalance dates based on union of trading calendars
    all_dates = pd.Series(sorted({d for fac in sym_to_fac.values() for d in fac['date']}))
    rbd = _resample_dates(all_dates, freq)
    if len(rbd) == 0:
        return {'params': {'freq': freq, 'top_k': top_k, 'commission_bps': commission_bps},
                'dates': [], 'selected_each_period': [], 'period_returns': [],
                'cum_curve': [], 'summary': {'cum_return': 0.0, 'avg_period_ret': 0.0, 'periods': 0}}

    selected_each: List[List[str]] = []
    period_rets: List[float] = []
    turnover_flags: List[float] = []

    prev_picks: List[str] = []
    for d in rbd:
        # Rank universe at date d by latest score up to d
        scores: List[Tuple[str, float]] = []
        for sym, fac in sym_to_fac.items():
            fac_d = fac[fac['date'] <= d]
            if fac_d.empty:
                continue
            scores.append((sym, _score_latest(fac_d)))
        if not scores:
            selected_each.append([])
            period_rets.append(0.0)
            turnover_flags.append(0.0)
            continue
        scores.sort(key=lambda x: (np.nan_to_num(x[1], nan=-1e9)), reverse=True)
        picks = [s for s, _ in scores[:top_k]]
        selected_each.append(picks)
        # turnover: 比例变动近似（换手率=变化数量/持仓数）
        if prev_picks:
            changed = len(set(prev_picks) ^ set(picks))
            turnover_flags.append(changed / float(top_k))
        else:
            turnover_flags.append(1.0)
        prev_picks = picks

        # Compute next-day returns for picks
        rets = []
        for sym in picks:
            fac = sym_to_fac[sym]
            idx = fac.index[fac['date'] == d]
            if len(idx) == 0:
                fac_d = fac[fac['date'] <= d]
                if fac_d.empty:
                    continue
                i = fac_d.index[-1]
            else:
                i = idx[0]
            if i + 1 < len(fac):
                r = float(fac['close'].pct_change().iloc[i + 1]) if not np.isnan(fac['close'].pct_change().iloc[i + 1]) else 0.0
                rets.append(max(0.0, r - cost))
        port_ret = float(np.mean(rets)) if rets else 0.0
        period_rets.append(port_ret)

    # Build cumulative curve
    cum = 1.0
    curve = []
    levels = []
    wins = 0
    for d, r in zip(rbd, period_rets):
        cum *= (1.0 + r)
        curve.append({'date': d, 'cum': cum})
        levels.append(cum)
        if r > 0:
            wins += 1

    # Benchmark placeholder (None if unavailable)
    bench = None

    summary = {
        'cum_return': cum - 1.0,
        'avg_period_ret': float(np.mean(period_rets)) if period_rets else 0.0,
        'periods': len(rbd),
        'max_drawdown': _max_drawdown(levels),
        'win_rate': (wins / len(period_rets)) if period_rets else 0.0,
        'avg_turnover': float(np.mean(turnover_flags)) if turnover_flags else 0.0
    }

    return {
        'params': {'freq': freq, 'top_k': top_k, 'commission_bps': commission_bps},
        'dates': rbd,
        'selected_each_period': selected_each,
        'period_returns': period_rets,
        'cum_curve': curve,
        'benchmark': bench,
        'summary': summary
    }