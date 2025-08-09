#!/usr/bin/env python3
"""
X(Twitter) 数据抓取与持久化工具
- 使用 snscrape 抓取公开推文（无需官方API Key）
- 可选将数据写入 MongoDB（若可用）
- 提供基于互动的简单打分与过滤
"""
from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import List, Dict, Optional, Iterable

# 日志
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')

# 可选指标
try:
    from tradingagents.utils.metrics import metrics
except Exception:
    metrics = None

# 可选导入：MongoDB
try:
    from tradingagents.config.database_manager import get_mongodb_client, is_mongodb_available
except Exception:  # pragma: no cover
    def get_mongodb_client():
        return None
    def is_mongodb_available() -> bool:
        return False

# 可选导入：snscrape
try:
    import snscrape.modules.twitter as sntwitter  # type: ignore
    SNSCRAPE_AVAILABLE = True
except Exception:  # pragma: no cover
    sntwitter = None
    SNSCRAPE_AVAILABLE = False

# 可选导入：StockUtils 获取A股/公司名称信息
try:
    from tradingagents.utils.stock_utils import StockUtils
except Exception:  # pragma: no cover
    StockUtils = None


@dataclass
class XPost:
    id: str
    date: str
    username: str
    content: str
    url: str
    like_count: int
    retweet_count: int
    reply_count: int
    lang: str
    symbol: Optional[str] = None
    company_name: Optional[str] = None

    def engagement_score(self) -> float:
        # 简单互动分：赞*1 + 转发*2 + 回复*1.5
        return self.like_count * 1.0 + self.retweet_count * 2.0 + self.reply_count * 1.5


def _date(s: str) -> _dt.date:
    return _dt.datetime.strptime(s, "%Y-%m-%d").date()


def build_query_for_a_share(symbol: str, company_name: Optional[str], start_date: str, end_date: str, lang_zh: bool = True) -> str:
    # 关键词：股票代码，公司中文名，常见词汇组合
    parts: List[str] = []
    if company_name:
        parts.append(f'("{company_name}" OR {symbol})')
    else:
        parts.append(f'({symbol})')
    # 指定时间范围（X使用 since: 与 until:）
    parts.append(f'since:{start_date}')
    # until 是排他到达日，+1天保证包含 end_date 当日
    until_date = (_date(end_date) + _dt.timedelta(days=1)).strftime('%Y-%m-%d')
    parts.append(f'until:{until_date}')
    # 语言过滤
    if lang_zh:
        parts.append('lang:zh')
    # 去掉回复(可选)与广告样式关键词（谨慎）
    query = ' '.join(parts)
    logger.debug(f"[X] 构造查询: {query}")
    return query


def scrape_x_posts(query: str, limit: int = 200) -> List[XPost]:
    if not SNSCRAPE_AVAILABLE:
        raise ImportError("未安装 snscrape，请先: pip install snscrape")
    results: List[XPost] = []
    import time as _t
    t0 = _t.perf_counter()
    try:
        scraper = sntwitter.TwitterSearchScraper(query)
        for i, tweet in enumerate(scraper.get_items()):
            if i >= limit:
                break
            try:
                results.append(
                    XPost(
                        id=str(tweet.id),
                        date=tweet.date.strftime('%Y-%m-%d'),
                        username=str(tweet.user.username),
                        content=tweet.rawContent or '',
                        url=f"https://twitter.com/{tweet.user.username}/status/{tweet.id}",
                        like_count=int(getattr(tweet, 'likeCount', 0) or 0),
                        retweet_count=int(getattr(tweet, 'retweetCount', 0) or 0),
                        reply_count=int(getattr(tweet, 'replyCount', 0) or 0),
                        lang=str(getattr(tweet, 'lang', '')),
                    )
                )
            except Exception:
                continue
    except Exception as e:  # pragma: no cover
        logger.error(f"[X] 抓取失败: {e}")
    finally:
        if metrics:
            metrics.inc("x_scrape_requests_total", {"source": "snscrape"})
            metrics.inc("x_posts_total", {}, value=len(results))
            metrics.hist("x_scrape_latency_seconds", (_t.perf_counter()-t0))
    return results


def persist_to_mongodb(posts: Iterable[XPost], symbol: Optional[str], company_name: Optional[str]) -> int:
    if not is_mongodb_available():
        return 0
    client = get_mongodb_client()
    if not client:
        return 0
    import time as _t
    t0 = _t.perf_counter()
    upserts_cnt = 0
    try:
        db = client.get_database('tradingagents')
        col = db.get_collection('x_posts')
        # 索引（幂等）
        try:
            col.create_index('id', unique=True)
            col.create_index([('symbol', 1), ('date', -1)])
        except Exception:
            pass
        ops = []
        for p in posts:
            doc = {
                'id': p.id,
                'date': p.date,
                'username': p.username,
                'content': p.content,
                'url': p.url,
                'like_count': p.like_count,
                'retweet_count': p.retweet_count,
                'reply_count': p.reply_count,
                'lang': p.lang,
                'symbol': symbol,
                'company_name': company_name,
                'engagement_score': p.engagement_score(),
                '_created_at': _dt.datetime.utcnow(),
            }
            ops.append({'update_one': {
                'filter': {'id': p.id},
                'update': {'$set': doc},
                'upsert': True
            }})
        if not ops:
            return 0
        # 批量写入
        from pymongo import UpdateOne  # type: ignore
        bulk_ops = [UpdateOne(o['update_one']['filter'], o['update_one']['update'], upsert=True) for o in ops]
        res = col.bulk_write(bulk_ops, ordered=False)
        upserts = (res.upserted_count or 0) + (res.modified_count or 0)
        upserts_cnt = int(upserts)
        return upserts_cnt
    except Exception as e:  # pragma: no cover
        logger.error(f"[X] MongoDB持久化失败: {e}")
        return 0
    finally:
        if metrics:
            metrics.inc("x_mongo_upserts_total", {}, value=upserts_cnt)
            metrics.hist("x_mongo_latency_seconds", (_t.perf_counter()-t0))


def fetch_and_store_x_for_a_share(symbol: str, start_date: str, end_date: str, limit: int = 200) -> List[XPost]:
    company_name: Optional[str] = None
    try:
        if StockUtils is not None:
            info = StockUtils.get_market_info(symbol)
            if info and info.get('is_china') and info.get('company_name'):
                company_name = info['company_name']
    except Exception:
        pass

    query = build_query_for_a_share(symbol, company_name, start_date, end_date, lang_zh=True)
    posts = scrape_x_posts(query, limit=limit)
    for p in posts:
        p.symbol = symbol
        p.company_name = company_name
    # 可选持久化
    saved = persist_to_mongodb(posts, symbol, company_name)
    if saved:
        logger.info(f"[X] 已写入MongoDB: {saved} 条")
    if metrics:
        metrics.inc("x_pipeline_runs_total", {"market": "A"})
    return posts


def format_posts_as_markdown(posts: List[XPost], top_n: int = 50) -> str:
    if not posts:
        return ""
    # 根据互动度排序，取前N
    posts_sorted = sorted(posts, key=lambda p: p.engagement_score(), reverse=True)[:top_n]
    lines = ["## X平台舆情（按互动度排序）\n"]
    for p in posts_sorted:
        lines.append(f"### @{p.username} · {p.date} · 分数: {p.engagement_score():.1f}")
        lines.append(p.content.replace('\n', ' ').strip())
        lines.append(f"链接: {p.url}")
        lines.append("")
    return "\n".join(lines)