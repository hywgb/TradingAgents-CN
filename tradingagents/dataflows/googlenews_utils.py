import json
from bs4 import BeautifulSoup
from datetime import datetime
import asyncio
import random

# 导入日志模块
from tradingagents.utils.logging_manager import get_logger
logger = get_logger('agents')
try:
    from tradingagents.dataflows.cache_utils import cache_get_json, cache_set_json
except Exception:
    cache_get_json = cache_set_json = None
try:
    from tradingagents.utils.metrics import metrics
except Exception:
    metrics = None


async def _fetch_page(url: str, headers):
    # Random jitter to avoid detection
    await asyncio.sleep(random.uniform(0.5, 1.5))
    from .http_client import get_http_client
    client = await get_http_client()
    import time
    t0 = time.perf_counter()
    resp = await client.get(url, headers=headers, max_attempts=5)
    t1 = time.perf_counter()
    try:
        from tradingagents.utils.metrics import metrics as _m
        _m.hist('http_latency_seconds', {'host': 'www.google.com', 'path': 'news'}, t1 - t0)
    except Exception:
        pass
    return resp


def getNewsData(query, start_date, end_date):
    """
    Scrape Google News search results for a given query and date range.
    query: str - search query
    start_date: str - start date in the format yyyy-mm-dd or mm/dd/yyyy
    end_date: str - end date in the format yyyy-mm-dd or mm/dd/yyyy
    """
    if "-" in start_date:
        start_date = datetime.strptime(start_date, "%Y-%m-%d")
        start_date = start_date.strftime("%m/%d/%Y")
    if "-" in end_date:
        end_date = datetime.strptime(end_date, "%Y-%m-%d")
        end_date = end_date.strftime("%m/%d/%Y")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/101.0.4951.54 Safari/537.36"
        )
    }

    # 先查缓存
    cache_key = f"news:google:{query}:{start_date}:{end_date}"
    if cache_get_json:
        cached = cache_get_json(cache_key)
        if cached:
            if metrics:
                metrics.inc("cache_hit_total", {"cache": "google_news"})
            return cached
        else:
            if metrics:
                metrics.inc("cache_miss_total", {"cache": "google_news"})
    news_results = []
    async def _run():
        page = 0
        while True:
            offset = page * 10
            url = (
                f"https://www.google.com/search?q={query}"
                f"&tbs=cdr:1,cd_min:{start_date},cd_max:{end_date}"
                f"&tbm=nws&start={offset}"
            )
            try:
                response = await _fetch_page(url, headers)
                soup = BeautifulSoup(response.content, "html.parser")
                results_on_page = soup.select("div.SoaBEf")
                if not results_on_page:
                    break
                for el in results_on_page:
                    try:
                        link = el.find("a")["href"]
                        title = el.select_one("div.MBeuO").get_text()
                        snippet = el.select_one(".GI74Re").get_text()
                        date = el.select_one(".LfVVr").get_text()
                        source = el.select_one(".NUnG9d span").get_text()
                        news_results.append({
                            "link": link, "title": title, "snippet": snippet, "date": date, "source": source
                        })
                    except Exception as e:
                        logger.error(f"Error processing result: {e}")
                        continue
                next_link = soup.find("a", id="pnnext")
                if not next_link:
                    break
                page += 1
            except Exception as e:
                logger.error(f"获取Google新闻失败: {e}")
                break
    try:
        asyncio.run(_run())
    except RuntimeError:
        # 如果外部已有事件循环（如在某些环境），则使用新loop
        loop = asyncio.new_event_loop()
        loop.run_until_complete(_run())
        loop.close()
    # 写缓存
    if cache_set_json:
        try:
            cache_set_json(cache_key, news_results)
        except Exception:
            pass
    return news_results
