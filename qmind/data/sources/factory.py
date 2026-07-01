"""
数据源工厂 — 统一获取市场数据。
"""

from __future__ import annotations

import logging

from qmind.data.symbol_map import detect_source
from qmind.graph.state import MarketData

logger = logging.getLogger(__name__)


class DataSourceFactory:
    """数据源工厂"""

    async def fetch_market_data(
        self,
        symbol: str,
        source: str = "auto",
        interval: str = "1h",
    ) -> MarketData:
        """自动选择数据源获取市场数据"""
        if source == "auto":
            source = detect_source(symbol)

        try:
            if source == "yfinance":
                from qmind.data.sources.yfinance_source import YFinanceSource
                return await YFinanceSource().fetch_klines(symbol, interval=interval)
            elif source == "tushare":
                from qmind.data.sources.tushare_source import TushareSource
                return await TushareSource().fetch_daily(symbol)
            elif source == "mock":
                from qmind.data.sources.mock_source import MockSource
                return await MockSource().fetch_klines(symbol, interval=interval)
            elif source in ("binance", "crypto"):
                from qmind.data.sources.binance_source import BinanceSource
                bs = BinanceSource()
                try:
                    return await bs.fetch_klines(symbol, interval=interval)
                except Exception as e:
                    logger.warning(
                        f"Binance 数据源不可用: {e}. "
                        f"试试 --source mock (模拟) 或 --source yfinance (美股)"
                    )
                    raise
                finally:
                    await bs.close()
            else:
                raise ValueError(f"Unsupported data source: {source}")
        except Exception as e:
            logger.warning(f"获取 {symbol} 数据失败 ({source}): {e}")
            return MarketData(symbol=symbol)
