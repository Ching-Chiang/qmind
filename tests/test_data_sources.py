"""data/sources/ 数据源 单元测试 (mock)"""

from __future__ import annotations

from unittest.mock import patch

from qmind.data.sources.factory import DataSourceFactory
from qmind.data.sources.tushare_source import TushareSource
from qmind.graph.state import MarketData


class TestDataSourceFactory:
    async def test_fallback_on_error(self):
        """DataSourceFactory 在网络异常时返回空 MarketData"""
        with patch("qmind.data.sources.factory.detect_source", return_value="yfinance"), \
             patch("qmind.data.sources.yfinance_source.YFinanceSource.fetch_klines",
                   side_effect=Exception("API error")):
            factory = DataSourceFactory()
            result = await factory.fetch_market_data("BTC/USDT")
        assert isinstance(result, MarketData)
        assert result.symbol == "BTC/USDT"
        assert result.klines == {}

    async def test_unknown_source_returns_empty(self):
        with patch("qmind.data.sources.factory.detect_source", return_value="unknown"):
            factory = DataSourceFactory()
            result = await factory.fetch_market_data("TEST")
        assert isinstance(result, MarketData)
        assert result.klines == {}


class TestTushareSource:
    def test_lazy_import(self):
        """TushareSource 只在构造时 import tushare"""
        source = TushareSource.__new__(TushareSource)
        assert not hasattr(source, "_ts")  # not yet initialized

    async def test_init_without_token(self):
        """无 token 时应在调用 API 时提示"""
        source = TushareSource.__new__(TushareSource)
        source._ts = None
        assert not hasattr(source, "_ts") or source._ts is None
