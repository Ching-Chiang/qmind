"""数据模型 单元测试"""

from __future__ import annotations

from datetime import datetime

import pytest
from pydantic import ValidationError

from qmind.graph.state import (
    OHLCV,
    AgentState,
    AnalystReport,
    MarketConditionVector,
    MarketData,
    MemoryEntry,
    TradeDecision,
)


class TestOHLCV:
    def test_basic_ohlcv(self):
        k = OHLCV(timestamp=1719600000000, open=100.0, high=105.0, low=99.0, close=103.5, volume=10000)
        assert k.open == 100.0
        assert k.close == 103.5

    def test_as_of_optional(self):
        k = OHLCV(timestamp=1719600000000, open=1, high=2, low=1, close=1.5, volume=100)
        assert k.as_of is None

        dt = datetime(2026, 6, 29, 12, 0, 0)
        k2 = OHLCV(timestamp=1719600000000, open=1, high=2, low=1, close=1.5, volume=100, as_of=dt)
        assert k2.as_of == dt


class TestMarketData:
    def test_empty_market_data(self):
        md = MarketData(symbol="BTC/USDT")
        assert md.symbol == "BTC/USDT"
        assert md.klines == {}
        assert md.news == []

    def test_with_klines(self):
        md = MarketData(
            symbol="ETH/USDT",
            klines={
                "1h": [OHLCV(timestamp=1719600000000, open=3000, high=3100, low=2980, close=3050, volume=50000)],
                "1d": [],
            },
        )
        assert md.klines["1h"][0].close == 3050


class TestAnalystReport:
    def test_valid_report(self):
        report = AnalystReport(
            analyst="technical",
            stance="bullish",
            confidence=0.75,
            core_reason="MACD 金叉 + 突破阻力位",
            support_price=85000.0,
            resistance_price=90000.0,
        )
        assert report.analyst == "technical"
        assert report.stance == "bullish"

    def test_confidence_range(self):
        with pytest.raises(ValidationError):
            AnalystReport(analyst="test", stance="bullish", confidence=1.5, core_reason="test")
        with pytest.raises(ValidationError):
            AnalystReport(analyst="test", stance="bullish", confidence=-0.1, core_reason="test")


class TestTradeDecision:
    def test_valid_decision(self):
        d = TradeDecision(
            decision="LONG",
            symbol="BTC/USDT",
            entry={"type": "limit", "price": 87200},
            position_size_pct=12.5,
            confidence=0.72,
            time_horizon="4h",
        )
        assert d.decision == "LONG"
        assert d.position_size_pct == 12.5

    def test_hold_decision(self):
        d = TradeDecision(decision="HOLD")
        assert d.decision == "HOLD"
        assert d.position_size_pct == 0.0


class TestMemoryEntry:
    def test_default_factory(self):
        entry = MemoryEntry()
        assert entry.lessons == []
        assert entry.trade_outcome == {}
        assert entry.embedding is None

    def test_with_lessons(self):
        from qmind.graph.state import Lesson
        entry = MemoryEntry(
            symbol="BTC/USDT",
            lessons=[
                Lesson(lesson="趋势突破后不要急于入场", confidence=0.85),
                Lesson(lesson="RSI 超卖区金叉可靠性更高", confidence=0.72),
            ],
            market_condition=MarketConditionVector(trend="uptrend", volatility="medium"),
        )
        assert len(entry.lessons) == 2
        assert entry.market_condition.trend == "uptrend"


class TestAgentState:
    def test_typeddict_fields(self):
        state: AgentState = {
            "symbol": "BTC/USDT",
            "timeframe": "1h",
            "timestamp": 1719600000000,
            "market_data": None,
            "analyses": [],
            "debate": None,
            "disagreement": 0.0,
            "decision": None,
            "risk": None,
            "execution_result": None,
            "evaluation": None,
            "errors": [],
            "debug_info": {},
        }
        assert state["symbol"] == "BTC/USDT"
        assert state["disagreement"] == 0.0
