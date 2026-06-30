"""cvrf.py 反思引擎 + cvrf_pipeline.py 闭环 单元测试 (mock LLM)"""

from __future__ import annotations

from datetime import datetime
from unittest.mock import AsyncMock

import pytest

from qmind.graph.state import Lesson, MarketConditionVector
from qmind.learning.cvrf import CVRFReflection
from qmind.learning.cvrf_pipeline import CVRFPipeline
from qmind.learning.evaluator import TradeRecord
from qmind.learning.memory import MemoryStore
from qmind.llm.client import LLMClient


class TestCVRFReflection:
    @pytest.fixture
    def reflection(self):
        return CVRFReflection(LLMClient())

    async def test_extract_market_condition(self, reflection):
        """mock LLM 返回市况特征"""
        trade = TradeRecord(
            trade_id="t1", symbol="BTC/USDT", decision="LONG",
            entry_price=80000, exit_price=88000, position_size=0.1,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 16, 0, 0),
            highest_price=90000, lowest_price=78000,
        )
        mock_result = MarketConditionVector(trend="uptrend", volatility="medium", momentum=0.7)
        reflection.parser.parse = AsyncMock(return_value=mock_result)

        result = await reflection.extract_market_condition(trade)
        assert result.trend == "uptrend"
        assert result.volatility == "medium"
        assert result.momentum == 0.7

    async def test_reflect_returns_lessons(self, reflection):
        """mock LLM 返回教训列表"""
        from qmind.learning.cvrf import CVRFLessons
        from qmind.graph.state import TradeEvaluation
        trade = TradeRecord(
            trade_id="t1", symbol="BTC/USDT", decision="LONG",
            entry_price=80000, exit_price=88000, position_size=0.1,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 16, 0, 0),
        )
        evaluation = TradeEvaluation(pnl_pct=10.0, hold_duration="8h")
        mock_lessons = CVRFLessons(lessons=[
            Lesson(lesson="趋势中不要过早止盈", confidence=0.85, source="exit"),
            Lesson(lesson="突破确认后再加仓", confidence=0.72, source="entry"),
        ])
        reflection.parser.parse = AsyncMock(return_value=mock_lessons)

        result = await reflection.reflect(trade, evaluation)
        assert len(result) == 2
        assert result[0].lesson == "趋势中不要过早止盈"


class TestCVRFPipeline:
    @pytest.fixture
    def pipeline(self):
        reflection = CVRFReflection(LLMClient())
        memory = MemoryStore(":memory:")
        return CVRFPipeline(reflection, memory)

    async def test_process_trade_full_cycle(self, pipeline):
        """完整闭环：评估 → 反思 → 存储"""
        trade = TradeRecord(
            trade_id="t1", symbol="BTC/USDT", decision="LONG",
            entry_price=80000, exit_price=88000, position_size=0.1,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 16, 0, 0),
            highest_price=90000, lowest_price=78000,
        )

        # Mock both LLM calls
        pipeline.reflection.reflect = AsyncMock(return_value=[
            Lesson(lesson="test lesson", confidence=0.8, source="entry"),
        ])
        pipeline.reflection.extract_market_condition = AsyncMock(
            return_value=MarketConditionVector(trend="uptrend", volatility="medium"),
        )

        entry = await pipeline.process_trade(trade)
        assert entry.id is not None and entry.id > 0
        assert len(entry.lessons) == 1
        assert entry.lessons[0].lesson == "test lesson"
        assert entry.trade_outcome["pnl_pct"] == 10.0  # (88000-80000)/80000*100
        assert pipeline.memory.count() == 1

    async def test_batch_process_isolates_failures(self, pipeline):
        """批量处理中单笔失败不影响其他"""
        t1 = TradeRecord(
            trade_id="t1", symbol="BTC/USDT", decision="LONG",
            entry_price=80000, exit_price=88000, position_size=0.1,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 16, 0, 0),
        )
        t2 = TradeRecord(
            trade_id="t2", symbol="ETH/USDT", decision="SHORT",
            entry_price=3000, exit_price=2800, position_size=2.0,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 12, 0, 0),
        )

        # Mock: t1 succeeds, t2 fails
        pipeline.reflection.reflect = AsyncMock(side_effect=[
            [Lesson(lesson="ok", confidence=0.5, source="entry")],
            Exception("LLM error"),
        ])
        pipeline.reflection.extract_market_condition = AsyncMock(
            return_value=MarketConditionVector(trend="sideways"),
        )

        entries = await pipeline.batch_process([t1, t2])
        assert len(entries) == 1  # t2 failed
        assert entries[0].symbol == "BTC/USDT"
