"""CVRF 学习系统 单元测试"""

from __future__ import annotations

from datetime import datetime

from qmind.graph.state import Lesson, MarketConditionVector, MemoryEntry
from qmind.learning.evaluator import TradeEvaluator, TradeRecord
from qmind.learning.injector import LessonInjector
from qmind.learning.memory import MemoryStore


class TestTradeEvaluator:
    """交易结果评估器测试"""

    def test_long_profit(self):
        record = TradeRecord(
            trade_id="t1", symbol="BTC/USDT", decision="LONG",
            entry_price=80000, exit_price=88000, position_size=0.1,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 16, 0, 0),
        )
        result = TradeEvaluator().evaluate(record)
        assert result.pnl_abs == 800.0  # (88000-80000) * 0.1
        assert result.pnl_pct == 10.0   # (88000-80000)/80000 * 100
        assert result.hold_duration == "8.0h"

    def test_short_loss(self):
        record = TradeRecord(
            trade_id="t2", symbol="ETH/USDT", decision="SHORT",
            entry_price=3500, exit_price=3850, position_size=1.0,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 8, 30, 0),
        )
        result = TradeEvaluator().evaluate(record)
        assert result.pnl_abs == -350.0  # (3500-3850) * 1
        assert result.pnl_pct == -10.0   # (3500-3850)/3500 * 100
        assert result.hold_duration == "30m"

    def test_mae_mfe(self):
        record = TradeRecord(
            trade_id="t3", symbol="SOL/USDT", decision="LONG",
            entry_price=100, exit_price=120, position_size=10,
            entry_time=datetime(2026, 6, 29, 8, 0, 0),
            exit_time=datetime(2026, 6, 29, 12, 0, 0),
            highest_price=130, lowest_price=95,
        )
        result = TradeEvaluator().evaluate(record)
        assert result.mae == -5.0   # (95-100)/100 * 100
        assert result.mfe == 30.0   # (130-100)/100 * 100


class TestMemoryStore:
    """记忆库持久化测试"""

    def setup_method(self):
        self.store = MemoryStore(":memory:")  # 内存数据库

    def test_save_and_count(self):
        entry = MemoryEntry(
            symbol="BTC/USDT",
            market_condition=MarketConditionVector(trend="uptrend", volatility="medium"),
            lessons=[Lesson(lesson="趋势跟踪有效", confidence=0.85, source="entry")],
            trade_outcome={"pnl_pct": 5.0},
        )
        entry_id = self.store.save(entry)
        assert entry_id > 0
        assert self.store.count() == 1

    def test_retrieve_recent(self):
        for i in range(5):
            self.store.save(MemoryEntry(
                symbol=f"TEST{i}",
                lessons=[Lesson(lesson=f"lessons{i}", confidence=0.5)],
                trade_outcome={},
            ))
        recent = self.store.get_recent(limit=3)
        assert len(recent) == 3

    def test_similarity_search(self):
        # 保存几个不同市况的记忆
        cond1 = MarketConditionVector(trend="uptrend", volatility="low", momentum=0.8)
        cond2 = MarketConditionVector(trend="downtrend", volatility="high", momentum=-0.6)
        cond3 = MarketConditionVector(trend="uptrend", volatility="medium", momentum=0.5)

        self.store.save(MemoryEntry(
            symbol="BTC/USDT", market_condition=cond1,
            lessons=[Lesson(lesson="上升趋势中持有", confidence=0.9)],
            trade_outcome={"pnl": 10},
        ))
        self.store.save(MemoryEntry(
            symbol="ETH/USDT", market_condition=cond2,
            lessons=[Lesson(lesson="下跌趋势不抄底", confidence=0.8)],
            trade_outcome={"pnl": -5},
        ))
        self.store.save(MemoryEntry(
            symbol="BTC/USDT", market_condition=cond3,
            lessons=[Lesson(lesson="中等波动谨慎加仓", confidence=0.7)],
            trade_outcome={"pnl": 3},
        ))

        # 搜索类似 uptrend 的教训
        query = MarketConditionVector(trend="uptrend", volatility="low", momentum=0.9)
        results = self.store.search_similar(query, top_k=2)

        assert len(results) == 2  # 应该找到 2 个 uptrend
        # 第一个应该是 cond1 (相似度最高)
        assert results[0][0].market_condition.trend == "uptrend"

    def test_cosine_similarity(self):
        a = [1.0, 0.0, 0.5]
        b = [0.5, 0.0, 1.0]
        sim = MemoryStore._cosine_similarity(a, b)
        assert 0.0 < sim < 1.0

    def test_cosine_similarity_empty(self):
        assert MemoryStore._cosine_similarity([], []) == 0.0
        assert MemoryStore._cosine_similarity([1.0], []) == 0.0


class TestLessonInjector:
    """教训注入器测试"""

    def setup_method(self):
        self.store = MemoryStore(":memory:")
        self.injector = LessonInjector(self.store)

    def test_empty_injection(self):
        cond = MarketConditionVector(trend="uptrend")
        result = self.injector.build_injection(cond)
        assert result == ""  # 空库应返回空

    def test_injection_with_data(self):
        self.store.save(MemoryEntry(
            symbol="BTC/USDT",
            market_condition=MarketConditionVector(trend="uptrend", volatility="low"),
            lessons=[Lesson(lesson="不要 FOMO 追高", confidence=0.85)],
            trade_outcome={"pnl": 5},
        ))

        cond = MarketConditionVector(trend="uptrend", volatility="low")
        result = self.injector.build_injection(cond)
        assert "不要 FOMO 追高" in result
        assert "注意" in result

    def test_inject_into_prompt(self):
        self.store.save(MemoryEntry(
            symbol="BTC/USDT",
            market_condition=MarketConditionVector(trend="uptrend"),
            lessons=[Lesson(lesson="测试教训", confidence=0.7)],
            trade_outcome={},
        ))

        original = "这是原始 prompt"
        enriched = self.injector.inject_into_prompt(
            original,
            MarketConditionVector(trend="uptrend"),
        )
        assert "测试教训" in enriched
        assert "这是原始 prompt" in enriched
