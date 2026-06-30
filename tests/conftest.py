"""pytest 共享 fixtures — 用于所有测试"""

from __future__ import annotations

from datetime import datetime

import pytest

from qmind.data.time_guard import TimeGuard
from qmind.execution.dry_run import DryRunExchange
from qmind.graph.state import OHLCV, MarketData, TradeDecision
from qmind.learning.memory import MemoryStore
from qmind.llm.client import CostTracker, LLMClient
from qmind.llm.router import LLMRouter
from qmind.llm.structured import StructuredParser

# ── Existing fixtures ──

@pytest.fixture
def cost_tracker() -> CostTracker:
    return CostTracker()


@pytest.fixture
def llm_client(cost_tracker: CostTracker) -> LLMClient:
    return LLMClient(cost_tracker=cost_tracker)


@pytest.fixture
def llm_router(llm_client: LLMClient) -> LLMRouter:
    return LLMRouter(client=llm_client)


@pytest.fixture
def structured_parser(llm_client: LLMClient) -> StructuredParser:
    return StructuredParser(client=llm_client, model="claude-sonnet-4-6")


@pytest.fixture
def time_guard() -> TimeGuard:
    return TimeGuard(decision_time=datetime(2026, 6, 29, 12, 0, 0))


# ── New shared fixtures for testing ──

@pytest.fixture
def sample_klines() -> list[OHLCV]:
    """生成 100 根示例 K 线"""
    import math
    klines = []
    for i in range(100):
        klines.append(OHLCV(
            timestamp=1719600000000 + i * 3600000,
            open=100.0 + math.sin(i / 10) * 5,
            high=105.0 + math.sin(i / 10) * 5,
            low=95.0 + math.sin(i / 10) * 5,
            close=100.0 + math.sin(i / 10) * 5 + math.cos(i / 5) * 2,
            volume=1000.0 + math.sin(i / 5) * 200,
        ))
    return klines


@pytest.fixture
def mock_market_data(sample_klines: list[OHLCV]) -> MarketData:
    """带 100 根 1h K 线的 MarketData"""
    return MarketData(symbol="BTC/USDT", klines={"1h": sample_klines})


@pytest.fixture
def empty_market_data() -> MarketData:
    """空 K 线的 MarketData"""
    return MarketData(symbol="BTC/USDT")


@pytest.fixture
def sample_trade_decision() -> TradeDecision:
    """示例交易决策"""
    return TradeDecision(
        decision="LONG",
        symbol="BTC/USDT",
        entry={"type": "limit", "price": 86200.0, "quantity": 0.15},
        stop_loss={"price": 85100.0, "type": "stop_market", "reason": "支撑位跌破"},
        take_profit=[{"price": 89500.0, "ratio": 0.5, "reason": "第一压力位"}],
        position_size_pct=12.5,
        confidence=0.72,
        time_horizon="4h",
        reasoning_chain={"data_cot": "K线在支撑位附近企稳", "concept_cot": "下跌末端反转", "thesis_cot": "建议做多"},
        risk_reward_ratio=2.5,
        max_acceptable_loss_pct=1.15,
    )


@pytest.fixture
def dry_run_exchange() -> DryRunExchange:
    """初始余额 10000 的模拟交易所"""
    return DryRunExchange(initial_balance=10000.0)


@pytest.fixture
def memory_store() -> MemoryStore:
    """:memory: SQLite 记忆库"""
    return MemoryStore(":memory:")
