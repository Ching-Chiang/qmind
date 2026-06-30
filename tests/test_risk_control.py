"""risk.py 完整三角风控流程 单元测试 (mock LLM)"""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from qmind.agents.protocol import RiskReview
from qmind.agents.risk import TriangleRiskControl
from qmind.graph.state import MarketData, TradeDecision
from qmind.llm.client import LLMClient


@pytest.fixture
def risk_control():
    return TriangleRiskControl(LLMClient())


@pytest.fixture
def decision():
    return TradeDecision(
        decision="LONG", symbol="BTC/USDT",
        entry={"price": 86200}, stop_loss={"price": 85100},
        position_size_pct=12.5, confidence=0.72,
        risk_reward_ratio=2.5, time_horizon="4h",
    )


class TestTriangleRiskControl:
    async def test_all_approve(self, risk_control, decision):
        """三方都通过"""
        md = MarketData(symbol="BTC/USDT")
        approve = RiskReview(role="aggressive", decision="approve", reason="good")

        for parser_name in ("aggressive_parser", "conservative_parser", "neutral_parser"):
            parser = getattr(risk_control, parser_name)
            parser.parse = AsyncMock(return_value=approve)

        result = await risk_control.review(decision, md)
        assert result.approved is True
        assert result.veto_count == 0

    async def test_one_reject(self, risk_control, decision):
        """一方否决则整体否决"""
        md = MarketData(symbol="BTC/USDT")
        approve = RiskReview(role="aggressive", decision="approve")
        reject = RiskReview(role="conservative", decision="reject", reason="too risky")

        risk_control.aggressive_parser.parse = AsyncMock(return_value=approve)
        risk_control.conservative_parser.parse = AsyncMock(return_value=reject)
        risk_control.neutral_parser.parse = AsyncMock(return_value=approve)

        result = await risk_control.review(decision, md)
        assert result.approved is False
        assert "conservative" in result.vetoed_by

    async def test_all_exceptions_fallback_reject(self, risk_control, decision):
        """三方 LLM 都异常则全部回落为 reject"""
        md = MarketData(symbol="BTC/USDT")
        for parser_name in ("aggressive_parser", "conservative_parser", "neutral_parser"):
            parser = getattr(risk_control, parser_name)
            parser.parse = AsyncMock(side_effect=Exception("API error"))

        result = await risk_control.review(decision, md)
        assert result.approved is False
        assert len(result.vetoed_by) == 3

    async def test_cvar_fail_adds_to_vetoed(self, risk_control, decision):
        """CVaR 不通过时加入否决"""
        md = MarketData(symbol="BTC/USDT")
        approve = RiskReview(role="aggressive", decision="approve")
        for parser_name in ("aggressive_parser", "conservative_parser", "neutral_parser"):
            getattr(risk_control, parser_name).parse = AsyncMock(return_value=approve)

        # 构造一个必然不通过 CVaR 的决策
        decision.position_size_pct = 100
        decision.confidence = 0.8
        decision.entry = {"price": 100}
        decision.stop_loss = {"price": 1}

        result = await risk_control.review(decision, md)
        assert result.approved is False
        assert "cvar_constraint" in result.vetoed_by
