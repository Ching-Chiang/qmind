"""execution/validator.py 下单校验器 单元测试"""

from __future__ import annotations

import pytest

from qmind.execution.dry_run import DryRunExchange
from qmind.execution.validator import OrderValidator
from qmind.graph.state import TradeDecision


@pytest.fixture
def validator():
    return OrderValidator(DryRunExchange(10000))


class TestOrderValidator:
    @pytest.mark.asyncio
    async def test_valid_long(self, validator):
        d = TradeDecision(decision="LONG", symbol="BTC/USDT",
                          entry={"price": 86200}, stop_loss={"price": 85100},
                          position_size_pct=12.5, confidence=0.72, risk_reward_ratio=2.5)
        assert (await validator.validate(d)).valid is True

    @pytest.mark.asyncio
    async def test_hold_skipped(self, validator):
        assert (await validator.validate(TradeDecision(decision="HOLD"))).valid is True

    @pytest.mark.asyncio
    async def test_zero_entry_price_rejected(self, validator):
        d = TradeDecision(decision="LONG", entry={"price": 0}, stop_loss={"price": 95})
        assert (await validator.validate(d)).valid is False

    @pytest.mark.asyncio
    async def test_zero_position_rejected(self, validator):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 95},
                          position_size_pct=0)
        assert (await validator.validate(d)).valid is False

    @pytest.mark.asyncio
    async def test_exceeds_max_position(self, validator):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 95},
                          position_size_pct=50, confidence=0.8, risk_reward_ratio=2.0)
        result = await validator.validate(d)
        assert result.valid is False
        assert "超过" in result.reason

    @pytest.mark.asyncio
    async def test_low_risk_reward(self, validator):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 99},
                          position_size_pct=10, confidence=0.8, risk_reward_ratio=0.5)
        assert (await validator.validate(d)).valid is False
