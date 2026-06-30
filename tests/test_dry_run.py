"""execution/dry_run.py 模拟交易所 单元测试"""

from __future__ import annotations

import pytest

from qmind.execution.dry_run import DryRunExchange


class TestDryRunExchange:
    @pytest.fixture
    def ex(self):
        return DryRunExchange(initial_balance=10000.0)

    @pytest.mark.asyncio
    async def test_initial_balance(self, ex):
        bal = await ex.get_balance("USDT")
        assert bal[0].free == 10000.0
        assert bal[0].total == 10000.0

    @pytest.mark.asyncio
    async def test_unknown_price_returns_zero(self, ex):
        assert await ex.get_price("UNKNOWN") == 0.0

    @pytest.mark.asyncio
    async def test_update_price(self, ex):
        ex.update_price("BTC/USDT", 86200)
        assert await ex.get_price("BTC/USDT") == 86200.0

    @pytest.mark.asyncio
    async def test_buy_reduces_balance(self, ex):
        ex.update_price("BTC/USDT", 86200)
        await ex.place_order("BTC/USDT", "buy", "limit", 0.1, 86200)
        bal = await ex.get_balance("USDT")
        assert bal[0].free == pytest.approx(10000 - 8620)

    @pytest.mark.asyncio
    async def test_sell_profit(self, ex):
        ex.update_price("BTC/USDT", 86200)
        await ex.place_order("BTC/USDT", "buy", "limit", 0.1, 86200)
        await ex.place_order("BTC/USDT", "sell", "market", 0.1, 87000)
        bal = await ex.get_balance("USDT")
        assert bal[0].free > 10000.0

    @pytest.mark.asyncio
    async def test_cancel_order(self, ex):
        order = await ex.place_order("BTC/USDT", "buy", "limit", 0.1, 86200)
        assert await ex.cancel_order("BTC/USDT", order.order_id) is True
        cancelled = await ex.get_order("BTC/USDT", order.order_id)
        assert cancelled.status == "cancelled"

    @pytest.mark.asyncio
    async def test_cancel_nonexistent(self, ex):
        assert await ex.cancel_order("BTC/USDT", "nonexistent") is False

    @pytest.mark.asyncio
    async def test_positions_after_buy(self, ex):
        ex.update_price("BTC/USDT", 86200)
        await ex.place_order("BTC/USDT", "buy", "limit", 0.15, 86200)
        positions = await ex.get_positions("BTC/USDT")
        assert len(positions) == 1
        assert positions[0].quantity == 0.15
        assert positions[0].side == "long"
