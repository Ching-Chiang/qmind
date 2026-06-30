"""tools/portfolio.py 持仓工具 单元测试"""

from __future__ import annotations

import pytest

from qmind.tools.portfolio import PortfolioTool


class TestPortfolioTool:
    @pytest.fixture
    def pt(self):
        return PortfolioTool(dry_run=True, initial_balance=5000)

    async def test_initial_balance(self, pt):
        bal = await pt.get_balance()
        assert bal["total_balance"] == 5000
        assert bal["mode"] == "dry_run"

    async def test_empty_positions(self, pt):
        positions = await pt.get_positions()
        assert positions == []

    async def test_empty_pnl(self, pt):
        pnl = await pt.get_pnl()
        assert pnl["realized_pnl"] == 0.0

    async def test_empty_history(self, pt):
        history = await pt.get_trade_history()
        assert history == []
