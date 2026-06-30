"""execution/factory.py 交易所工厂 单元测试"""

from __future__ import annotations

import pytest

from qmind.execution.factory import ExchangeFactory


class TestExchangeFactory:
    def test_dry_run_default(self):
        ex = ExchangeFactory.create("dry_run", dry_run=True)
        assert ex.name == "dry_run"
        assert ex.dry_run is True

    def test_dry_run_with_balance(self):
        ex = ExchangeFactory.create("dry_run", dry_run=True, config={"initial_balance": 50000})
        import asyncio
        bal = asyncio.run(ex.get_balance("USDT"))
        assert bal[0].total == 50000

    def test_dry_run_overrides_unknown_name(self):
        """dry_run=True 时任何名字都返回 DryRunExchange"""
        ex = ExchangeFactory.create("nonexistent_xyz", dry_run=True)
        assert ex.name == "dry_run"
        assert ex.dry_run is True

    def test_live_unknown_exchange_raises(self):
        with pytest.raises(ValueError, match="Unknown exchange"):
            ExchangeFactory.create("nonexistent", dry_run=False)
