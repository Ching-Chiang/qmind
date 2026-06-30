"""execution/base.py 数据类 单元测试"""

from __future__ import annotations

from qmind.execution.base import Balance, OrderResult, Position


class TestOrderResult:
    def test_defaults(self):
        o = OrderResult(order_id="1", symbol="T", side="buy", type="limit",
                        price=100, quantity=1, status="filled")
        assert o.filled_quantity == 0.0
        assert o.avg_fill_price == 0.0

    def test_with_fill(self):
        o = OrderResult(order_id="1", symbol="T", side="buy", type="market",
                        price=100, quantity=1, status="filled",
                        filled_quantity=1.0, avg_fill_price=101.5)
        assert o.avg_fill_price == 101.5


class TestBalance:
    def test_total_is_free_plus_locked(self):
        b = Balance(asset="USDT", free=8000, locked=2000, total=10000)
        assert b.total == 10000


class TestPosition:
    def test_short_position(self):
        p = Position(symbol="BTC/USDT", side="short", quantity=0.5,
                     entry_price=90000, mark_price=85000, pnl_unrealized=2500,
                     leverage=2)
        assert p.pnl_unrealized == 2500
        assert p.leverage == 2
