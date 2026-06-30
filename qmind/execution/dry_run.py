"""
DryRun 执行器 — 模拟下单，不实际执行交易。

用于 Phase 5 前的开发测试。
"""

from __future__ import annotations

import uuid
from typing import Any

from qmind.execution.base import Balance, ExchangeBase, OrderResult, Position


class DryRunExchange(ExchangeBase):
    """模拟交易所 — 不下实单"""

    def __init__(self, initial_balance: float = 10000.0):
        super().__init__("dry_run", dry_run=True)
        self._balance: dict[str, Balance] = {
            "USDT": Balance(asset="USDT", free=initial_balance, locked=0.0, total=initial_balance),
        }
        self._positions: dict[str, Position] = {}
        self._orders: dict[str, OrderResult] = {}
        self._prices: dict[str, float] = {}

    async def get_price(self, symbol: str) -> float:
        return self._prices.get(symbol, 0.0)

    async def get_balance(self, asset: str = "") -> list[Balance]:
        if asset:
            return [self._balance.get(asset, Balance(asset=asset, free=0, locked=0, total=0))]
        return list(self._balance.values())

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        **_kwargs: Any,
    ) -> OrderResult:
        order_id = f"dry_{uuid.uuid4().hex[:12]}"
        fill_price = price if price > 0 else self._prices.get(symbol, 0.0)

        result = OrderResult(
            order_id=order_id,
            symbol=symbol,
            side=side,
            type=order_type,
            price=fill_price,
            quantity=quantity,
            status="filled",
            filled_quantity=quantity,
            avg_fill_price=fill_price,
        )
        self._orders[order_id] = result

        # 模拟持仓更新
        cost = fill_price * quantity
        if side == "buy":
            self._balance["USDT"] = Balance(
                asset="USDT",
                free=self._balance["USDT"].free - cost,
                locked=self._balance["USDT"].locked,
                total=self._balance["USDT"].total - cost,
            )
            existing = self._positions.get(symbol)
            new_qty = existing.quantity + quantity if existing else quantity
            self._positions[symbol] = Position(
                symbol=symbol, side="long", quantity=new_qty,
                entry_price=fill_price, mark_price=fill_price, pnl_unrealized=0,
            )
        else:  # sell
            self._balance["USDT"] = Balance(
                asset="USDT",
                free=self._balance["USDT"].free + cost,
                locked=self._balance["USDT"].locked,
                total=self._balance["USDT"].total + cost,
            )

        return result

    async def cancel_order(self, _symbol: str, order_id: str) -> bool:
        order = self._orders.get(order_id)
        if order:
            order.status = "cancelled"
            return True
        return False

    async def get_order(self, _symbol: str, order_id: str) -> OrderResult | None:
        return self._orders.get(order_id)

    async def get_positions(self, symbol: str = "") -> list[Position]:
        if symbol:
            pos = self._positions.get(symbol)
            return [pos] if pos else []
        return list(self._positions.values())
