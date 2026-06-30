"""
Binance REST + WebSocket 适配器。

使用 ccxt 做 REST 接口，websockets 做实时行情流。
"""

from __future__ import annotations

import time
from typing import Any

from qmind.execution.base import Balance, ExchangeBase, OrderResult, Position


class BinanceExchange(ExchangeBase):
    """Binance 交易所适配器"""

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        dry_run: bool = True,
        testnet: bool = True,
    ):
        super().__init__("binance", dry_run=dry_run)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self._ccxt = None

    @property
    def exchange(self):
        if self._ccxt is None:
            import ccxt.async_support as ccxt
            exchange_class = ccxt.binance if not self.testnet else ccxt.binanceusdm
            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "enableRateLimit": True,
            }
            if self.testnet:
                config["options"] = {"defaultType": "future"}
            self._ccxt = exchange_class(config)
        return self._ccxt

    async def get_price(self, symbol: str) -> float:
        ticker = await self.exchange.fetch_ticker(symbol)
        return ticker["last"]

    async def get_balance(self, asset: str = "") -> list[Balance]:
        ccxt_balance = await self.exchange.fetch_balance()
        result = []
        for currency, data in ccxt_balance["total"].items():
            if data > 0 or (asset and currency == asset):
                free = ccxt_balance["free"].get(currency, 0)
                locked = ccxt_balance["used"].get(currency, 0)
                result.append(Balance(asset=currency, free=free, locked=locked, total=data))
        return result

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        **_kwargs: Any,
    ) -> OrderResult:
        if self.dry_run:
            return OrderResult(
                order_id=f"dry_{int(time.time()*1000)}",
                symbol=symbol, side=side, type=order_type,
                price=price, quantity=quantity,
                status="filled", filled_quantity=quantity, avg_fill_price=price,
            )

        ccxt_side = "buy" if side.lower() == "buy" else "sell"
        ccxt_type = order_type.lower()

        if ccxt_type == "limit":
            order = await self.exchange.create_limit_order(symbol, ccxt_side, quantity, price)
        else:
            order = await self.exchange.create_market_order(symbol, ccxt_side, quantity)

        return OrderResult(
            order_id=order["id"],
            symbol=order["symbol"],
            side=order["side"],
            type=order["type"],
            price=float(order.get("price", 0)),
            quantity=float(order.get("amount", 0)),
            status=order["status"],
            filled_quantity=float(order.get("filled", 0)),
            avg_fill_price=float(order.get("average", 0)),
            raw=order,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, symbol: str, order_id: str) -> OrderResult | None:
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return OrderResult(
                order_id=order["id"],
                symbol=order["symbol"],
                side=order["side"],
                type=order["type"],
                price=float(order.get("price", 0)),
                quantity=float(order.get("amount", 0)),
                status=order["status"],
                filled_quantity=float(order.get("filled", 0)),
                avg_fill_price=float(order.get("average", 0)),
            )
        except Exception:
            return None

    async def get_positions(self, symbol: str = "") -> list[Position]:
        try:
            ccxt_positions = await self.exchange.fetch_positions([symbol] if symbol else [])
            positions = []
            for p in ccxt_positions:
                qty = float(p.get("contracts", 0))
                if qty != 0:
                    positions.append(Position(
                        symbol=p["symbol"],
                        side="long" if qty > 0 else "short",
                        quantity=abs(qty),
                        entry_price=float(p.get("entryPrice", 0)),
                        mark_price=float(p.get("markPrice", 0)),
                        pnl_unrealized=float(p.get("unrealizedPnl", 0)),
                        leverage=int(p.get("leverage", 1)),
                    ))
            return positions
        except Exception:
            return []
