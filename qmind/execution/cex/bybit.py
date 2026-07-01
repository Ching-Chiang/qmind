"""
Bybit REST + WebSocket adapter.

Uses ccxt for REST API with support for:
- Spot and linear perpetual (USDT-margined) markets
- Unified margin/contract account
- Testnet via sandbox mode
- Dry-run simulation
"""

from __future__ import annotations

import time
from typing import Any

from qmind.execution.base import Balance, ExchangeBase, OrderResult, Position


class BybitExchange(ExchangeBase):
    """Bybit exchange adapter.

    Supports spot trading and USDT linear perpetual contracts through ccxt.
    Testnet is enabled via sandbox mode (``set_sandbox_mode(True)``), which
    reroutes all requests to ``https://api-testnet.bybit.com/``.

    Parameters
    ----------
    api_key:
        Bybit API key. Can be empty for public endpoints in dry-run mode.
    api_secret:
        Bybit API secret. Can be empty for public endpoints in dry-run mode.
    testnet:
        If True, enable Bybit testnet via sandbox mode.
        Default is True to prevent accidental mainnet trading.
    dry_run:
        If True, ``place_order`` returns a simulated filled order without
        hitting the exchange API. Default is True.
    market_type:
        Market type to trade: ``"linear"`` (USDT perpetual, default),
        ``"spot"``, or ``"inverse"`` (coin-M perpetual).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        testnet: bool = True,
        dry_run: bool = True,
        market_type: str = "linear",
    ):
        super().__init__("bybit", dry_run=dry_run)
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet
        self.market_type = market_type
        self._ccxt: Any = None

    # ------------------------------------------------------------------
    # ccxt instance (lazy-initialised)
    # ------------------------------------------------------------------

    @property
    def exchange(self) -> Any:
        """Lazy-initialised ccxt async Bybit instance.

        Once created, the instance is cached for the lifetime of this adapter.
        Sandbox mode (testnet) is enabled before use.
        """
        if self._ccxt is not None:
            return self._ccxt

        import ccxt.async_support as ccxt

        config: dict[str, Any] = {
            "apiKey": self.api_key,
            "secret": self.api_secret,
            "enableRateLimit": True,
            "rateLimit": 50,  # 50 ms between requests for Bybit
        }

        # Market-type-specific config
        if self.market_type == "linear":
            config["options"] = {"defaultType": "linear"}
        elif self.market_type == "spot":
            config["options"] = {"defaultType": "spot"}
        elif self.market_type == "inverse":
            config["options"] = {"defaultType": "inverse"}

        exchange: Any = ccxt.bybit(config)

        if self.testnet:
            exchange.set_sandbox_mode(True)

        self._ccxt = exchange
        return self._ccxt

    # ------------------------------------------------------------------
    # Market data
    # ------------------------------------------------------------------

    async def get_price(self, symbol: str) -> float:
        """Fetch the latest traded price for *symbol*.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``, ``"ETH/USDT"``.

        Returns
        -------
        float
            Last traded price.
        """
        ticker = await self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    # ------------------------------------------------------------------
    # Account / balance
    # ------------------------------------------------------------------

    async def get_balance(self, asset: str = "") -> list[Balance]:
        """Fetch wallet balances.

        For the unified margin account (``linear`` / ``inverse`` market types)
        this returns the cross-margin wallet balance. For ``spot`` it returns
        the spot wallet balance.

        Parameters
        ----------
        asset:
            If provided, only return balances for this asset. When empty,
            returns all non-zero balances.

        Returns
        -------
        list[Balance]
            List of wallet balances.
        """
        ccxt_balance = await self.exchange.fetch_balance()
        result: list[Balance] = []
        for currency, total in ccxt_balance["total"].items():
            if total is not None and (total > 0 or (asset and currency.upper() == asset.upper())):
                free = ccxt_balance["free"].get(currency, 0) or 0
                locked = ccxt_balance["used"].get(currency, 0) or 0
                result.append(
                    Balance(
                        asset=currency,
                        free=float(free),
                        locked=float(locked),
                        total=float(total),
                    )
                )
        return result

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        **kwargs: Any,
    ) -> OrderResult:
        """Place an order on Bybit.

        Parameters
        ----------
        symbol:
            Trading pair, e.g. ``"BTC/USDT"``.
        side:
            ``"buy"`` or ``"sell"``.
        order_type:
            ``"market"`` or ``"limit"``.
        quantity:
            For linear perpetual: contract quantity in USD (not coin amount).
            For spot: base-currency amount.
        price:
            Limit price (ignored for market orders).
        **kwargs:
            Additional ccxt order parameters — e.g. ``{"reduceOnly": True}``,
            ``{"positionIdx": 0}`` for hedge mode, ``{"stopPrice": ...}``.

        Returns
        -------
        OrderResult
            Filled order info (simulated if ``dry_run`` is True).
        """
        ccxt_side: str = side.lower()
        ccxt_type: str = order_type.lower()

        # -- Dry-run: return a fake filled order --------------------------
        if self.dry_run:
            order_id = f"dry_{int(time.time() * 1000)}_{self.name}"
            fill_price = price if ccxt_type == "limit" else 0.0
            return OrderResult(
                order_id=order_id,
                symbol=symbol,
                side=ccxt_side,
                type=ccxt_type,
                price=fill_price,
                quantity=quantity,
                status="filled",
                filled_quantity=quantity,
                avg_fill_price=fill_price,
            )

        # -- Live order ---------------------------------------------------
        if ccxt_type == "limit":
            if price <= 0:
                raise ValueError("Limit orders require a valid price > 0")
            order = await self.exchange.create_limit_order(
                symbol, ccxt_side, quantity, price, kwargs
            )
        else:
            order = await self.exchange.create_market_order(
                symbol, ccxt_side, quantity, kwargs
            )

        return OrderResult(
            order_id=str(order["id"]),
            symbol=str(order["symbol"]),
            side=str(order["side"]),
            type=str(order["type"]),
            price=float(order.get("price", 0) or 0),
            quantity=float(order.get("amount", 0) or 0),
            status=str(order["status"]),
            filled_quantity=float(order.get("filled", 0) or 0),
            avg_fill_price=float(order.get("average", 0) or 0),
            raw=order,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an open order.

        Parameters
        ----------
        symbol:
            Trading pair the order belongs to.
        order_id:
            Exchange order ID to cancel.

        Returns
        -------
        bool
            True if cancellation succeeded, False otherwise.
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, symbol: str, order_id: str) -> OrderResult | None:
        """Fetch a single order by ID.

        Parameters
        ----------
        symbol:
            Trading pair the order belongs to.
        order_id:
            Exchange order ID.

        Returns
        -------
        OrderResult or None
            Order details, or None if the order is not found / an error
            occurred.
        """
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return OrderResult(
                order_id=str(order["id"]),
                symbol=str(order["symbol"]),
                side=str(order["side"]),
                type=str(order["type"]),
                price=float(order.get("price", 0) or 0),
                quantity=float(order.get("amount", 0) or 0),
                status=str(order["status"]),
                filled_quantity=float(order.get("filled", 0) or 0),
                avg_fill_price=float(order.get("average", 0) or 0),
            )
        except Exception:
            return None

    # ------------------------------------------------------------------
    # Positions (contracts only)
    # ------------------------------------------------------------------

    async def get_positions(self, symbol: str = "") -> list[Position]:
        """Fetch open positions.

        Only meaningful for perpetual contract accounts (``linear`` /
        ``inverse``). Returns an empty list for spot markets.

        Parameters
        ----------
        symbol:
            If provided, only return positions for this symbol. Otherwise
            return all open positions.

        Returns
        -------
        list[Position]
            Open positions with current mark-to-market data.
        """
        try:
            symbols = [symbol] if symbol else []
            ccxt_positions = await self.exchange.fetch_positions(symbols)
        except Exception:
            return []

        positions: list[Position] = []
        for p in ccxt_positions:
            qty = float(p.get("contracts", 0) or 0)
            if qty == 0:
                continue

            positions.append(
                Position(
                    symbol=str(p["symbol"]),
                    side="long" if qty > 0 else "short",
                    quantity=abs(qty),
                    entry_price=float(p.get("entryPrice", 0) or 0),
                    mark_price=float(p.get("markPrice", 0) or 0),
                    pnl_unrealized=float(p.get("unrealizedPnl", 0) or 0),
                    leverage=int(p.get("leverage", 1) or 1),
                )
            )

        return positions

    # ------------------------------------------------------------------
    # Lifecycle helpers
    # ------------------------------------------------------------------

    async def close(self) -> None:
        """Close the underlying ccxt exchange session.

        Must be called when the adapter is no longer needed to release
        network resources (aiohttp connector pool).
        """
        if self._ccxt is not None:
            await self._ccxt.close()
            self._ccxt = None
