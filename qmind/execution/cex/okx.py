"""
OKX REST + WebSocket 适配器。

使用 ccxt 做 REST 接口，websockets 做实时行情流。
OKX 特有差异：
  - passphrase 作为第三凭据（api_key / api_secret / passphrase）
  - 测试网通过 ccxt.okx testnet 标志启用
  - 订单类型名称与 Binance 兼容（market / limit）
  - WS 频道命名不同（tickers / candles / instruments）
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from qmind.execution.base import Balance, ExchangeBase, OrderResult, Position


class OKXExchange(ExchangeBase):
    """OKX 交易所适配器"""

    REST_BASE_MAINNET = "https://www.okx.com"
    REST_BASE_TESTNET = "https://www.okx.cab"
    WS_URL_MAINNET = "wss://ws.okx.com:8443/ws/v5/public"
    WS_URL_TESTNET = "wss://wspap.okx.com:8443/ws/v5/public"
    WS_URL_PRIVATE_MAINNET = "wss://ws.okx.com:8443/ws/v5/private"
    WS_URL_PRIVATE_TESTNET = "wss://wspap.okx.com:8443/ws/v5/private"

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        passphrase: str = "",
        testnet: bool = True,
        dry_run: bool = True,
    ):
        super().__init__("okx", dry_run=dry_run)
        self.api_key = api_key
        self.api_secret = api_secret
        self.passphrase = passphrase
        self.testnet = testnet
        self._ccxt = None
        self._ws = None
        self._ws_private = None
        self._ws_subscriptions: dict[str, asyncio.Event] = {}
        self._ws_prices: dict[str, float] = {}
        self._ws_lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # ccxt 懒加载
    # ------------------------------------------------------------------

    @property
    def exchange(self):
        """懒初始化 ccxt.async_support.okx 实例。"""
        if self._ccxt is None:
            import ccxt.async_support as ccxt

            config: dict[str, Any] = {
                "apiKey": self.api_key,
                "secret": self.api_secret,
                "password": self.passphrase,
                "enableRateLimit": True,
                "options": {
                    "defaultType": "swap",  # 永续合约优先
                },
            }
            if self.testnet:
                config["urls"] = {
                    "api": {
                        "rest": self.REST_BASE_TESTNET,
                    }
                }
            self._ccxt = ccxt.okx(config)
        return self._ccxt

    # ------------------------------------------------------------------
    # 核心 REST 接口
    # ------------------------------------------------------------------

    async def get_price(self, symbol: str) -> float:
        """获取当前最新成交价。

        Args:
            symbol: 交易对，如 ``BTC/USDT``。

        Returns:
            最新成交价。
        """
        ticker = await self.exchange.fetch_ticker(symbol)
        return float(ticker["last"])

    async def get_balance(self, asset: str = "") -> list[Balance]:
        """获取账户余额。

        Args:
            asset: 筛选指定资产（可选）。

        Returns:
            Balance 列表。
        """
        ccxt_balance = await self.exchange.fetch_balance()
        result: list[Balance] = []
        for currency, total in ccxt_balance.get("total", {}).items():
            if total and total > 0 or (asset and currency == asset):
                free = ccxt_balance.get("free", {}).get(currency, 0.0)
                locked = ccxt_balance.get("used", {}).get(currency, 0.0)
                result.append(
                    Balance(
                        asset=currency,
                        free=float(free),
                        locked=float(locked),
                        total=float(total),
                    )
                )
        return result

    async def place_order(
        self,
        symbol: str,
        side: str,
        order_type: str,
        quantity: float,
        price: float = 0.0,
        **kwargs: Any,
    ) -> OrderResult:
        """下单。

        Args:
            symbol: 交易对。
            side: ``buy`` 或 ``sell``。
            order_type: ``market`` 或 ``limit``。
            quantity: 数量（合约张数或币数）。
            price: 限价单价格（市价单可省略）。
            **kwargs: 透传给 ccxt 的额外参数，如 ``{"marginMode": "cross"}``。

        Returns:
            OrderResult。
        """
        if self.dry_run:
            return OrderResult(
                order_id=f"dry_{int(time.time() * 1000)}",
                symbol=symbol,
                side=side,
                type=order_type,
                price=price,
                quantity=quantity,
                status="filled",
                filled_quantity=quantity,
                avg_fill_price=price,
            )

        ccxt_side = "buy" if side.lower() == "buy" else "sell"
        ccxt_type = order_type.lower()

        # OKX 限价单明确传递 price，市价单走 create_market_order
        order: dict[str, Any]
        if ccxt_type == "limit":
            order = await self.exchange.create_limit_order(
                symbol, ccxt_side, quantity, price, params=kwargs
            )
        else:
            order = await self.exchange.create_market_order(
                symbol, ccxt_side, quantity, params=kwargs
            )

        return OrderResult(
            order_id=str(order["id"]),
            symbol=str(order["symbol"]),
            side=str(order["side"]),
            type=str(order["type"]),
            price=float(order.get("price", 0.0) or 0.0),
            quantity=float(order.get("amount", 0.0)),
            status=str(order["status"]),
            filled_quantity=float(order.get("filled", 0.0)),
            avg_fill_price=float(order.get("average", 0.0) or 0.0),
            raw=order,
        )

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """撤单。

        Args:
            symbol: 交易对。
            order_id: 订单 ID。

        Returns:
            是否成功。
        """
        try:
            await self.exchange.cancel_order(order_id, symbol)
            return True
        except Exception:
            return False

    async def get_order(self, symbol: str, order_id: str) -> OrderResult | None:
        """查询订单状态。

        Args:
            symbol: 交易对。
            order_id: 订单 ID。

        Returns:
            OrderResult，订单不存在或异常时返回 ``None``。
        """
        try:
            order = await self.exchange.fetch_order(order_id, symbol)
            return OrderResult(
                order_id=str(order["id"]),
                symbol=str(order["symbol"]),
                side=str(order["side"]),
                type=str(order["type"]),
                price=float(order.get("price", 0.0) or 0.0),
                quantity=float(order.get("amount", 0.0)),
                status=str(order["status"]),
                filled_quantity=float(order.get("filled", 0.0)),
                avg_fill_price=float(order.get("average", 0.0) or 0.0),
                raw=order,
            )
        except Exception:
            return None

    async def get_positions(self, symbol: str = "") -> list[Position]:
        """获取持仓列表。

        Args:
            symbol: 可选，筛选指定交易对。

        Returns:
            Position 列表。
        """
        try:
            symbols = [symbol] if symbol else []
            ccxt_positions = await self.exchange.fetch_positions(symbols)
        except Exception:
            return []

        positions: list[Position] = []
        for p in ccxt_positions:
            qty = float(p.get("contracts", 0.0) or 0.0)
            if qty == 0.0:
                continue
            side_raw = str(p.get("side", "long"))
            positions.append(
                Position(
                    symbol=str(p["symbol"]),
                    side="long" if side_raw == "long" else "short",
                    quantity=abs(qty),
                    entry_price=float(p.get("entryPrice", 0.0) or 0.0),
                    mark_price=float(p.get("markPrice", 0.0) or 0.0),
                    pnl_unrealized=float(p.get("unrealizedPnl", 0.0) or 0.0),
                    leverage=int(p.get("leverage", 1) or 1),
                )
            )
        return positions

    # ------------------------------------------------------------------
    # WebSocket 实时行情
    # ------------------------------------------------------------------

    async def _ensure_ws(self) -> None:
        """确保公共 WebSocket 连接已建立。"""
        if self._ws is not None and not self._ws.closed:
            return

        import websockets as ws

        url = self.WS_URL_TESTNET if self.testnet else self.WS_URL_MAINNET
        self._ws = await ws.connect(url, ping_interval=20, ping_timeout=10)

        # 启动后台消息分发协程
        asyncio.ensure_future(self._ws_dispatch())

    async def _ws_dispatch(self) -> None:
        """从 WS 读取消息并分发给对应的等待者。"""
        import orjson

        ws = self._ws
        if ws is None:
            return

        try:
            async for raw in ws:
                try:
                    msg = orjson.loads(raw)
                except Exception:
                    continue

                # OKX 推送格式: {"arg": {"channel": "...", "instId": "..."}, "data": [...]}
                arg = msg.get("arg", {})
                channel = arg.get("channel", "")
                inst_id = arg.get("instId", "")

                if channel == "tickers" and inst_id:
                    data = msg.get("data", [])
                    if data:
                        last_str = data[0].get("last", "")
                        try:
                            self._ws_prices[inst_id] = float(last_str)
                        except (ValueError, TypeError):
                            pass

                    # 通知等待者
                    event = self._ws_subscriptions.get(inst_id)
                    if event and not event.is_set():
                        event.set()
        except Exception:
            pass  # 连接断开后由 _ensure_ws 重建

    async def subscribe_ticker(self, symbol: str) -> None:
        """订阅实时 ticker。

        Args:
            symbol: 交易对，如 ``BTC/USDT``。

        Notes:
            OKX WS 使用 ``instId`` 格式（无斜杠），但 SDK 内部会做转换。
            订阅成功后可通过 ``self._ws_prices[symbol]`` 读取最新价格。
        """
        await self._ensure_ws()
        inst_id = symbol.replace("/", "-")

        payload = {
            "op": "subscribe",
            "args": [
                {
                    "channel": "tickers",
                    "instId": inst_id,
                }
            ],
        }

        async with self._ws_lock:
            event = asyncio.Event()
            self._ws_subscriptions[inst_id] = event

            ws = self._ws
            if ws is None:
                return

            import orjson

            await ws.send(orjson.dumps(payload))
            # 等待第一条 ticker 数据到达，超时 5 秒
            try:
                await asyncio.wait_for(event.wait(), timeout=5.0)
            except asyncio.TimeoutError:
                pass

    async def get_ws_price(self, symbol: str) -> float | None:
        """获取 WebSocket 推送的最新价格。

        Args:
            symbol: 交易对。

        Returns:
            最新价格，若尚未收到推送则返回 ``None``。
        """
        return self._ws_prices.get(symbol)

    async def close(self) -> None:
        """关闭所有网络连接。"""
        if self._ccxt is not None:
            await self._ccxt.close()
            self._ccxt = None

        if self._ws is not None and not self._ws.closed:
            await self._ws.close()
            self._ws = None

        if self._ws_private is not None and not self._ws_private.closed:
            await self._ws_private.close()
            self._ws_private = None
