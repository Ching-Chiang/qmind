"""
数据源适配器 — Binance API (加密货币)。

使用 ccxt 的 Binance 实现获取现货和合约 K 线数据。
所有数据附带 as_of 时间戳防止 Point-in-Time 偏差。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qmind.graph.state import MarketData, OHLCV

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h",
    "1d": "1d",
}


class BinanceSource:
    """Binance 数据源适配器"""

    def __init__(self) -> None:
        self._exchange: Any = None

    @property
    def exchange(self) -> Any:
        if self._exchange is None:
            import ccxt.async_support as ccxt
            import aiohttp
            connector = aiohttp.TCPConnector(ssl=True, force_close=True)
            self._exchange = ccxt.binance({
                "enableRateLimit": True,
                "options": {"defaultType": "spot"},
                "aiohttp_connector": connector,
                "proxies": {"http": "", "https": ""},
            })
        return self._exchange

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> MarketData:
        """获取 Binance K 线数据"""
        ccxt_symbol = self._normalize_symbol(symbol)
        tf = _INTERVAL_MAP.get(interval, "1h")
        raw = await self.exchange.fetch_ohlcv(ccxt_symbol, tf, limit=limit)

        now = datetime.utcnow()
        klines = [
            OHLCV(
                timestamp=int(k[0]),
                open=float(k[1]),
                high=float(k[2]),
                low=float(k[3]),
                close=float(k[4]),
                volume=float(k[5]),
                as_of=now,
            )
            for k in raw
        ]

        return MarketData(
            symbol=ccxt_symbol,
            klines={interval: klines},
            timestamp=int(datetime.utcnow().timestamp() * 1000),
            as_of=now,
        )

    async def close(self) -> None:
        if self._exchange:
            await self._exchange.close()
            self._exchange = None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        """统一交易对格式为 Binance 标准"""
        s = symbol.upper().strip()
        if "/" not in s:
            # USDT 永续合约优先
            if s.endswith("USDT"):
                s = s + "/USDT"
            elif s.endswith("BTC"):
                s = s + "/BTC"
            else:
                s = s + "/USDT"
        return s
