"""
数据源适配器 — Binance 公开 REST API (加密货币)。

直接 httpx 请求 Binance 公开接口，不走 ccxt（避免复杂市场加载）。
支持 HTTP/SOCKS5 代理（通过 httpx 原生 proxy 支持）。
"""

from __future__ import annotations

from datetime import datetime

import httpx

from qmind.graph.state import MarketData, OHLCV

_INTERVAL_MAP: dict[str, str] = {
    "1m": "1m", "5m": "5m", "15m": "15m",
    "1h": "1h", "4h": "4h",
    "1d": "1d",
}


class BinanceSource:
    """Binance 公开 REST API 适配器"""

    API_BASE = "https://api.binance.com"

    def __init__(self, proxy: str = "") -> None:
        self._proxy = proxy or self._detect_proxy()
        self._client: httpx.AsyncClient | None = None

    @staticmethod
    def _detect_proxy() -> str:
        """从环境变量或 Windows 系统代理自动检测"""
        import os
        for var in ("all_proxy", "ALL_PROXY", "https_proxy", "HTTPS_PROXY"):
            val = os.environ.get(var, "")
            if val:
                return val
        try:
            import winreg
            with winreg.OpenKey(
                winreg.HKEY_CURRENT_USER,
                r"Software\Microsoft\Windows\CurrentVersion\Internet Settings",
            ) as key:
                enabled, _ = winreg.QueryValueEx(key, "ProxyEnable")
                if enabled:
                    server, _ = winreg.QueryValueEx(key, "ProxyServer")
                    return f"http://{server}"
        except Exception:
            pass
        return ""

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            kwargs: dict = {}
            if self._proxy:
                kwargs["proxy"] = self._proxy
            self._client = httpx.AsyncClient(**kwargs)
        return self._client

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        limit: int = 200,
    ) -> MarketData:
        """获取 Binance K 线数据"""
        pair = self._normalize_symbol(symbol).replace("/", "")
        tf = _INTERVAL_MAP.get(interval, "1h")
        url = f"{self.API_BASE}/api/v3/klines?symbol={pair}&interval={tf}&limit={limit}"

        resp = await self.client.get(url)
        resp.raise_for_status()
        raw: list = resp.json()

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
            symbol=self._normalize_symbol(symbol),
            klines={interval: klines},
            timestamp=int(now.timestamp() * 1000),
            as_of=now,
        )

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    @staticmethod
    def _normalize_symbol(symbol: str) -> str:
        s = symbol.upper().strip()
        if "/" not in s:
            if s.endswith("USDT"):
                s += "/USDT"
            elif s.endswith("BTC"):
                s += "/BTC"
            else:
                s += "/USDT"
        return s
