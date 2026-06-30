"""
数据源适配器 — yfinance (美股)。

所有数据附带 as_of 时间戳防止 Point-in-Time 偏差。
"""

from __future__ import annotations

from datetime import UTC, datetime

from qmind.data.symbol_map import to_yfinance
from qmind.graph.state import OHLCV, MarketData


class YFinanceSource:
    """yfinance 数据源适配器"""

    def __init__(self):
        self._yf = None

    @property
    def yf(self):
        if self._yf is None:
            import yfinance as yf
            self._yf = yf
        return self._yf

    async def fetch_klines(
        self,
        symbol: str,
        interval: str = "1h",
        _start: str | None = None,
        _end: str | None = None,
    ) -> MarketData:
        """获取 K 线数据"""
        yf_symbol = to_yfinance(symbol)
        interval_map = {
            "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "60m", "4h": "60m", "1d": "1d", "1w": "1wk",
        }
        yf_interval = interval_map.get(interval, "60m")
        period = "1mo"
        if interval in ("1d", "1w"):
            period = "1y"
        if interval == "1m":
            period = "7d"

        ticker = self.yf.Ticker(yf_symbol)
        hist = ticker.history(period=period, interval=yf_interval)

        now = datetime.now(UTC)
        klines: list[OHLCV] = []
        for idx, row in hist.iterrows():
            ts = int(idx.timestamp() * 1000)
            if ts > int(now.timestamp() * 1000):
                continue  # 过滤未来数据
            klines.append(OHLCV(
                timestamp=ts,
                open=float(row["Open"]),
                high=float(row["High"]),
                low=float(row["Low"]),
                close=float(row["Close"]),
                volume=float(row["Volume"]),
                as_of=now,
            ))

        return MarketData(
            symbol=yf_symbol,
            klines={interval: klines},
            timestamp=int(now.timestamp() * 1000),
            as_of=now,
        )
