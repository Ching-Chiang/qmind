"""
数据源适配器 — Tushare (A 股/港股/宏观)。

所有数据附带 as_of 时间戳防止 Point-in-Time 偏差。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from qmind.graph.state import OHLCV, MarketData


class TushareSource:
    """Tushare 数据源适配器"""

    def __init__(self, token: str = ""):
        import tushare as ts
        if token:
            ts.set_token(token)
        self._ts = ts
        self._pro: Any | None = None

    @property
    def pro(self) -> Any:
        if self._pro is None:
            self._pro = self._ts.pro_api()
        return self._pro

    async def fetch_daily(
        self,
        symbol: str,
        start_date: str = "",
        end_date: str = "",
    ) -> MarketData:
        """获取 A 股日线数据"""
        df = self.pro.daily(ts_code=symbol, start_date=start_date, end_date=end_date)

        now = datetime.utcnow()
        klines: list[OHLCV] = []
        for _, row in df.iterrows():
            dt = datetime.strptime(str(row["trade_date"]), "%Y%m%d")
            ts_ms = int(dt.timestamp() * 1000)
            klines.append(OHLCV(
                timestamp=ts_ms,
                open=float(row["open"]),
                high=float(row["high"]),
                low=float(row["low"]),
                close=float(row["close"]),
                volume=float(row["vol"]),
                as_of=now,
            ))

        return MarketData(
            symbol=symbol,
            klines={"1d": klines},
            timestamp=int(now.timestamp() * 1000),
            as_of=now,
        )
