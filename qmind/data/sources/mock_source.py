"""
模拟数据源 — 离线测试用，不依赖外部 API。
"""

from datetime import datetime, timedelta
import random

from qmind.graph.state import MarketData, OHLCV


class MockSource:
    """模拟数据源，生成假 K 线数据用于测试"""

    def __init__(self, seed: int = 42):
        self.rng = random.Random(seed)

    async def fetch_klines(self, symbol: str, interval: str = "1h", count: int = 200) -> MarketData:
        """生成 count 根模拟 K 线"""
        now = int(datetime.utcnow().timestamp() * 1000)
        interval_ms = {"1m": 60000, "5m": 300000, "15m": 900000,
                       "1h": 3600000, "4h": 14400000, "1d": 86400000}.get(interval, 3600000)

        price = 150.0 if "/" not in symbol else 50000.0
        klines = []
        for i in range(count):
            change = self.rng.gauss(0, 0.02) * price
            high = price + abs(self.rng.gauss(0, 0.015) * price)
            low = price - abs(self.rng.gauss(0, 0.015) * price)
            klines.append(OHLCV(
                timestamp=now - (count - i) * interval_ms,
                open=round(price, 2),
                high=round(max(high, price, low), 2),
                low=round(min(low, price, high), 2),
                close=round(price + change, 2),
                volume=round(self.rng.uniform(1000, 100000), 2),
                as_of=datetime.utcnow(),
            ))
            price = klines[-1].close

        return MarketData(
            symbol=symbol,
            klines={interval: klines},
            timestamp=now,
            as_of=datetime.utcnow(),
        )
