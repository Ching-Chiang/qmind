"""
时间完整性校验器 (Time Guard)。

禁止任何 timestamp >= 决策时刻的数据进入 prompt，
防止 Point-in-Time (PIT) 偏差。详见 P0 修正 #2。
"""

from __future__ import annotations

from datetime import datetime
from typing import Any


class TimeGuardError(Exception):
    """数据时间戳违反时间完整性"""
    pass


class TimeGuard:
    """时间完整性校验器"""

    def __init__(self, decision_time: datetime | None = None, tolerance_sec: int = 60):
        self.decision_time = decision_time or datetime.utcnow()
        self.tolerance_sec = tolerance_sec

    def check_timestamp(self, data_timestamp: int, label: str = "data") -> None:
        """检查 Unix ms 时间戳是否 <= 决策时间 + 容差"""
        data_dt = datetime.utcfromtimestamp(data_timestamp / 1000)
        deadline = self.decision_time.timestamp() + self.tolerance_sec
        if data_dt.timestamp() > deadline:
            offset_sec = data_dt.timestamp() - self.decision_time.timestamp()
            raise TimeGuardError(
                f"{label} 时间戳 {data_dt} 晚于决策时间 {self.decision_time} "
                f"(偏差 {offset_sec:.0f}s, 容差 {self.tolerance_sec}s) — 涉嫌 Look-Ahead 偏差"
            )

    def check_market_data(self, market_data: Any) -> None:
        """检查 MarketData 对象的时间完整性"""
        if hasattr(market_data, "timestamp") and market_data.timestamp:
            self.check_timestamp(market_data.timestamp, "market_data")
        if (
            hasattr(market_data, "as_of")
            and market_data.as_of
            and isinstance(market_data.as_of, datetime)
            and market_data.as_of > self.decision_time
        ):
                raise TimeGuardError(
                    f"market_data.as_of {market_data.as_of} 晚于决策时间 {self.decision_time}"
                )

    def check_klines(self, klines: list[Any], label: str = "klines") -> None:
        """检查 K 线序列中是否包含未来数据"""
        for k in klines:
            if hasattr(k, "timestamp") and k.timestamp:
                self.check_timestamp(k.timestamp, label)

    def validate_as_of(self, data: Any, path: str = "") -> None:
        """递归检查嵌套数据结构中的 as_of 字段"""
        if isinstance(data, datetime):
            if data > self.decision_time:
                raise TimeGuardError(
                    f"{path}.as_of={data} 晚于决策时间 {self.decision_time}"
                )
        elif isinstance(data, dict):
            for k, v in data.items():
                self.validate_as_of(v, f"{path}.{k}")
        elif isinstance(data, list):
            for i, v in enumerate(data):
                self.validate_as_of(v, f"{path}[{i}]")
