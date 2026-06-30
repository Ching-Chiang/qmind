"""
策略基类 — Freqtrade 三层抽象。

populate_indicators(df) -> df
populate_entry_signal(df) -> df
populate_exit_signal(df) -> df
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import pandas as pd


class BaseStrategy(ABC):
    """策略基类 — 三层信号生成"""

    name: str = "base"
    description: str = ""

    # 可配置参数
    timeframe: str = "1h"
    params: dict[str, Any] = {}

    @abstractmethod
    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """添加技术指标列"""
        ...

    @abstractmethod
    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成入场信号列 (enter_long, enter_short)"""
        ...

    @abstractmethod
    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成出场信号列 (exit_long, exit_short)"""
        ...

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)
