"""
Money Flow Index (MFI) 超买超卖策略。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("mfi", "MFI 超买超卖")
class MFIStrategy(BaseStrategy):
    """Money Flow Index 超买超卖均值回归策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        period = self.params.get("period", 14)
        df["mfi"] = ta.volume.money_flow_index(
            df["high"], df["low"], df["close"], df["volume"], period
        )
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)
        df["enter_long"] = (df["mfi"] < oversold) & (df["mfi"].shift(1) >= oversold)
        df["enter_short"] = (df["mfi"] > overbought) & (df["mfi"].shift(1) <= overbought)
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["mfi"] > 50
        df["exit_short"] = df["mfi"] < 50
        return df
