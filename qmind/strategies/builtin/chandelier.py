"""
Chandelier Exit 策略。
基于 ATR 的 trailing stop 参考线：价格突破其上方则做多，跌破其下方则做空。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("chandelier", "Chandelier Exit ATR 跟踪")
class ChandelierStrategy(BaseStrategy):
    """Chandelier Exit 趋势跟踪策略"""
    timeframe = "1d"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        atr_period = self.params.get("atr_period", 22)
        multiplier = self.params.get("multiplier", 3.0)

        df["atr"] = ta.volatility.average_true_range(
            df["high"], df["low"], df["close"], atr_period
        )
        # Long exit: 22-day high - ATR * 3
        df["chandelier_long"] = df["high"].rolling(atr_period).max() - df["atr"] * multiplier
        # Short exit: 22-day low + ATR * 3
        df["chandelier_short"] = df["low"].rolling(atr_period).min() + df["atr"] * multiplier
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # enter_long: close crosses above chandelier_long
        df["enter_long"] = (
            (df["close"] > df["chandelier_long"]) &
            (df["close"].shift(1) <= df["chandelier_long"].shift(1))
        )
        # enter_short: close crosses below chandelier_short
        df["enter_short"] = (
            (df["close"] < df["chandelier_short"]) &
            (df["close"].shift(1) >= df["chandelier_short"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # exit_long when close drops back below chandelier_long
        df["exit_long"] = df["close"] < df["chandelier_long"]
        # exit_short when close rises back above chandelier_short
        df["exit_short"] = df["close"] > df["chandelier_short"]
        return df
