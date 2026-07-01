"""
Stochastic Oscillator Strategy.

%K 上穿 %D 且 %K < 20  → enter_long
%K 下穿 %D 且 %K > 80  → enter_short
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("stoch", "随机指标金叉死叉")
class StochasticStrategy(BaseStrategy):
    """随机指标 (Stochastic) 金叉死叉策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        k_window = self.params.get("k_window", 14)
        smooth_k = self.params.get("smooth_k", 3)
        smooth_d = self.params.get("smooth_d", 3)
        df["stoch_k"] = ta.momentum.stoch(
            df["high"], df["low"], df["close"], k_window, smooth_k
        )
        df["stoch_d"] = df["stoch_k"].rolling(smooth_d).mean()
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)
        # %K 上穿 %D 且 %K 在超卖区 → 做多
        df["enter_long"] = (
            (df["stoch_k"] > df["stoch_d"]) &
            (df["stoch_k"].shift(1) <= df["stoch_d"].shift(1)) &
            (df["stoch_k"] < oversold)
        )
        # %K 下穿 %D 且 %K 在超买区 → 做空
        df["enter_short"] = (
            (df["stoch_k"] < df["stoch_d"]) &
            (df["stoch_k"].shift(1) >= df["stoch_d"].shift(1)) &
            (df["stoch_k"] > overbought)
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df["exit_long"] = df["stoch_k"] > 80
        df["exit_short"] = df["stoch_k"] < 20
        return df
