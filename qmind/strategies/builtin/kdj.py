"""
KDJ 随机指标策略。

KDJ 在经典随机指标（KD）基础上增加 J 线：
  J = 3*K - 2*D

入场信号基于超买/超卖区间内的金叉/死叉。
"""

from __future__ import annotations

import pandas as pd
import ta

from qmind.strategies.base import BaseStrategy
from qmind.strategies.registry import register_strategy


@register_strategy("kdj", "KDJ 随机指标")
class KDJStrategy(BaseStrategy):
    """KDJ 随机指标超买超卖策略"""
    timeframe = "1h"

    def populate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        k_period = self.params.get("k_period", 9)
        d_period = self.params.get("d_period", 3)
        # ta.momentum.stoch returns (K, D) as a DataFrame
        stoch = ta.momentum.stoch(
            df["high"], df["low"], df["close"],
            window=k_period, smooth_window=d_period
        )
        df["kdj_k"] = stoch
        df["kdj_d"] = stoch.rolling(d_period).mean()
        df["kdj_j"] = 3 * df["kdj_k"] - 2 * df["kdj_d"]
        return df

    def populate_entry_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        oversold = self.params.get("oversold", 20)
        overbought = self.params.get("overbought", 80)
        # 金叉：K < 20 且 K 上穿 D
        df["enter_long"] = (
            (df["kdj_k"] < oversold) &
            (df["kdj_k"] > df["kdj_d"]) &
            (df["kdj_k"].shift(1) <= df["kdj_d"].shift(1))
        )
        # 死叉：K > 80 且 K 下穿 D
        df["enter_short"] = (
            (df["kdj_k"] > overbought) &
            (df["kdj_k"] < df["kdj_d"]) &
            (df["kdj_k"].shift(1) >= df["kdj_d"].shift(1))
        )
        return df

    def populate_exit_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        # KDJ 回归中值区域时平仓
        df["exit_long"] = df["kdj_k"] > 80
        df["exit_short"] = df["kdj_k"] < 20
        return df
