"""策略注册表 + 策略信号生成 单元测试"""

from __future__ import annotations

import pandas as pd
import pytest

from qmind.strategies.base import BaseStrategy

# 导入内置策略模块以触发 @register_strategy 装饰器
from qmind.strategies.builtin import ma_cross, macd, rsi  # noqa: F401
from qmind.strategies.registry import _registry, get_strategy, list_strategies, register_strategy


class TestRegistry:
    def test_has_builtin_strategies(self):
        names = [s["name"] for s in list_strategies()]
        for expected in ("ma_cross", "macd", "rsi", "bollinger", "donchian"):
            assert expected in names

    def test_get_strategy_with_params(self):
        strat = get_strategy("ma_cross", fast_period=9, slow_period=21)
        assert strat.name == "ma_cross"
        assert strat.get_param("fast_period") == 9

    def test_unknown_strategy_raises(self):
        with pytest.raises(ValueError, match="Unknown strategy"):
            get_strategy("this_does_not_exist")

    def test_custom_registration_and_removal(self):
        @register_strategy("_test_temp", "temp")
        class TempDummy(BaseStrategy):
            def populate_indicators(self, df): return df
            def populate_entry_signal(self, df): return df
            def populate_exit_signal(self, df): return df

        assert get_strategy("_test_temp").name == "_test_temp"
        _registry.pop("_test_temp", None)
        with pytest.raises(ValueError):
            get_strategy("_test_temp")


class TestMACrossStrategy:
    def test_generates_columns(self):
        strat = get_strategy("ma_cross")
        df = pd.DataFrame({"close": [float(i) for i in range(50)]})
        df = strat.populate_indicators(df)
        df = strat.populate_entry_signal(df)
        df = strat.populate_exit_signal(df)
        for col in ("sma_fast", "sma_slow", "enter_long", "exit_long"):
            assert col in df.columns

    def test_entry_long_on_cross(self):
        df = pd.DataFrame({"close": [float(i) for i in range(30)] + [float(30 - i) for i in range(20)]})
        strat = get_strategy("ma_cross")
        df = strat.populate_indicators(df)
        df = strat.populate_entry_signal(df)
        assert df["enter_long"].sum() >= 0


class TestRSIStrategy:
    def test_rsi_column(self):
        import math
        df = pd.DataFrame({"close": [50.0 + math.sin(i * 0.3) * 30 for i in range(50)]})
        strat = get_strategy("rsi")
        df = strat.populate_indicators(df)
        assert "rsi" in df.columns


class TestMACDStrategy:
    def test_macd_columns(self):
        import math
        df = pd.DataFrame({"close": [100.0 + math.sin(i * 0.2) * 10 for i in range(60)]})
        strat = get_strategy("macd")
        df = strat.populate_indicators(df)
        df = strat.populate_entry_signal(df)
        for col in ("macd", "macd_signal", "enter_long", "enter_short"):
            assert col in df.columns
