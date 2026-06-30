"""risk.py _calculate_cvar 纯函数 单元测试"""

from __future__ import annotations

import pytest

from qmind.agents.risk import TriangleRiskControl
from qmind.graph.state import TradeDecision


@pytest.fixture
def rc():
    return TriangleRiskControl.__new__(TriangleRiskControl)


class TestCVaRCalculation:
    def test_normal_long_passes(self, rc):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 95},
                          position_size_pct=10, confidence=0.8)
        cvar = rc._calculate_cvar(d, 10000)
        assert cvar.current_exposure == 50.0  # 10% * 10000 * 5/100
        assert cvar.cvar_threshold == 500.0   # 10000 * 0.05
        assert cvar.passed is True

    def test_exceeds_threshold_fails(self, rc):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 50},
                          position_size_pct=50, confidence=0.8)
        cvar = rc._calculate_cvar(d, 10000)
        assert cvar.passed is False

    def test_low_confidence_halves_threshold(self, rc):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 95},
                          position_size_pct=10, confidence=0.4)
        cvar = rc._calculate_cvar(d, 10000)
        assert cvar.cvar_threshold == 250.0  # 500 * 0.5

    def test_medium_confidence_eight_tenths(self, rc):
        d = TradeDecision(decision="LONG", entry={"price": 100}, stop_loss={"price": 95},
                          position_size_pct=10, confidence=0.6)
        cvar = rc._calculate_cvar(d, 10000)
        assert cvar.cvar_threshold == 400.0  # 500 * 0.8

    def test_zero_entry_auto_pass(self, rc):
        cvar = rc._calculate_cvar(TradeDecision(decision="LONG", entry={"price": 0}), 10000)
        assert cvar.passed is True

    def test_hold_auto_pass(self, rc):
        cvar = rc._calculate_cvar(TradeDecision(decision="HOLD"), 10000)
        assert cvar.passed is True
