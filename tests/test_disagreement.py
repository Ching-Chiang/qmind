"""disagreement.py 分歧检测器 单元测试"""

from __future__ import annotations

from qmind.agents.researchers.disagreement import compute_disagreement
from qmind.graph.state import AnalystReport


def _r(analyst: str, stance: str, conf: float, core: str = "") -> AnalystReport:
    return AnalystReport(analyst=analyst, stance=stance, confidence=conf, core_reason=core)


class TestComputeDisagreement:
    def test_empty_list(self):
        r = compute_disagreement([])
        assert r["delta"] == 0.0 and r["needs_debate"] is False

    def test_single_analyst(self):
        r = compute_disagreement([_r("a", "bullish", 0.85)])
        assert r["delta"] == 0.0 and r["needs_debate"] is False
        assert r["strongest_analyst"] == "a"

    def test_all_same_stance(self):
        r = compute_disagreement([
            _r("a", "bullish", 0.7), _r("b", "bullish", 0.8), _r("c", "bullish", 0.6),
        ])
        assert r["delta"] < 0.15 and r["needs_debate"] is False

    def test_polar_opposites_high_disagreement(self):
        r = compute_disagreement([_r("a", "bullish", 0.9), _r("b", "bearish", 0.9)])
        assert r["delta"] >= 0.15 and r["needs_debate"] is True
        assert r["level"] == "high"

    def test_mixed_stances(self):
        r = compute_disagreement([
            _r("a", "bullish", 0.8), _r("b", "bearish", 0.7),
            _r("c", "neutral", 0.5), _r("d", "bullish", 0.6),
        ])
        assert r["needs_debate"] is True
        assert r["strongest_analyst"] == "a"

    def test_zero_confidence_fallback(self):
        r = compute_disagreement([_r("a", "bullish", 0.0), _r("b", "bearish", 0.0)])
        assert r["delta"] >= 0

    def test_identifies_strongest(self):
        r = compute_disagreement([
            _r("a", "bullish", 0.3), _r("b", "neutral", 0.9), _r("c", "bearish", 0.6),
        ])
        assert r["strongest_analyst"] == "b"
        assert r["strongest_stance"] == "neutral"
