"""routers.py 条件路由函数 单元测试"""

from __future__ import annotations

from qmind.agents.protocol import RiskFinalVerdict
from qmind.graph.routers import route_after_analysis, route_after_risk
from qmind.graph.state import AgentState


def _state(disagreement: float = 0.0, risk: RiskFinalVerdict | None = None) -> AgentState:
    return {
        "symbol": "T", "timeframe": "1h", "timestamp": 0,
        "market_data": None, "analyses": [], "debate": None,
        "disagreement": disagreement,
        "decision": None, "risk": risk,
        "execution_result": None, "evaluation": None,
        "errors": [], "debug_info": {},
    }


class TestRouteAfterAnalysis:
    def test_zero_disagreement_skips_debate(self):
        assert route_after_analysis(_state(disagreement=0.0)) == "skip_debate"

    def test_low_disagreement_skips_debate(self):
        assert route_after_analysis(_state(disagreement=0.14)) == "skip_debate"

    def test_boundary_triggers_debate(self):
        assert route_after_analysis(_state(disagreement=0.15)) == "start_debate"

    def test_high_disagreement_triggers_debate(self):
        assert route_after_analysis(_state(disagreement=0.5)) == "start_debate"
        assert route_after_analysis(_state(disagreement=0.99)) == "start_debate"


class TestRouteAfterRisk:
    def test_approved_executes(self):
        risk = RiskFinalVerdict(approved=True, veto_count=0)
        assert route_after_risk(_state(risk=risk)) == "execute"

    def test_rejected(self):
        risk = RiskFinalVerdict(approved=False, veto_count=2, vetoed_by=["conservative"])
        assert route_after_risk(_state(risk=risk)) == "rejected"

    def test_cvar_fail_rejected(self):
        risk = RiskFinalVerdict(approved=False, veto_count=0, vetoed_by=["cvar_constraint"])
        assert route_after_risk(_state(risk=risk)) == "rejected"

    def test_none_risk_executes(self):
        """当前代码: risk=None 时路由到 execute"""
        assert route_after_risk(_state()) == "execute"
