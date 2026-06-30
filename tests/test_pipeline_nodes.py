"""pipeline.py 各节点 单元测试 (mock LLM + DataSource)"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest

from qmind.agents.protocol import DebateResultProtocol, RiskFinalVerdict
from qmind.graph.pipeline import QMindPipeline
from qmind.graph.state import OHLCV, AgentState, MarketData, TradeDecision
from qmind.llm.client import LLMClient


@pytest.fixture
def pipeline():
    return QMindPipeline(LLMClient())


@pytest.fixture
def base_state() -> AgentState:
    return {
        "symbol": "TEST/USDT", "timeframe": "1h", "timestamp": 0,
        "market_data": None, "analyses": [], "debate": None, "disagreement": 0.0,
        "decision": None, "risk": None, "execution_result": None, "evaluation": None,
        "errors": [], "debug_info": {},
    }


class TestCollectDataNode:
    async def test_returns_empty_on_failure(self, pipeline, base_state):
        """DataSourceFactory 失败时返回空 MarketData"""
        with patch("qmind.data.sources.factory.DataSourceFactory.fetch_market_data",
                   side_effect=Exception("network error")):
            result = await pipeline.collect_data(base_state)
        md = result.get("market_data")
        assert md is not None
        assert md.symbol == "TEST/USDT"
        assert md.klines == {}

    async def test_syncs_price_to_exchange(self, pipeline, base_state):
        """采集数据后应同步价格到 exchange"""
        from qmind.execution.dry_run import DryRunExchange
        ex = DryRunExchange(10000)
        pipeline.exchange = ex

        kline = OHLCV(timestamp=1000, open=100, high=101, low=99, close=100.5, volume=1000)
        md = MarketData(symbol="TEST/USDT", klines={"1h": [kline]})
        with patch("qmind.data.sources.factory.DataSourceFactory.fetch_market_data",
                   return_value=md):
            await pipeline.collect_data(base_state)
        assert await ex.get_price("TEST/USDT") == 100.5


class TestAnalyzeNode:
    async def test_no_market_data_returns_error(self, pipeline, base_state):
        base_state["market_data"] = None
        result = await pipeline.analyze(base_state)
        assert "errors" in result
        assert result["analyses"] == []

    async def test_runs_analysts_and_computes_disagreement(self, pipeline, base_state):
        md = MarketData(symbol="TEST/USDT")
        base_state["market_data"] = md
        with patch.object(pipeline.analyst_runner, "run_all", return_value=[]):
            result = await pipeline.analyze(base_state)
        assert "analyses" in result
        assert result["disagreement"] == 0.0


class TestDebateNode:
    async def test_skip_on_no_data(self, pipeline, base_state):
        result = await pipeline.debate(base_state)
        assert "errors" in result

    async def test_runs_full_debate_chain(self, pipeline, base_state):
        md = MarketData(symbol="TEST/USDT")
        base_state["market_data"] = md
        from qmind.graph.state import AnalystReport
        base_state["analyses"] = [AnalystReport(analyst="test", stance="bullish", confidence=0.5, core_reason="x")]
        base_state["debug_info"] = {"disagreement_details": {"delta": 0.3}}

        # Mock all three debate sub-agents
        pipeline.trust_agent.verify = AsyncMock(return_value={"assessment": "ok", "concerns": []})
        pipeline.skeptic_agent.scrutinize = AsyncMock(return_value={"gaps": [], "worst_case": "none"})
        pipeline.debate_leader.lead = AsyncMock(return_value=DebateResultProtocol(
            rounds_completed=1, converged=True, confidence_downgrade=0.8, position_size_reduction=0.2,
        ))

        result = await pipeline.debate(base_state)
        debate = result.get("debate")
        assert debate is not None
        assert debate["consensus_confidence"] == 0.8


class TestDecideNode:
    async def test_no_market_data_returns_error(self, pipeline, base_state):
        result = await pipeline.decide(base_state)
        assert "errors" in result

    async def test_applies_confidence_downgrade(self, pipeline, base_state):
        md = MarketData(symbol="TEST/USDT", klines={"1h": [
            OHLCV(timestamp=1000, open=100, high=101, low=99, close=100, volume=1000) for _ in range(5)
        ]})
        base_state["market_data"] = md
        base_state["debate"] = {"consensus_confidence": 0.5}

        with patch.object(pipeline.single_agent, "analyze",
                          return_value=TradeDecision(decision="LONG", confidence=0.8,
                                                     position_size_pct=10, risk_reward_ratio=2.0)):
            result = await pipeline.decide(base_state)
        decision = result["decision"]
        assert decision.confidence == 0.4  # 0.8 * 0.5
        assert decision.position_size_pct == 5.0  # 10 * 0.5 (since 0.5 < 0.7)

    async def test_no_debate_uses_full_confidence(self, pipeline, base_state):
        md = MarketData(symbol="TEST/USDT", klines={"1h": [
            OHLCV(timestamp=1000, open=100, high=101, low=99, close=100, volume=1000) for _ in range(5)
        ]})
        base_state["market_data"] = md
        base_state["debate"] = None
        with patch.object(pipeline.single_agent, "analyze",
                          return_value=TradeDecision(decision="HOLD", confidence=0.5)):
            result = await pipeline.decide(base_state)
        assert result["decision"].confidence == 0.5


class TestExecuteNode:
    async def test_dry_run_execution(self, pipeline, base_state):
        base_state["decision"] = TradeDecision(decision="LONG", symbol="TEST/USDT")
        result = await pipeline.execute(base_state)
        assert result["execution_result"]["status"] == "dry_run"

    async def test_no_decision(self, pipeline, base_state):
        result = await pipeline.execute(base_state)
        assert result["execution_result"]["status"] == "no_decision"

    async def test_live_execution_calls_exchange(self, pipeline, base_state):
        from qmind.execution.dry_run import DryRunExchange
        ex = DryRunExchange(10000)
        ex.dry_run = False
        pipeline.exchange = ex
        ex.update_price("TEST/USDT", 100)

        base_state["decision"] = TradeDecision(
            decision="LONG", symbol="TEST/USDT",
            entry={"type": "market", "price": 100, "quantity": 1},
        )
        with patch.object(ex, "place_order", return_value=AsyncMock()) as mock_place:
            mock_place.return_value.order_id = "mock_001"
            result = await pipeline.execute(base_state)
        assert result["execution_result"]["status"] == "live"


class TestRejectNode:
    async def test_rejected_trade(self, pipeline, base_state):
        base_state["risk"] = RiskFinalVerdict(approved=False, veto_count=1, vetoed_by=["conservative"])
        result = await pipeline.reject(base_state)
        assert result["execution_result"]["status"] == "rejected"
        assert "conservative" in result["execution_result"]["reason"]
