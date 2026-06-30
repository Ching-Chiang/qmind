"""
QMind 五阶段 LangGraph 主流水线。

阶段:
1. 数据采集 → 2. 多维分析 (4 分析师并行) → 3. 辩论 (分歧驱动) → 4. 决策 → 5. 风控

基于论文审阅修正:
- 辩论不做方向判断，只做风险降级
- δ < 0.15 跳过辩论
- 三角风控一票否决制
- CVaR 硬约束校验
"""

from __future__ import annotations

import logging
from typing import Any

from langgraph.graph import END, StateGraph

from qmind.agents.analysts.runner import AnalystRunner
from qmind.agents.researchers.disagreement import compute_disagreement
from qmind.agents.researchers.leader import DebateLeader
from qmind.agents.researchers.skeptic import SkepticAgent
from qmind.agents.researchers.trust import TrustAgent
from qmind.agents.risk import TriangleRiskControl
from qmind.agents.single_agent import SingleTradingAgent
from qmind.graph.routers import (
    route_after_analysis,
    route_after_risk,
)
from qmind.graph.state import AgentState, MarketData
from qmind.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QMindPipeline:
    """QMind 五阶段交易流水线"""

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.analyst_runner = AnalystRunner(llm_client)
        self.trust_agent = TrustAgent(llm_client)
        self.skeptic_agent = SkepticAgent(llm_client)
        self.debate_leader = DebateLeader(llm_client)
        self.single_agent = SingleTradingAgent(llm_client)
        self.risk_control = TriangleRiskControl(llm_client)

        self.graph = self._build_graph()

    # ── 阶段 1: 数据采集 ──

    async def collect_data(self, state: AgentState) -> dict[str, Any]:
        """采集市场数据"""
        symbol = state.get("symbol", "UNKNOWN")
        timeframe = state.get("timeframe", "1h")

        # 尝试从数据源获取
        from qmind.data.sources.factory import DataSourceFactory
        factory = DataSourceFactory()
        try:
            market_data = await factory.fetch_market_data(symbol, interval=timeframe)
        except Exception as e:
            logger.warning(f"数据采集失败: {e}，使用空数据")
            market_data = MarketData(symbol=symbol)

        return {"market_data": market_data}

    # ── 阶段 2: 多维分析 ──

    async def analyze(self, state: AgentState) -> dict[str, Any]:
        """并行运行 4 个分析师"""
        market_data = state.get("market_data")
        if market_data is None:
            return {"analyses": [], "disagreement": 0.0, "errors": ["无市场数据"]}

        reports = await self.analyst_runner.run_all(market_data)
        disagreement = compute_disagreement(reports)

        return {
            "analyses": reports,
            "disagreement": disagreement["delta"],
            "debug_info": {"disagreement_details": disagreement},
        }

    # ── 阶段 3: 辩论（分歧驱动） ──

    async def debate(self, state: AgentState) -> dict[str, Any]:
        """辩论阶段 — 分歧高时启动"""
        market_data = state.get("market_data")
        analyses = state.get("analyses", [])
        disagreement_info = state.get("debug_info", {}).get("disagreement_details", {})

        if not market_data or not analyses:
            return {"errors": ["缺少分析数据，无法辩论"]}

        # Trust Agent 验证论据质量
        trust = await self.trust_agent.verify(market_data, analyses)
        # Skeptic Agent 找漏洞
        skeptic = await self.skeptic_agent.scrutinize(market_data, analyses, trust)
        # Debate Leader 输出降级因子
        debate_result = await self.debate_leader.lead(
            market_data, analyses, disagreement_info, trust, skeptic,
        )

        return {
            "debate": {
                "rounds": debate_result.rounds_completed,
                "converged": debate_result.converged,
                "final_stance": None,  # 辩论不做方向判断
                "bull_core_argument": "",
                "bear_core_counter": "",
                "agreement_points": debate_result.agreement_points,
                "disagreement_points": debate_result.disagreement_points,
                "consensus_confidence": debate_result.confidence_downgrade,
            },
        }

    # ── 阶段 4: 交易决策 ──

    async def decide(self, state: AgentState) -> dict[str, Any]:
        """交易决策 — 使用单 Agent 或辩论降级结果"""
        market_data = state.get("market_data")
        if market_data is None:
            return {"errors": ["无市场数据，无法决策"]}

        # 获取辩论降级因子（如果有）
        debate_result = state.get("debate")
        confidence_downgrade = 1.0
        if debate_result and debate_result.get("consensus_confidence"):
            confidence_downgrade = debate_result["consensus_confidence"]

        # 使用单 Agent 做决策
        decision = await self.single_agent.analyze(market_data)

        # 应用辩论降级
        decision.confidence *= confidence_downgrade
        if confidence_downgrade < 0.7:
            decision.position_size_pct *= confidence_downgrade

        return {"decision": decision}

    # ── 阶段 5: 风控审核 ──

    async def review_risk(self, state: AgentState) -> dict[str, Any]:
        """三角风控审核 + CVaR 硬约束"""
        market_data = state.get("market_data")
        decision = state.get("decision")

        if decision is None:
            return {"errors": ["无决策，无法风控"]}

        if market_data:
            risk_result = await self.risk_control.review(decision, market_data)
        else:
            risk_result = await self.risk_control.review(decision, MarketData(symbol=decision.symbol))

        return {"risk": risk_result}

    # ── 执行/拒绝终端节点 ──

    async def execute(self, state: AgentState) -> dict[str, Any]:
        """执行交易（或 dryRun 记录）"""
        decision = state.get("decision")
        if decision is None:
            return {"execution_result": {"status": "no_decision"}}

        execution = {
            "status": "dry_run" if True else "executed",  # 当前 default dryRun
            "symbol": decision.symbol,
            "decision": decision.decision,
            "entry": decision.entry,
            "position_size_pct": decision.position_size_pct,
            "confidence": decision.confidence,
        }
        return {"execution_result": execution}

    async def reject(self, state: AgentState) -> dict[str, Any]:
        """交易被取消 — 记录原因"""
        risk = state.get("risk")
        return {
            "execution_result": {
                "status": "rejected",
                "reason": (
                    f"风控否决: {', '.join(risk.vetoed_by) if risk and risk.vetoed_by else 'CVaR约束'}"
                ),
            },
        }

    # ── 构建 LangGraph ──

    def _build_graph(self) -> StateGraph:
        """构建五阶段 StateGraph"""
        workflow = StateGraph(AgentState)

        # 注册节点
        workflow.add_node("collect_data", self.collect_data)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("debate", self.debate)
        workflow.add_node("decide", self.decide)
        workflow.add_node("review_risk", self.review_risk)
        workflow.add_node("execute", self.execute)
        workflow.add_node("rejected", self.reject)

        # 设置入口
        workflow.set_entry_point("collect_data")

        # 主线: 数据 → 分析
        workflow.add_edge("collect_data", "analyze")

        # 条件: 分析 → 跳辩论或直接决策
        workflow.add_conditional_edges(
            "analyze",
            route_after_analysis,
            {
                "skip_debate": "decide",
                "start_debate": "debate",
            },
        )

        # 辩论 → 决策
        workflow.add_edge("debate", "decide")

        # 决策 → 风控
        workflow.add_edge("decide", "review_risk")

        # 条件: 风控 → 执行或取消
        workflow.add_conditional_edges(
            "review_risk",
            route_after_risk,
            {
                "execute": "execute",
                "rejected": "rejected",
            },
        )

        # 终端
        workflow.add_edge("execute", END)
        workflow.add_edge("rejected", END)

        return workflow.compile()

    async def run(self, symbol: str, timeframe: str = "1h") -> AgentState:
        """运行一次完整流水线"""
        initial: AgentState = {
            "symbol": symbol,
            "timeframe": timeframe,
            "timestamp": 0,
            "market_data": None,
            "analyses": [],
            "debate": None,
            "disagreement": 0.0,
            "decision": None,
            "risk": None,
            "execution_result": None,
            "evaluation": None,
            "errors": [],
            "debug_info": {},
        }
        return await self.graph.ainvoke(initial)
