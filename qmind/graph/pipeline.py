"""
QMind 六阶段 LangGraph 主流水线。

阶段:
1. 数据采集 → 2. 多维分析 (4 分析师并行) → 3. 辩论 (分歧驱动) → 4. 决策 → 5. 风控
6. CVRF 学习（交易后自动评估、反思、存入记忆库）

基于论文审阅修正:
- 辩论不做方向判断，只做风险降级
- δ < 0.15 跳过辩论
- 三角风控一票否决制
- CVaR 硬约束校验
- TimeGuard 时间完整性检查 (P0 修正 #2)
- CVRF 闭环学习（Phase 4）
"""

from __future__ import annotations

import logging
import time as time_module
from datetime import datetime
from typing import Any

from langgraph.graph import END, StateGraph

from qmind.agents.analysts.runner import AnalystRunner
from qmind.agents.researchers.disagreement import compute_disagreement
from qmind.agents.researchers.leader import DebateLeader
from qmind.agents.researchers.skeptic import SkepticAgent
from qmind.agents.researchers.trust import TrustAgent
from qmind.agents.risk import TriangleRiskControl
from qmind.agents.single_agent import SingleTradingAgent
from qmind.data.time_guard import TimeGuard, TimeGuardError
from qmind.graph.routers import (
    route_after_analysis,
    route_after_risk,
)
from qmind.graph.state import (
    AgentState,
    Lesson,
    MarketData,
    MemoryEntry,
    TradeEvaluation,
)
from qmind.learning.cvrf import CVRFReflection
from qmind.learning.cvrf_pipeline import CVRFPipeline
from qmind.learning.evaluator import TradeRecord
from qmind.learning.memory import MemoryStore
from qmind.llm.client import LLMClient

logger = logging.getLogger(__name__)


class QMindPipeline:
    """QMind 六阶段交易流水线 + CVRF 学习闭环"""

    def __init__(
        self,
        llm_client: LLMClient,
        exchange=None,
        cvrf_pipeline: CVRFPipeline | None = None,
        memory_store: MemoryStore | None = None,
        cvrf_reflection: CVRFReflection | None = None,
    ):
        self.llm_client = llm_client
        self.exchange = exchange
        self.analyst_runner = AnalystRunner(llm_client)
        self.trust_agent = TrustAgent(llm_client)
        self.skeptic_agent = SkepticAgent(llm_client)
        self.debate_leader = DebateLeader(llm_client)
        self.single_agent = SingleTradingAgent(llm_client)
        self.risk_control = TriangleRiskControl(llm_client)

        # CVRF 学习组件（支持外部注入 + 懒初始化）
        self.cvrf_pipeline = cvrf_pipeline
        self.memory_store = memory_store
        self.cvrf_reflection = cvrf_reflection

        self.graph = self._build_graph()

    # ── CVRF 懒初始化 ──

    def _ensure_cvrf(self) -> None:
        """确保 CVRF 组件已初始化（懒加载）"""
        if self.cvrf_pipeline is not None:
            return
        if self.cvrf_reflection is None:
            self.cvrf_reflection = CVRFReflection(self.llm_client)
        if self.memory_store is None:
            self.memory_store = MemoryStore()
        self.cvrf_pipeline = CVRFPipeline(self.cvrf_reflection, self.memory_store)

    # ── 辅助方法 ──

    @staticmethod
    def _get_current_price(market_data: MarketData | None) -> float | None:
        """从市场数据中获取最新价格"""
        if market_data is None:
            return None
        for tf in ("1h", "5m", "1m", "1d"):
            klines = market_data.klines.get(tf)
            if klines:
                return klines[-1].close
        return None

    @staticmethod
    def _summarize_analyses(analyses: list) -> str:
        """将分析师报告汇总为文本摘要"""
        if not analyses:
            return ""
        lines = []
        for a in analyses:
            lines.append(
                f"[{a.analyst}] {a.stance} (conf={a.confidence:.2f}): {a.core_reason}"
            )
        return "\n".join(lines)

    @staticmethod
    def _format_market_context(market_data: MarketData | None) -> str:
        """格式化市场数据为文本上下文"""
        if market_data is None:
            return ""
        lines = [f"标的: {market_data.symbol}"]
        for tf, klines in market_data.klines.items():
            if klines:
                last = klines[-1]
                lines.append(
                    f"[{tf}] O:{last.open:.2f} H:{last.high:.2f} "
                    f"L:{last.low:.2f} C:{last.close:.2f} V:{last.volume:.2f}"
                )
        return "\n".join(lines)

    @staticmethod
    def _build_rejection_lessons(decision, risk) -> list[Lesson]:
        """从风控否决结果构建教训条目"""
        lessons = []
        if risk and risk.vetoed_by:
            for vetoer in risk.vetoed_by:
                lessons.append(Lesson(
                    lesson=(
                        f"{vetoer} 否决了交易 {decision.symbol} {decision.decision}"
                    ),
                    confidence=0.9,
                    source="risk_control",
                ))
        if risk and risk.cvar_check and not risk.cvar_check.get("passed", True):
            lessons.append(Lesson(
                lesson=(
                    f"CVaR 硬约束未通过: "
                    f"exposure={risk.cvar_check.get('current_cvar_exposure')} > "
                    f"threshold={risk.cvar_check.get('threshold')}"
                ),
                confidence=0.85,
                source="cvar",
            ))
        if not lessons:
            lessons.append(Lesson(
                lesson="交易被拒绝（无详细风控记录）",
                confidence=0.5,
                source="unknown",
            ))
        return lessons

    # ── 阶段 1: 数据采集 ──

    async def collect_data(self, state: AgentState) -> dict[str, Any]:
        """采集市场数据 + TimeGuard 时间完整性检查 + 同步价格到交易所"""
        symbol = state.get("symbol", "UNKNOWN")
        timeframe = state.get("timeframe", "1h")

        from qmind.data.sources.factory import DataSourceFactory
        factory = DataSourceFactory()
        try:
            market_data = await factory.fetch_market_data(symbol, interval=timeframe)
        except Exception as e:
            logger.warning(f"数据采集失败: {e}，使用空数据")
            market_data = MarketData(symbol=symbol)

        # ── TimeGuard 时间完整性检查 ──
        time_guard = TimeGuard()
        try:
            time_guard.check_market_data(market_data)
            for kline_key, klines in market_data.klines.items():
                time_guard.check_klines(klines, label=f"klines.{kline_key}")
        except TimeGuardError as e:
            logger.warning(f"数据采集阶段时间完整性违规，丢弃问题 K 线: {e}")
            market_data.klines = {}

        # 同步价格到交易所
        if self.exchange and hasattr(self.exchange, "update_price"):
            for klines in market_data.klines.values():
                if klines:
                    self.exchange.update_price(market_data.symbol, klines[-1].close)
                    break

        return {"market_data": market_data}

    # ── 阶段 2: 多维分析 ──

    async def analyze(self, state: AgentState) -> dict[str, Any]:
        """并行运行 4 个分析师 — 含 TimeGuard 二次检查"""
        market_data = state.get("market_data")
        if market_data is None:
            return {"analyses": [], "disagreement": 0.0, "errors": ["无市场数据"]}

        # ── TimeGuard 二次检查（传递到分析师前的最后防线） ──
        time_guard = TimeGuard()
        try:
            time_guard.check_market_data(market_data)
            for kline_key, klines in market_data.klines.items():
                time_guard.check_klines(klines, label=f"klines.{kline_key}")
        except TimeGuardError as e:
            logger.warning(f"分析阶段时间完整性违规，丢弃问题数据: {e}")
            market_data.klines = {}

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
        if debate_result is not None:
            confidence_downgrade = debate_result.get("consensus_confidence") or 1.0

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
            risk_result = await self.risk_control.review(
                decision, MarketData(symbol=decision.symbol),
            )

        return {"risk": risk_result}

    # ── 执行/拒绝终端节点 ──

    async def execute(self, state: AgentState) -> dict[str, Any]:
        """执行交易 — 通过 Exchange 下单或 dryRun 记录"""
        decision = state.get("decision")
        if decision is None:
            return {"execution_result": {"status": "no_decision"}}

        if self.exchange and not self.exchange.dry_run:
            try:
                order = await self.exchange.place_order(
                    symbol=decision.symbol,
                    side="buy" if decision.decision == "LONG" else "sell",
                    order_type=decision.entry.get("type", "limit"),
                    quantity=decision.entry.get("quantity", 0),
                    price=decision.entry.get("price", 0),
                )
                execution = {
                    "status": "live",
                    "order_id": order.order_id,
                    "order": order,
                }
            except Exception as e:
                execution = {"status": "error", "error": str(e)}
        else:
            execution = {
                "status": "dry_run",
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

    # ── 阶段 6: CVRF 学习 ──

    async def learn(self, state: AgentState) -> dict[str, Any]:
        """
        CVRF 学习节点：交易后自动评估、反思、存入记忆库。

        处理路径:
        - dry_run:  完整评估（模拟出场，立即学习）
        - live:     同 dry_run（当前市价快照作为模拟出场）
        - rejected: 记录风控否决教训，不做交易评估
        - error:    跳过学习
        - no_decision: 跳过学习
        """
        self._ensure_cvrf()

        execution_result = state.get("execution_result") or {}
        status = execution_result.get("status", "unknown")
        decision = state.get("decision")
        market_data = state.get("market_data")

        if decision is None:
            logger.warning("learn: 跳过 — 无决策记录")
            return {"errors": ["无决策记录，跳过 CVRF 学习"]}

        if status == "error":
            logger.warning(
                "learn: 跳过 — 执行出错: %s", execution_result.get("error"),
            )
            return {
                "errors": [
                    f"执行出错，跳过 CVRF 学习: {execution_result.get('error')}",
                ],
            }

        if status == "no_decision":
            logger.warning("learn: 跳过 — 无决策可执行")
            return {"errors": []}

        try:
            analyses = state.get("analyses", [])
            analysis_summary = self._summarize_analyses(analyses)
            market_context = self._format_market_context(market_data)

            if status in ("dry_run", "live"):
                return await self._learn_from_execution(
                    decision, market_data, status,
                    analysis_summary, market_context,
                )
            elif status == "rejected":
                return await self._learn_from_rejection(
                    decision, state.get("risk"),
                    analysis_summary, market_context,
                )
            else:
                logger.warning("learn: 跳过未知状态 '%s'", status)
                return {"errors": []}

        except Exception as e:
            logger.error("CVRF 学习节点异常: %s", e, exc_info=True)
            return {"errors": [f"CVRF 学习节点异常: {e}"]}

    async def _learn_from_execution(
        self,
        decision,
        market_data,
        status: str,
        analysis_summary: str,
        market_context: str,
    ) -> dict[str, Any]:
        """处理已执行交易的学习流程"""
        entry_price = decision.entry.get("price", 0)
        current_price = self._get_current_price(market_data)

        # 对于刚执行的交易，使用当前市价作为模拟出场价。
        # 实盘交易的实际盈亏需等平仓后评估（此处仅为快照）。
        exit_price = current_price or entry_price

        trade = TradeRecord(
            trade_id=f"{status}_{decision.symbol}_{int(time_module.time())}",
            symbol=decision.symbol,
            decision=decision.decision,
            entry_price=entry_price,
            exit_price=exit_price,
            position_size=decision.entry.get("quantity", 0),
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow(),
            stop_loss=(
                decision.stop_loss.get("price")
                if decision.stop_loss else None
            ),
            take_profit=[
                tp.get("price", 0) for tp in (decision.take_profit or [])
            ],
            slippage_bps=0.0,
            is_dry_run=(status == "dry_run"),
        )

        entry = await self.cvrf_pipeline.process_trade(
            trade, analysis_summary, market_context,
        )

        lesson_count = len(entry.lessons)
        logger.info(
            "CVRF 学习完成: %d 条教训 (trade=%s, status=%s)",
            lesson_count, trade.trade_id, status,
        )

        evaluation = TradeEvaluation(
            pnl_abs=entry.trade_outcome.get("pnl_abs", 0),
            pnl_pct=entry.trade_outcome.get("pnl_pct", 0),
            hold_duration="0s",
            mae=entry.trade_outcome.get("mae", 0),
            mfe=entry.trade_outcome.get("mfe", 0),
            slippage=entry.trade_outcome.get("slippage", 0),
            execution_quality="simulated",
            lessons=entry.lessons,
        )

        return {
            "evaluation": evaluation,
            "debug_info": {
                "memory_entry": entry.model_dump(mode="json"),
                "memory_entry_id": entry.id,
            },
        }

    async def _learn_from_rejection(
        self,
        decision,
        risk,
        analysis_summary: str,
        market_context: str,
    ) -> dict[str, Any]:
        """处理被风控拒绝交易的学习流程"""
        lessons = self._build_rejection_lessons(decision, risk)

        # 提取市况特征（使用虚拟交易记录）
        dummy_trade = TradeRecord(
            trade_id=f"rejected_{decision.symbol}_{int(time_module.time())}",
            symbol=decision.symbol,
            decision=decision.decision,
            entry_price=decision.entry.get("price", 0),
            exit_price=0,
            position_size=0,
            entry_time=datetime.utcnow(),
            exit_time=datetime.utcnow(),
            is_dry_run=True,
        )
        condition = await self.cvrf_reflection.extract_market_condition(dummy_trade)

        # 构建记忆条目（无交易盈亏，仅有拒绝教训）
        entry = MemoryEntry(
            symbol=decision.symbol,
            timestamp=datetime.utcnow(),
            market_condition=condition,
            lessons=lessons,
            trade_outcome={
                "status": "rejected",
                "reason": (
                    f"风控否决: {', '.join(risk.vetoed_by)}"
                    if risk and risk.vetoed_by
                    else "CVaR 约束"
                ),
            },
            was_bull_correct=None,
            was_bear_correct=None,
        )

        entry_id = self.memory_store.save(entry)
        logger.info(
            "CVRF 拒绝教训已记录 (id=%d, %d 条)",
            entry_id, len(lessons),
        )
        entry.id = entry_id

        return {
            "debug_info": {
                "memory_entry": entry.model_dump(mode="json"),
                "memory_entry_id": entry_id,
            },
        }

    # ── 构建 LangGraph ──

    def _build_graph(self) -> StateGraph:
        """构建六阶段 StateGraph（含 CVRF 学习节点）"""
        workflow = StateGraph(AgentState)

        # 注册节点
        workflow.add_node("collect_data", self.collect_data)
        workflow.add_node("analyze", self.analyze)
        workflow.add_node("debate", self.debate)
        workflow.add_node("decide", self.decide)
        workflow.add_node("review_risk", self.review_risk)
        workflow.add_node("execute", self.execute)
        workflow.add_node("rejected", self.reject)
        workflow.add_node("learn", self.learn)  # CVRF 学习节点

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

        # 执行/拒绝 → CVRF 学习 → 结束
        workflow.add_edge("execute", "learn")
        workflow.add_edge("rejected", "learn")
        workflow.add_edge("learn", END)

        return workflow.compile()

    async def run(self, symbol: str, timeframe: str = "1h") -> AgentState:
        """运行一次完整流水线（含 CVRF 学习）"""
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
