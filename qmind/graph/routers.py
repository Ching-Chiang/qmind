"""
LangGraph 条件路由函数。

基于论文审阅结论:
- 分歧 δ < 0.15 → 跳过辩论，直接采信最强 Agent
- 分歧 δ >= 0.15 → 进入风控审核模式（方向锁定 + 单轮 + 仓位缩减）
- 任一风控否决 → 取消执行，记录原因
"""

from __future__ import annotations

import logging

from qmind.graph.state import AgentState

logger = logging.getLogger(__name__)


def route_after_analysis(state: AgentState) -> str:
    """分析师结束后路由: 根据分歧度决定是否跳过辩论"""
    disagreement = state.get("disagreement", 0.0)

    if disagreement < 0.15:
        logger.info(f"分歧度 δ={disagreement:.3f} < 0.15 → 跳过辩论，直接决策")
        return "skip_debate"  # 跳转到决策
    else:
        logger.info(f"分歧度 δ={disagreement:.3f} >= 0.15 → 启动辩论+风控审核")
        return "start_debate"  # 进入辩论


def route_after_risk(state: AgentState) -> str:
    """风控结束后路由: 任一否决则取消"""
    risk = state.get("risk")
    if risk is None:
        return "execute"

    if risk.approved:
        logger.info("风控通过，进入执行")
        return "execute"
    else:
        vetoed = risk.vetoed_by
        logger.warning(f"风控否决 ({', '.join(vetoed) if vetoed else 'CVaR约束未通过'}) → 取消交易")
        return "rejected"  # 交易被取消


def route_after_debate(state: AgentState) -> str:
    """辩论结束后路由: 降级后进入决策或直接取消"""
    return "decide"


def route_after_decision(state: AgentState) -> str:
    """决策结束后路由: 进入风控"""
    return "review_risk"
