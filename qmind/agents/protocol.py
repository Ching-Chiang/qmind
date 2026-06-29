"""
Agent 间结构化通信协议 Schema。

每个分析师/研究员/交易员/风控角色的输出 JSON Schema，
确保各阶段输出格式一致，可被 Pydantic 校验。
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# 分析师通信协议
# ──────────────────────────────────────────────

class Signal(BaseModel):
    """单个信号"""
    name: str = ""
    value: Any = None
    direction: Literal["bullish", "bearish", "neutral"] = "neutral"
    weight: float = Field(default=1.0, ge=0.0, le=1.0)
    description: str = ""


class TechnicalReport(BaseModel):
    """技术面分析师报告"""
    analyst: Literal["technical"] = "technical"
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    core_reason: str
    key_signals: list[Signal] = []
    risk_factors: list[str] = []
    support_price: float | None = None
    resistance_price: float | None = None
    trend_analysis: str = ""
    indicator_summary: dict[str, Any] = {}
    pattern_recognition: list[str] = []
    risk_reward_assessment: str = ""


class FundamentalReport(BaseModel):
    """基本面分析师报告"""
    analyst: Literal["fundamental"] = "fundamental"
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    core_reason: str
    key_signals: list[Signal] = []
    risk_factors: list[str] = []
    valuation_metrics: dict[str, Any] = {}
    industry_comparison: str = ""
    financial_health: str = ""
    growth_outlook: str = ""


class SentimentReport(BaseModel):
    """市场情绪分析师报告"""
    analyst: Literal["sentiment"] = "sentiment"
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    core_reason: str
    key_signals: list[Signal] = []
    risk_factors: list[str] = []
    long_short_ratio: float | None = None
    funding_rate: float | None = None
    open_interest_change: float | None = None
    social_media_sentiment: float | None = None
    whale_activity: list[str] = []


class NewsReport(BaseModel):
    """宏观/新闻分析师报告"""
    analyst: Literal["news"] = "news"
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    core_reason: str
    key_signals: list[Signal] = []
    risk_factors: list[str] = []
    macro_events: list[dict[str, Any]] = []
    policy_impact: str = ""
    economic_indicators: dict[str, Any] = {}


# ──────────────────────────────────────────────
# 研究员（辩论）通信协议
# ──────────────────────────────────────────────

class BullArgument(BaseModel):
    """多方论证"""
    position: Literal["bull"] = "bull"
    core_thesis: str
    evidence: list[str] = []
    key_levels: dict[str, float] = {}
    counter_argument_rebuttal: str = ""


class BearArgument(BaseModel):
    """空方论证"""
    position: Literal["bear"] = "bear"
    core_thesis: str
    evidence: list[str] = []
    key_levels: dict[str, float] = {}
    counter_argument_rebuttal: str = ""


class DebateRoundLog(BaseModel):
    """辩论轮次记录"""
    round_number: int
    bull_statement: str = ""
    bear_statement: str = ""
    convergence_score: float = 0.0


class DebateResultProtocol(BaseModel):
    """辩论结果"""
    rounds_completed: int = 0
    converged: bool = False
    final_assessment: str = ""
    disagreement_points: list[str] = []
    agreement_points: list[str] = []
    confidence_downgrade: float = Field(default=0.0, ge=0.0, le=1.0)
    position_size_reduction: float = Field(default=0.0, ge=0.0, le=1.0)


# ──────────────────────────────────────────────
# 交易员通信协议
# ──────────────────────────────────────────────

class CoTChain(BaseModel):
    """三级推理链"""
    data_cot: str = ""
    concept_cot: str = ""
    thesis_cot: str = ""


class OrderInstruction(BaseModel):
    """订单指令"""
    type: Literal["market", "limit", "stop_market", "stop_limit"] = "limit"
    price: float = 0.0
    quantity: float = 0.0
    order_type: Literal["GTC", "IOC", "FOK"] = "GTC"


class TakeProfitTarget(BaseModel):
    """止盈目标"""
    price: float
    ratio: float = Field(default=0.5, ge=0.0, le=1.0)
    reason: str = ""


class TradeInstruction(BaseModel):
    """完整的交易指令"""
    decision: Literal["LONG", "SHORT", "HOLD"]
    symbol: str = ""
    entry: OrderInstruction = Field(default_factory=OrderInstruction)
    stop_loss: OrderInstruction | None = None
    take_profit: list[TakeProfitTarget] = []
    position_size_pct: float = 0.0
    confidence: float = 0.0
    time_horizon: str = ""
    reasoning_chain: CoTChain = Field(default_factory=CoTChain)
    risk_reward_ratio: float = 0.0
    max_acceptable_loss_pct: float = 0.0


# ──────────────────────────────────────────────
# 风控通信协议
# ──────────────────────────────────────────────

class RiskReview(BaseModel):
    """单个风控审核"""
    role: Literal["aggressive", "conservative", "neutral"]
    decision: Literal["approve", "reject", "modify"]
    reason: str = ""
    risk_assessment: str = ""
    suggested_position_size_pct: float | None = None
    suggested_stop_loss: float | None = None
    concerns: list[str] = []


class CVaRCheck(BaseModel):
    """CVaR 硬约束检查结果"""
    passed: bool = False
    current_exposure: float = 0.0
    cvar_threshold: float = 0.0
    margin: float = 0.0
    calculation_details: str = ""


class RiskFinalVerdict(BaseModel):
    """风控终审结果"""
    approved: bool = False
    veto_count: int = 0
    vetoed_by: list[str] = []
    adjustments: dict[str, Any] = {}
    aggressive_review: RiskReview | None = None
    conservative_review: RiskReview | None = None
    neutral_review: RiskReview | None = None
    cvar_check: CVaRCheck | None = None
    final_position_size_pct: float = 0.0
    final_stop_loss: float | None = None
