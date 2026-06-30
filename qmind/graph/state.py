"""
QMind — AgentState TypedDict for LangGraph.

每个阶段更新 state，LangGraph 的 StateGraph 通过 reducer 管理状态转移。
所有字段 optional，因为各阶段逐步填充。
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

# ──────────────────────────────────────────────
# Market Data Models (Pydantic)
# ──────────────────────────────────────────────

class OHLCV(BaseModel):
    """单根 K 线"""
    timestamp: int  # Unix ms
    open: float
    high: float
    low: float
    close: float
    volume: float
    as_of: datetime | None = None


class OrderBookLevel(BaseModel):
    price: float
    size: float


class MarketData(BaseModel):
    """统一格式的市场数据包"""
    symbol: str
    klines: dict[str, list[OHLCV]] = {}  # timeframe -> klines
    orderbook: dict[str, list[OrderBookLevel]] = {}
    funding_rate: float | None = None
    open_interest: float | None = None
    news: list[dict[str, Any]] = []
    timestamp: int = 0
    as_of: datetime | None = None


# ──────────────────────────────────────────────
# Analyst Reports
# ──────────────────────────────────────────────

class AnalystReport(BaseModel):
    """分析师输出的结构化报告"""
    analyst: str  # fundamental / technical / sentiment / news
    stance: Literal["bullish", "neutral", "bearish"]
    confidence: float = Field(ge=0.0, le=1.0)
    core_reason: str
    key_signals: list[dict[str, Any]] = []
    risk_factors: list[str] = []
    support_price: float | None = None
    resistance_price: float | None = None
    details: str = ""


class DebateRound(BaseModel):
    """单轮辩论记录"""
    round_number: int
    bull_argument: str
    bear_argument: str
    bull_evidence: list[str] = []
    bear_evidence: list[str] = []


class DebateResult(BaseModel):
    """辩论纪要"""
    rounds: int = 0
    converged: bool = False
    final_stance: Literal["bullish", "bearish", "neutral"] | None = None
    bull_core_argument: str = ""
    bear_core_counter: str = ""
    agreement_points: list[str] = []
    disagreement_points: list[str] = []
    consensus_confidence: float = 0.0
    debate_transcript: list[DebateRound] = []


class TradeDecision(BaseModel):
    """交易员输出的结构化决策指令"""
    decision: Literal["LONG", "SHORT", "HOLD"]
    symbol: str = ""
    entry: dict[str, Any] = {}
    stop_loss: dict[str, Any] = {}
    take_profit: list[dict[str, Any]] = []
    position_size_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    time_horizon: str = ""
    reasoning_chain: dict[str, str] = {}
    risk_reward_ratio: float = 0.0
    max_acceptable_loss_pct: float = 0.0


class RiskOpinion(BaseModel):
    """单个风控角色的审核意见"""
    role: Literal["aggressive", "conservative", "neutral"]
    approved: bool = True
    reason: str = ""
    suggested_adjustments: dict[str, Any] = {}


class RiskResult(BaseModel):
    """风控审核结果"""
    approved: bool = False
    veto_count: int = 0
    vetoed_by: list[str] = []
    adjustments: dict[str, Any] = {}
    aggressive_opinion: str = ""
    conservative_opinion: str = ""
    neutral_opinion: str = ""
    cvar_check: dict[str, Any] = {}
    final_position_size_pct: float = 0.0


class Lesson(BaseModel):
    """CVRF 学习教训"""
    lesson: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str = ""


class TradeEvaluation(BaseModel):
    """交易结果评估"""
    pnl_abs: float = 0.0
    pnl_pct: float = 0.0
    hold_duration: str = ""
    mae: float = 0.0
    mfe: float = 0.0
    slippage: float = 0.0
    execution_quality: str = ""
    lessons: list[Lesson] = []


# ──────────────────────────────────────────────
# AgentState — LangGraph 全局状态
# ──────────────────────────────────────────────

def _merge_dicts(a: dict, b: dict) -> dict:
    if a is None:
        a = {}
    if b is None:
        b = {}
    return {**a, **b}


def _merge_lists(a: list, b: list) -> list:
    if a is None:
        a = []
    if b is None:
        b = []
    return a + b


class AgentState(TypedDict):
    """LangGraph 全局状态 — 五阶段流水线共享"""

    # 元信息
    symbol: str
    timeframe: str
    timestamp: int

    # 阶段 1: 数据采集
    market_data: MarketData | None

    # 阶段 2: 多维分析
    analyses: Annotated[list[AnalystReport], _merge_lists]

    # 阶段 3: 多空辩论
    debate: DebateResult | None
    disagreement: float

    # 阶段 4: 交易决策
    decision: TradeDecision | None

    # 阶段 5: 风控审核
    risk: RiskResult | None

    # 执行结果
    execution_result: dict[str, Any] | None

    # 学习
    evaluation: TradeEvaluation | None

    # 系统
    errors: Annotated[list[str], _merge_lists]
    debug_info: Annotated[dict[str, Any], _merge_dicts]


# ──────────────────────────────────────────────
# CVRF / Memory Models
# ──────────────────────────────────────────────

class MarketConditionVector(BaseModel):
    """市况特征向量（用于 CVRF 相似度检索）"""
    trend: str = ""
    volatility: str = ""
    market_cycle: str = ""
    momentum: float = 0.0
    volume_trend: str = ""
    extra: dict[str, float] = {}


class MemoryEntry(BaseModel):
    """记忆库条目"""
    id: int | None = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    symbol: str = ""
    market_condition: MarketConditionVector = Field(default_factory=MarketConditionVector)
    lessons: list[Lesson] = []
    trade_outcome: dict[str, Any] = {}
    was_bull_correct: bool | None = None
    was_bear_correct: bool | None = None
    embedding: list[float] | None = None
