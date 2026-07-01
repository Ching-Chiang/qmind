"""
消融实验框架 — 单 Agent 基线 vs 多 Agent 对比。

核心用途:
    在同一回测窗口上运行三种配置，比较绩效指标，
    量化多 Agent / 辩论的净增量贡献。

三种配置:
    1. single_agent:      直接使用 SingleTradingAgent，跳过分析师/辩论/风控
    2. multi_no_debate:   4 分析师并行 → 单 Agent 决策 → 三角风控 → 执行
    3. multi_debate:      4 分析师并行 → 修正辩论 (分歧驱动) → 决策 → 风控 → 执行

度量标准 (Alpha Illusion P1-P6 兼容):
    - Gross/Net PnL (必须同时报告两者)
    - Sharpe Ratio / Sortino Ratio / Calmar Ratio
    - Max Drawdown
    - Win Rate / Profit Factor / Expectancy
    - 总 LLM 调用次数 + 成本
    - 成本效率 (PnL per $ of LLM cost)
    - 平均分歧度 δ + 辩论触发率

辩论贡献度:
    debate_contribution = multi_debate.net_pnl - multi_no_debate.net_pnl
    如果为负，说明辩论消耗了价值而非创造价值。

LLM 调用预估 (tokens/call):
    分析师:       ~2000 input / ~500 output
    单 Agent:     ~3000 input / ~800 output
    风控:         ~1500 input / ~400 output
    Trust:        ~2000 input / ~500 output
    Skeptic:      ~2000 input / ~500 output
    Debate Leader: ~3000 input / ~600 output

设计原则 (基于 CLAUDE.md Phase 2.5 消融实验要求):
    1. 三种配置在同一回测窗口运行 (同一 walk-forward split)
    2. 每笔交易成本通过 CostModel 显式计算
    3. LLM 调用成本单独追踪 (不混入交易成本)
    4. 辩论轮次 Token 成本计入净 PnL (辩论成本 > 收益 = 负贡献)
    5. 角色分歧度 / 相似度测量
    6. 单 Agent 强基线 — 多 Agent 必须证明净增量贡献

TODO:
    - Phase 2/3 完成后接入真实回测引擎
    - 集成校准模块 (ECE <= 0.05 校验)
    - 跨体制分段报告 (牛市/熊市/震荡市)
"""

from __future__ import annotations

import logging
import random
import statistics
from datetime import datetime, timedelta
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from qmind.backtest.cost_model import CostModel, CostConfig
from qmind.backtest.partition import WalkForwardPartition
from qmind.learning.evaluator import TradeEvaluator, TradeRecord
from qmind.llm.client import LLMClient, CostTracker, MODEL_PRICING

logger = logging.getLogger(__name__)


# ============================================================================
# 消融配置枚举
# ============================================================================


class AblationConfig(StrEnum):
    """消融实验的三种配置名称。"""

    SINGLE_AGENT = "single_agent"
    """单 Agent 基线 — 无分析师、无辩论、无风控。"""

    MULTI_NO_DEBATE = "multi_no_debate"
    """多 Agent 无辩论 — 4 分析师 + 风控，跳过辩论。"""

    MULTI_DEBATE = "multi_debate"
    """多 Agent 含辩论 — 4 分析师 + 修正辩论 + 风控。"""


# ============================================================================
# LLM 调用协议 — 每种配置的预估调用模式
# ============================================================================

# 每种调用的预估 token 用量 (input_tokens, output_tokens)
# 基于实际 prompt 长度的保守估算
CALL_PROFILES: dict[str, tuple[int, int]] = {
    "technical_analyst": (2000, 500),
    "fundamental_analyst": (2000, 500),
    "sentiment_analyst": (2000, 500),
    "news_analyst": (2000, 500),
    "trust_agent": (2000, 500),
    "skeptic_agent": (2000, 500),
    "debate_leader": (3000, 600),
    "single_agent": (3000, 800),
    "single_agent_with_context": (4000, 1000),
    "risk_aggressive": (1500, 400),
    "risk_conservative": (1500, 400),
    "risk_neutral": (1500, 400),
}

# 每种调用的模型分配 (与实际 qmind 实现一致)
CALL_MODELS: dict[str, str] = {
    "technical_analyst": "claude-sonnet-4-6",
    "fundamental_analyst": "gpt-4o",
    "sentiment_analyst": "deepseek-chat",
    "news_analyst": "claude-sonnet-4-6",
    "trust_agent": "claude-sonnet-4-6",
    "skeptic_agent": "claude-sonnet-4-6",
    "debate_leader": "claude-sonnet-4-6",
    "single_agent": "claude-sonnet-4-6",
    "single_agent_with_context": "claude-sonnet-4-6",
    "risk_aggressive": "claude-sonnet-4-6",
    "risk_conservative": "claude-opus-4-8",
    "risk_neutral": "claude-sonnet-4-6",
}

# 三种配置的完整 LLM 调用模式
CONFIG_CALL_PATTERNS: dict[AblationConfig, list[tuple[str, int]]] = {
    AblationConfig.SINGLE_AGENT: [
        ("single_agent", 1),
    ],
    AblationConfig.MULTI_NO_DEBATE: [
        ("technical_analyst", 1),
        ("fundamental_analyst", 1),
        ("sentiment_analyst", 1),
        ("news_analyst", 1),
        ("single_agent_with_context", 1),
        ("risk_aggressive", 1),
        ("risk_conservative", 1),
        ("risk_neutral", 1),
    ],
    AblationConfig.MULTI_DEBATE: [
        ("technical_analyst", 1),
        ("fundamental_analyst", 1),
        ("sentiment_analyst", 1),
        ("news_analyst", 1),
        ("trust_agent", 1),
        ("skeptic_agent", 1),
        ("debate_leader", 1),
        ("single_agent_with_context", 1),
        ("risk_aggressive", 1),
        ("risk_conservative", 1),
        ("risk_neutral", 1),
    ],
}


def estimate_call_cost(call_name: str) -> float:
    """根据预估 token 用量和模型定价估算单次 LLM 调用的成本。

    Args:
        call_name: CALL_PROFILES 中的键名。

    Returns:
        估算的美元成本。
    """
    model = CALL_MODELS.get(call_name, "claude-sonnet-4-6")
    pricing = MODEL_PRICING.get(model)
    if pricing is None:
        return 0.0
    in_tok, out_tok = CALL_PROFILES.get(call_name, (1000, 300))
    input_cost = (in_tok / 1_000_000) * pricing["input"]
    output_cost = (out_tok / 1_000_000) * pricing["output"]
    return round(input_cost + output_cost, 8)


def compute_config_llm_metrics(config: AblationConfig) -> dict[str, Any]:
    """计算给定配置下的预估 LLM 调用指标。

    Returns:
        dict 包含:
        - total_calls: 总调用次数
        - total_cost: 总成本 (USD)
        - breakdown: 按调用类型细分的列表
    """
    pattern = CONFIG_CALL_PATTERNS.get(config, [])
    total_calls = sum(count for _, count in pattern)
    total_cost = 0.0
    breakdown: list[dict[str, Any]] = []

    for call_name, count in pattern:
        unit_cost = estimate_call_cost(call_name)
        subtotal = unit_cost * count
        total_cost += subtotal
        breakdown.append({
            "call_name": call_name,
            "model": CALL_MODELS.get(call_name, "unknown"),
            "count": count,
            "unit_cost": round(unit_cost, 8),
            "subtotal": round(subtotal, 8),
            "input_tokens": CALL_PROFILES.get(call_name, (0, 0))[0] * count,
            "output_tokens": CALL_PROFILES.get(call_name, (0, 0))[1] * count,
        })

    return {
        "total_calls": total_calls,
        "total_cost": round(total_cost, 8),
        "breakdown": breakdown,
    }


# ============================================================================
# 核心数据类型
# ============================================================================


class AblationResult(BaseModel):
    """单次消融实验运行的结果指标。

    所有指标在同一回测窗口上计算，确保可比性。
    每笔交易的 LLM 成本已计入 net_pnl_pct (从净收益中扣除)。
    """

    name: str = Field(
        description="配置名称: single_agent / multi_no_debate / multi_debate"
    )
    symbol: str = Field(description="交易标的")
    timeframe: str = Field(description="时间框架，如 1h, 4h, 1d")
    period: tuple[datetime, datetime] = Field(
        description="回测时间范围 (start, end)"
    )

    # ── 交易统计 ──
    total_trades: int = Field(ge=0, description="总交易次数")
    winning_trades: int = Field(ge=0, description="盈利交易次数")
    losing_trades: int = Field(ge=0, description="亏损交易次数")
    win_rate: float = Field(ge=0.0, le=1.0, description="胜率 = winning / total")

    # ── 收益指标 (Alpha Illusion 要求必须同时报告 Gross 和 Net) ──
    gross_pnl_pct: float = Field(description="Gross 收益率 (%)，不含交易成本")
    net_pnl_pct: float = Field(
        description="Net 收益率 (%)，扣除所有交易成本 + LLM 成本"
    )
    gross_pnl_abs: float = Field(description="Gross 绝对收益 (quote currency)")
    net_pnl_abs: float = Field(description="Net 绝对收益 (quote currency)")

    # ── 风险指标 ──
    sharpe_ratio: float = Field(description="年化 Sharpe Ratio (基于 Net PnL)")
    sortino_ratio: float = Field(default=0.0, description="年化 Sortino Ratio")
    max_drawdown_pct: float = Field(
        le=0.0, description="最大回撤 (%)，负值表示亏损"
    )
    calmar_ratio: float = Field(
        default=0.0,
        description="Calmar Ratio = 年化收益率 / |最大回撤|",
    )

    # ── 持仓特征 ──
    avg_holding_periods: float = Field(
        ge=0, description="平均持仓周期数 (以 timeframe 为单位)"
    )
    avg_holding_hours: float = Field(ge=0, description="平均持仓小时数")

    # ── LLM 成本追踪 ──
    total_llm_calls: int = Field(ge=0, description="总 LLM 调用次数")
    total_llm_cost: float = Field(ge=0.0, description="总 LLM 成本 (USD)")
    total_trading_cost: float = Field(
        ge=0.0, description="总交易成本 (USD, 佣金+滑点+点差)"
    )
    llm_cost_per_trade: float = Field(
        ge=0.0, description="每笔交易平均 LLM 成本 (USD)"
    )

    # ── 校准指标 (Phase 1 后集成) ──
    ece: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Expected Calibration Error (None = 未计算)",
    )
    calibration_passed: bool | None = Field(
        default=None,
        description="ECE <= 0.05 是否通过 (None = 未计算)",
    )

    # ── 分歧指标 (仅多 Agent 配置有意义) ──
    avg_disagreement: float = Field(
        default=0.0,
        ge=0.0,
        description="分析师间平均分歧度 δ (0-1)",
    )
    debate_rate: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="触发辩论的交易占比 (仅 multi_debate)",
    )

    # ── 交易明细 (用于成本敏感性分析) ──
    trades_detail: list[TradeRecord] = Field(
        default_factory=list,
        description="完整交易记录列表 (用于 CostModel.generate_report)",
    )

    # ── 追加质量指标 ──
    profit_factor: float = Field(
        default=0.0,
        description="盈亏比 = 总盈利 / |总亏损| (inf 表示无亏损交易)",
    )
    avg_win_pct: float = Field(default=0.0, description="平均盈利百分比")
    avg_loss_pct: float = Field(default=0.0, description="平均亏损百分比")
    expectancy: float = Field(
        default=0.0,
        description="期望收益 = win_rate * avg_win - (1-win_rate) * |avg_loss|",
    )


class ComparisonReport(BaseModel):
    """跨配置的消融对比报告。

    在同一回测窗口上对比三种配置的绩效，
    量化辩论贡献度和成本效率。
    """

    # ── 元信息 ──
    symbol: str = Field(description="交易标的")
    timeframe: str = Field(description="时间框架")
    period: tuple[datetime, datetime] = Field(description="回测时间范围")
    n_splits: int = Field(default=1, description="walk-forward 折数 (1=单窗口)")

    # ── 三套结果 ──
    results: list[AblationResult] = Field(
        description="三种配置的消融结果 (通常 3 个)"
    )

    # ── 最佳配置排名 (在 model_post_init 中自动计算) ──
    best_by_sharpe: str = Field(default="", description="按 Sharpe Ratio 的最优配置名")
    best_by_net_pnl: str = Field(default="", description="按 Net PnL 的最优配置名")
    best_by_cost_efficiency: str = Field(default="", description="按成本效率的最优配置名")
    best_by_calmar: str = Field(default="", description="按 Calmar Ratio 的最优配置名")
    best_by_sortino: str = Field(default="", description="按 Sortino Ratio 的最优配置名")

    # ── 辩论贡献度 (核心消融指标, model_post_init 自动计算) ──
    debate_contribution: float = Field(
        default=0.0,
        description="辩论净贡献 (百分点) = multi_debate.net_pnl - multi_no_debate.net_pnl",
    )
    debate_cost: float = Field(
        default=0.0,
        description="辩论额外成本 (USD) = multi_debate.llm_cost - multi_no_debate.llm_cost",
    )
    debate_contribution_net_of_cost: float = Field(
        default=0.0,
        description="扣除辩论成本后的净贡献 (百分点)",
    )

    # ── 成本效率 (model_post_init 自动计算) ──
    cost_efficiency: dict[str, float] = Field(
        default_factory=dict,
        description="各配置的 PnL per $ of LLM cost (net_pnl_pct / total_llm_cost)",
    )

    # ── 消融效果摘要 (model_post_init 自动计算) ──
    multi_agent_improvement: float = Field(
        default=0.0,
        description="多 Agent 相对单 Agent 的 Net PnL 提升 (百分点)",
    )
    debate_over_no_debate: float = Field(
        default=0.0,
        description="辩论相比无辩论的 Net PnL 变化 (百分点)",
    )

    # ── 分歧统计 ──
    avg_disagreement_across_configs: dict[str, float] = Field(
        default_factory=dict,
        description="各配置的平均分歧度",
    )

    # ── Alpha 幻觉指标 (Alpha Illusion P5) ──
    alpha_hallucination: dict[str, float] = Field(
        default_factory=dict,
        description="各配置的 alpha 幻觉 (百分点) = gross_pnl_pct - net_pnl_pct",
    )

    def model_post_init(self, __context: Any) -> None:
        """自动计算对比指标。"""
        self._compute_rankings()
        self._compute_debate_contribution()
        self._compute_cost_efficiency()
        self._compute_alpha_hallucination()
        self._compute_disagreement_stats()

    def _compute_rankings(self) -> None:
        """按各指标排序，确定最佳配置。"""
        if not self.results:
            return

        self.best_by_sharpe = max(self.results, key=lambda r: r.sharpe_ratio).name
        self.best_by_net_pnl = max(self.results, key=lambda r: r.net_pnl_pct).name
        self.best_by_calmar = max(self.results, key=lambda r: r.calmar_ratio).name
        self.best_by_sortino = max(self.results, key=lambda r: r.sortino_ratio).name

    def _compute_debate_contribution(self) -> None:
        """计算辩论贡献度 — 这是消融实验的核心指标。"""
        no_debate = _find_result(self.results, "multi_no_debate")
        with_debate = _find_result(self.results, "multi_debate")
        single = _find_result(self.results, "single_agent")

        if no_debate and with_debate:
            self.debate_contribution = round(
                with_debate.net_pnl_pct - no_debate.net_pnl_pct, 4
            )
            self.debate_cost = round(
                with_debate.total_llm_cost - no_debate.total_llm_cost, 8
            )
            # 将额外成本转换为 PnL 百分比
            # 假设账户规模 = 10000 USD (与 AblationStudy.account_balance 一致)
            implied_account = 10000.0
            cost_as_pnl_pct = (self.debate_cost / implied_account) * 100
            self.debate_contribution_net_of_cost = round(
                self.debate_contribution - cost_as_pnl_pct, 4
            )
            self.debate_over_no_debate = self.debate_contribution
        else:
            self.debate_contribution = 0.0
            self.debate_cost = 0.0
            self.debate_contribution_net_of_cost = 0.0
            self.debate_over_no_debate = 0.0

        if single and no_debate:
            self.multi_agent_improvement = round(
                no_debate.net_pnl_pct - single.net_pnl_pct, 4
            )
        else:
            self.multi_agent_improvement = 0.0

    def _compute_cost_efficiency(self) -> None:
        """计算各配置的成本效率 (每 1 USD LLM 成本带来多少 % 净收益)。"""
        self.cost_efficiency = {}
        for r in self.results:
            if r.total_llm_cost > 0:
                self.cost_efficiency[r.name] = round(
                    r.net_pnl_pct / r.total_llm_cost, 6
                )
            else:
                self.cost_efficiency[r.name] = float("inf")

        if self.cost_efficiency:
            self.best_by_cost_efficiency = max(
                self.cost_efficiency, key=self.cost_efficiency.get  # type: ignore[arg-type]
            )

    def _compute_alpha_hallucination(self) -> None:
        """计算各配置的 alpha 幻觉 = Gross - Net 差值。

        Alpha Illusion 论文要求: Gross - Net 的差值 = alpha 幻觉的量化度量。
        差值越大，说明越多的表面 alpha 被交易成本和 LLM 成本消耗。
        """
        self.alpha_hallucination = {}
        for r in self.results:
            self.alpha_hallucination[r.name] = round(
                r.gross_pnl_pct - r.net_pnl_pct, 4
            )

    def _compute_disagreement_stats(self) -> None:
        """收集各配置的平均分歧度。"""
        self.avg_disagreement_across_configs = {}
        for r in self.results:
            self.avg_disagreement_across_configs[r.name] = r.avg_disagreement

    def to_dict(self) -> dict[str, Any]:
        """将报告转为纯字典 (用于序列化 / JSON 输出)。"""
        return self.model_dump(mode="json")

    def summary_table(self) -> str:
        """生成终端可读的对比摘要表。"""
        lines: list[str] = []
        sep = "=" * 80
        sub = "-" * 80

        lines.append(sep)
        lines.append(f"  消融实验对比报告: {self.symbol} ({self.timeframe})")
        lines.append(
            f"  回测窗口: {self.period[0].strftime('%Y-%m-%d')} → "
            f"{self.period[1].strftime('%Y-%m-%d')}"
        )
        if self.n_splits > 1:
            lines.append(f"  Walk-Forward 折数: {self.n_splits}")
        lines.append(sep)

        # 指标对比表
        header = (
            f"  {'配置':<20} {'Gross%':>8} {'Net%':>8} {'Sharpe':>8} "
            f"{'Sortino':>8} {'DD%':>8} {'WR%':>6} {'Cost$':>8} {'Calls':>6}"
        )
        lines.append(header)
        lines.append(sub)

        for r in sorted(self.results, key=lambda x: x.net_pnl_pct, reverse=True):
            lines.append(
                f"  {r.name:<20}"
                f" {r.gross_pnl_pct:>7.2f}%"
                f" {r.net_pnl_pct:>7.2f}%"
                f" {r.sharpe_ratio:>7.2f}"
                f" {r.sortino_ratio:>7.2f}"
                f" {r.max_drawdown_pct:>7.2f}%"
                f" {r.win_rate * 100:>5.1f}%"
                f" {r.total_llm_cost:>7.6f}"
                f" {r.total_llm_calls:>5d}"
            )

        lines.append(sub)

        # 辩论贡献
        lines.append("")
        lines.append(f"  debate_contribution:           {self.debate_contribution:+.2f}%")
        lines.append(f"  debate_cost:                   ${self.debate_cost:.6f}")
        lines.append(
            f"  debate_contribution_net_of_cost: "
            f"{self.debate_contribution_net_of_cost:+.2f}%"
        )
        lines.append(f"  multi_agent_improvement:       {self.multi_agent_improvement:+.2f}%")
        lines.append("")

        # 成本效率
        lines.append(sub)
        lines.append("  成本效率 (Net PnL% / LLM Cost USD):")
        for name, eff in sorted(
            self.cost_efficiency.items(),
            key=lambda x: x[1] if x[1] != float("inf") else 1e9,
            reverse=True,
        ):
            eff_str = f"{eff:.4f}" if eff != float("inf") else "inf"
            lines.append(f"    {name:<20}: {eff_str}")
        lines.append("")

        # Alpha 幻觉
        lines.append(sub)
        lines.append("  Alpha 幻觉 (Gross - Net, 百分点 — 越小越好):")
        for name, gap in sorted(
            self.alpha_hallucination.items(),
            key=lambda x: x[1],
            reverse=True,
        ):
            lines.append(f"    {name:<20}: {gap:+.2f}%")
        lines.append("")

        # 最佳配置
        lines.append(sub)
        lines.append(f"  最佳 Sharpe:          {self.best_by_sharpe}")
        lines.append(f"  最佳 Net PnL:        {self.best_by_net_pnl}")
        lines.append(f"  最佳 Calmar:          {self.best_by_calmar}")
        lines.append(f"  最佳 Sortino:         {self.best_by_sortino}")
        lines.append(f"  最佳成本效率:        {self.best_by_cost_efficiency}")
        lines.append(sep)

        return "\n".join(lines)


def _find_result(
    results: list[AblationResult], name: str
) -> AblationResult | None:
    """在结果列表中按 name 查找 AblationResult。"""
    return next((r for r in results if r.name == name), None)


# ============================================================================
# 消融实验执行器
# ============================================================================


class AblationStudy:
    """消融实验框架 — 比较单 Agent / 多 Agent / 辩论 三种配置。

    使用方法::

        study = AblationStudy(llm_client=client)

        # 单窗口运行
        results = await study.run_all(
            symbol="BTC/USDT", timeframe="1h",
            start=datetime(2024, 1, 1), end=datetime(2024, 12, 31),
        )
        report = study.compare(results)
        print(report.summary_table())

        # Walk-forward 批量运行
        reports = await study.run_walk_forward(
            symbol="BTC/USDT", timeframe="1h",
            data=df, n_splits=5,
        )

    设计原则:
        1. 三种配置在相同的回测窗口上运行 (同一 walk-forward split)
        2. 每笔交易的成本通过 CostModel 显式计算
        3. LLM 调用成本单独追踪 (不混入交易成本)
        4. 辩论贡献 = (有辩论 - 无辩论) 的 Net PnL 差
        5. 所有指标支持 Alpha Illusion P1-P6 报告标准
    """

    # 模拟交易参数
    ACCOUNT_BALANCE: float = 10000.0
    """模拟账户余额 (USD)，用于计算收益率。"""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        cost_model: CostModel | None = None,
        cost_tracker: CostTracker | None = None,
        trade_evaluator: TradeEvaluator | None = None,
    ):
        """初始化消融实验框架。

        Args:
            llm_client: LLM 客户端。如果为 None，则使用模拟成本估算。
            cost_model: 交易成本模型。如果为 None，则使用默认配置。
            cost_tracker: LLM 成本追踪器。如果为 None，则新建一个。
            trade_evaluator: 交易评估器。如果为 None，则使用默认实现。
        """
        self.llm_client = llm_client
        self.cost_model = cost_model or CostModel()
        self.cost_tracker = cost_tracker or CostTracker()
        self.trade_evaluator = trade_evaluator or TradeEvaluator()

    # ── 三种运行模式 ──────────────────────────────────────────────────

    async def run_single_agent(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> AblationResult:
        """运行 **单 Agent** 配置。

        流程:
            1. 单 Agent 直接分析市场数据 → 输出 TradeDecision
            2. (跳过分析师/辩论/风控)
            3. 直接按决策执行
            4. 评估交易结果
            5. 收集指标

        LLM 调用: 1 次 (SingleTradingAgent.analyze, claude-sonnet-4-6)

        Args:
            symbol: 交易标的。
            timeframe: 时间框架。
            start: 回测开始时间。
            end: 回测结束时间。

        Returns:
            包含全部绩效指标的 AblationResult。
        """
        logger.info(
            "[Ablation] running single_agent: %s %s [%s → %s]",
            symbol, timeframe, start.date(), end.date(),
        )

        llm_metrics = compute_config_llm_metrics(AblationConfig.SINGLE_AGENT)

        mock_trades = self._generate_mock_trades(
            symbol=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            win_rate=0.48,       # 单 Agent 基线的假设胜率 (无分析师辅助)
            avg_return=0.012,    # 平均单笔收益 1.2%
            avg_holding=6,       # 平均持仓 6 根 K 线
            disagreement=None,   # 单 Agent 无分歧概念
        )

        return self._compute_result(
            name="single_agent",
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            mock_trades=mock_trades,
            llm_metrics=llm_metrics,
        )

    async def run_multi_agent_no_debate(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> AblationResult:
        """运行 **多 Agent 无辩论** 配置。

        流程:
            1. 4 个分析师并行分析 → AnalystReport x4
            2. 分歧检测: δ < 0.15 跳过辩论
            3. 单 Agent 综合分析师报告 → TradeDecision
            4. 三角风控审核 (激进/保守/中立)
            5. CVaR 硬约束校验
            6. 执行
            7. 评估

        LLM 调用: 8 次 (4 分析师 + 1 决策 + 3 风控)

        Args:
            symbol: 交易标的。
            timeframe: 时间框架。
            start: 回测开始时间。
            end: 回测结束时间。

        Returns:
            包含全部绩效指标的 AblationResult。
        """
        logger.info(
            "[Ablation] running multi_no_debate: %s %s [%s → %s]",
            symbol, timeframe, start.date(), end.date(),
        )

        llm_metrics = compute_config_llm_metrics(AblationConfig.MULTI_NO_DEBATE)

        # 多 Agent (无辩论) 假设:
        # - 分析师辅助提高胜率
        # - 风控减少大亏损
        # - 平均分歧度低 (< 0.15)
        mock_trades = self._generate_mock_trades(
            symbol=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            win_rate=0.52,
            avg_return=0.015,
            avg_holding=8,
            disagreement=0.10,
        )

        return self._compute_result(
            name="multi_no_debate",
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            mock_trades=mock_trades,
            llm_metrics=llm_metrics,
        )

    async def run_multi_agent_with_debate(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> AblationResult:
        """运行 **多 Agent 含辩论** 配置。

        流程:
            1. 4 个分析师并行分析 → AnalystReport x4
            2. 分歧检测: δ >= 0.15 → 启动辩论
            3. Trust Agent 验证论据 → Skeptic Agent 找漏洞
            4. Debate Leader 输出降级因子 (置信度降级 + 仓位缩减)
            5. 单 Agent 决策 (注入辩论降级)
            6. 三角风控审核
            7. CVaR 硬约束校验
            8. 执行
            9. 评估

        LLM 调用: 11+ 次 (4 分析师 + 3 辩论 + 1 决策 + 3 风控)

        Args:
            symbol: 交易标的。
            timeframe: 时间框架。
            start: 回测开始时间。
            end: 回测结束时间。

        Returns:
            包含全部绩效指标的 AblationResult。
        """
        logger.info(
            "[Ablation] running multi_debate: %s %s [%s → %s]",
            symbol, timeframe, start.date(), end.date(),
        )

        llm_metrics = compute_config_llm_metrics(AblationConfig.MULTI_DEBATE)

        # 多 Agent (含辩论) 预期:
        # - 降低最大回撤 (辩论过滤高风险交易)
        # - 胜率略高 (辩论过滤噪音信号)
        # - Net PnL 可能低于 no_debate (辩论成本 > 收益)
        # - 这是消融实验要验证的核心假设
        # - 部分高分歧交易被辩论降级/取消
        mock_trades = self._generate_mock_trades(
            symbol=symbol,
            start=start,
            end=end,
            timeframe=timeframe,
            win_rate=0.54,
            avg_return=0.013,
            avg_holding=9,
            disagreement=0.22,
        )

        return self._compute_result(
            name="multi_debate",
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            mock_trades=mock_trades,
            llm_metrics=llm_metrics,
        )

    # ── 批量运行 ──────────────────────────────────────────────────────────

    async def run_all(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[AblationResult]:
        """并行运行全部三种配置。

        Args:
            symbol: 交易标的。
            timeframe: 时间框架。
            start: 回测开始时间。
            end: 回测结束时间。

        Returns:
            三个 AblationResult 的列表。
        """
        import asyncio

        results = await asyncio.gather(
            self.run_single_agent(symbol, timeframe, start, end),
            self.run_multi_agent_no_debate(symbol, timeframe, start, end),
            self.run_multi_agent_with_debate(symbol, timeframe, start, end),
        )
        return list(results)

    def run_all_sync(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> list[AblationResult]:
        """同步版本的 run_all (用于测试或非异步环境)。"""
        import asyncio
        return asyncio.run(self.run_all(symbol, timeframe, start, end))

    # ── 对比方法 ──────────────────────────────────────────────────────────

    @staticmethod
    def compare(
        results: list[AblationResult],
        symbol: str = "",
        timeframe: str = "",
        period: tuple[datetime, datetime] | None = None,
        n_splits: int = 1,
    ) -> ComparisonReport:
        """比较多个消融实验结果，生成对比报告。

        Args:
            results: 至少一个 AblationResult 的列表。
            symbol: 交易标的 (自动从结果推断)。
            timeframe: 时间框架 (自动从结果推断)。
            period: 回测时间范围 (自动从结果推断)。
            n_splits: walk-forward 折数，默认 1。

        Returns:
            包含跨配置对比分析的 ComparisonReport。

        Raises:
            ValueError: results 为空。
        """
        if not results:
            raise ValueError("至少需要一个 AblationResult 进行比较")

        # 从第一个结果推断元信息
        first = results[0]
        inferred_symbol = symbol or first.symbol
        inferred_tf = timeframe or first.timeframe
        inferred_period = period or first.period

        return ComparisonReport(
            symbol=inferred_symbol,
            timeframe=inferred_tf,
            period=inferred_period,
            n_splits=n_splits,
            results=results,
        )

    # ── Walk-Forward 批量消融 ─────────────────────────────────────────

    async def run_walk_forward(
        self,
        symbol: str,
        timeframe: str,
        data: "Any",  # pandas DataFrame — 延迟导入避免硬依赖
        n_splits: int = 5,
        date_column: str = "timestamp",
    ) -> list[ComparisonReport]:
        """在 walk-forward 划分的多个窗口上运行消融实验。

        每个 fold 独立运行三种配置，生成对应时间窗口的 ComparisonReport。
        可用于评估配置绩效在不同市场体制下的稳定性。

        Args:
            symbol: 交易标的。
            timeframe: 时间框架。
            data: 含时间戳列的 pandas DataFrame。
            n_splits: walk-forward 折数。
            date_column: 时间戳列名。

        Returns:
            每个 fold 对应一个 ComparisonReport 的列表。

        Raises:
            TypeError: data 不是 pandas DataFrame。
        """
        import pandas as pd

        if not isinstance(data, pd.DataFrame):
            raise TypeError("data 必须是 pandas DataFrame")

        partitioner = WalkForwardPartition(n_splits=n_splits)
        splits = partitioner.split(data, date_column=date_column)

        reports: list[ComparisonReport] = []
        for split in splits:
            fold_data = partitioner.get_fold_data(data, split)
            test_df: pd.DataFrame = fold_data["test"]
            if test_df.empty:
                logger.warning("Fold %d: 测试集为空，跳过", split.fold)
                continue

            window_start = test_df[date_column].min().to_pydatetime()
            window_end = test_df[date_column].max().to_pydatetime()

            results = await self.run_all(
                symbol=symbol,
                timeframe=timeframe,
                start=window_start,
                end=window_end,
            )

            report = self.compare(
                results=results,
                symbol=symbol,
                timeframe=timeframe,
                period=(window_start, window_end),
                n_splits=n_splits,
            )
            reports.append(report)

            logger.info(
                "[Ablation] Fold %d/%d done: best_by_net_pnl=%s",
                split.fold + 1,
                n_splits,
                report.best_by_net_pnl,
            )

        return reports

    # ── 内部方法 ────────────────────────────────────────────────────────

    def _generate_mock_trades(
        self,
        symbol: str,
        start: datetime,
        end: datetime,
        timeframe: str,
        win_rate: float,
        avg_return: float,
        avg_holding: int,
        disagreement: float | None,
        n_trades: int = 50,
    ) -> list["_MockTrade"]:
        """生成模拟交易数据 (框架阶段使用)。

        Phase 1 中，此方法生成合成交易以验证框架结构和指标计算。
        后续接入真实回测引擎后将替换为实际交易记录。

        Args:
            symbol: 交易标的。
            start: 开始时间。
            end: 结束时间。
            timeframe: 时间框架。
            win_rate: 模拟胜率。
            avg_return: 平均单笔收益率。
            avg_holding: 平均持仓 K 线根数。
            disagreement: 平均分歧度 (None 表示单 Agent 配置)。
            n_trades: 模拟交易数量上限。

        Returns:
            _MockTrade 列表。
        """
        random.seed(42)  # 固定种子确保可复现性
        trades: list[_MockTrade] = []
        base_price = 50000.0

        tf_minutes = _timeframe_to_minutes(timeframe)
        total_minutes = int((end - start).total_seconds() / 60)
        step_minutes = max(avg_holding, 1) * tf_minutes

        current_minute = random.randint(0, step_minutes - 1)
        trade_count = 0

        while (
            current_minute + step_minutes < total_minutes
            and trade_count < n_trades
        ):
            trade_time = start + timedelta(minutes=current_minute)
            is_long = random.random() < 0.55
            is_win = random.random() < win_rate

            ret = avg_return * (1 + random.gauss(0, 0.5))
            if not is_win:
                ret = -ret * random.uniform(0.8, 1.2)

            entry = base_price * (1 + random.gauss(0, 0.02))
            # 防止负价格
            entry = max(entry, 0.01)
            exit_price = entry * (1 + ret) if is_long else entry * (1 - ret)
            exit_price = max(exit_price, 0.01)

            confidence = round(0.5 + random.random() * 0.4, 2)

            trades.append(
                _MockTrade(
                    timestamp=trade_time,
                    decision="LONG" if is_long else "SHORT",
                    entry_price=round(entry, 2),
                    exit_price=round(exit_price, 2),
                    position_size=self.ACCOUNT_BALANCE * 0.1 / entry,
                    confidence=confidence,
                    is_winning=is_win,
                    disagreement=disagreement,
                )
            )

            step_variation = random.randint(-2, 4) * tf_minutes
            current_minute += step_minutes + step_variation
            trade_count += 1

        logger.debug(
            "Generated %d mock trades (win_rate=%.2f, avg_return=%.4f)",
            len(trades), win_rate, avg_return,
        )
        return trades

    def _compute_result(
        self,
        name: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        mock_trades: list["_MockTrade"],
        llm_metrics: dict[str, Any],
    ) -> AblationResult:
        """从模拟交易和 LLM 指标计算完整的 AblationResult。"""
        if not mock_trades:
            return self._empty_result(name, symbol, timeframe, start, end, llm_metrics)

        # 1. 转为 TradeRecord (用于成本计算)
        trade_records = self._mock_to_trade_records(mock_trades, symbol, timeframe, name)

        # 2. 计算 Gross PnL 和追踪回撤
        gross_pnl_sum = 0.0
        daily_returns: list[float] = []
        peak_balance = self.ACCOUNT_BALANCE
        current_balance = self.ACCOUNT_BALANCE
        max_dd = 0.0

        winning_trades = 0
        losing_trades = 0
        total_win_pnl = 0.0
        total_loss_pnl = 0.0
        total_holding_periods = 0
        total_holding_hours = 0.0
        tf_minutes = _timeframe_to_minutes(timeframe)

        for mt in mock_trades:
            if mt.decision == "HOLD":
                continue

            pnl_abs = _compute_pnl_abs(mt)
            pnl_pct = _compute_pnl_pct(mt)

            gross_pnl_sum += pnl_abs
            current_balance += pnl_abs

            if pnl_abs > 0:
                winning_trades += 1
                total_win_pnl += pnl_abs
            else:
                losing_trades += 1
                total_loss_pnl += abs(pnl_abs)

            # Max Drawdown (基于账户余额曲线)
            if current_balance > peak_balance:
                peak_balance = current_balance
            dd = (current_balance - peak_balance) / peak_balance
            if dd < max_dd:
                max_dd = dd

            daily_returns.append(pnl_pct)

            # 持仓周期估算
            holding_k = max(1, _estimate_holding(timeframe))
            total_holding_periods += holding_k
            total_holding_hours += holding_k * tf_minutes / 60.0

        n_trades = len(mock_trades)

        # 3. 计算收益
        gross_return_pct = gross_pnl_sum / self.ACCOUNT_BALANCE * 100.0

        # 4. 计算交易成本
        cost_total = self.cost_model.calculate_total_cost(trade_records)
        trading_cost_abs = cost_total.total_cost

        # 5. LLM 成本
        total_llm_cost = llm_metrics["total_cost"]

        # 6. Net PnL = Gross - 交易成本 - LLM 成本
        net_pnl_abs = gross_pnl_sum - trading_cost_abs - total_llm_cost
        net_return_pct = net_pnl_abs / self.ACCOUNT_BALANCE * 100.0

        # 7. 风险指标
        sharpe = _compute_sharpe(daily_returns)
        sortino = _compute_sortino(daily_returns)
        calmar = (
            round(net_return_pct / abs(max_dd * 100), 4)
            if max_dd != 0
            else 0.0
        )

        # 8. 胜率和其他质量指标
        win_rate_val = winning_trades / n_trades if n_trades > 0 else 0.0
        profit_factor = (
            total_win_pnl / total_loss_pnl if total_loss_pnl > 0 else float("inf")
        )
        avg_win = (total_win_pnl / winning_trades) if winning_trades > 0 else 0.0
        avg_loss = (total_loss_pnl / losing_trades) if losing_trades > 0 else 0.0
        avg_win_pct = avg_win / self.ACCOUNT_BALANCE * 100.0
        avg_loss_pct = avg_loss / self.ACCOUNT_BALANCE * 100.0
        expectancy = win_rate_val * avg_win_pct - (1 - win_rate_val) * avg_loss_pct

        avg_holding_periods_val = (
            total_holding_periods / n_trades if n_trades > 0 else 0.0
        )
        avg_holding_hours_val = (
            total_holding_hours / n_trades if n_trades > 0 else 0.0
        )

        # 9. 分歧统计
        disagreements = [
            mt.disagreement for mt in mock_trades if mt.disagreement is not None
        ]
        avg_disagreement = (
            sum(disagreements) / len(disagreements) if disagreements else 0.0
        )
        debate_rate = (
            sum(1 for d in disagreements if d is not None and d >= 0.15)
            / len(disagreements)
            if disagreements
            else 0.0
        )

        llm_cost_per_trade = total_llm_cost / n_trades if n_trades > 0 else 0.0

        return AblationResult(
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            period=(start, end),
            total_trades=n_trades,
            winning_trades=winning_trades,
            losing_trades=losing_trades,
            win_rate=round(win_rate_val, 6),
            gross_pnl_pct=round(gross_return_pct, 4),
            net_pnl_pct=round(net_return_pct, 4),
            gross_pnl_abs=round(gross_pnl_sum, 2),
            net_pnl_abs=round(net_pnl_abs, 2),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            max_drawdown_pct=round(max_dd * 100.0, 4),
            calmar_ratio=round(calmar, 4),
            avg_holding_periods=round(avg_holding_periods_val, 2),
            avg_holding_hours=round(avg_holding_hours_val, 2),
            total_llm_calls=llm_metrics["total_calls"],
            total_llm_cost=round(total_llm_cost, 8),
            total_trading_cost=round(trading_cost_abs, 2),
            llm_cost_per_trade=round(llm_cost_per_trade, 8),
            avg_disagreement=round(avg_disagreement, 4),
            debate_rate=round(debate_rate, 4),
            trades_detail=trade_records,
            profit_factor=(
                round(profit_factor, 4)
                if profit_factor != float("inf")
                else float("inf")
            ),
            avg_win_pct=round(avg_win_pct, 4),
            avg_loss_pct=round(avg_loss_pct, 4),
            expectancy=round(expectancy, 4),
        )

    def _empty_result(
        self,
        name: str,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
        llm_metrics: dict[str, Any],
    ) -> AblationResult:
        """生成一个空交易列表的 AblationResult (零值填充)。"""
        return AblationResult(
            name=name,
            symbol=symbol,
            timeframe=timeframe,
            period=(start, end),
            total_trades=0,
            winning_trades=0,
            losing_trades=0,
            win_rate=0.0,
            gross_pnl_pct=0.0,
            net_pnl_pct=0.0,
            gross_pnl_abs=0.0,
            net_pnl_abs=0.0,
            sharpe_ratio=0.0,
            max_drawdown_pct=0.0,
            avg_holding_periods=0.0,
            avg_holding_hours=0.0,
            total_llm_calls=llm_metrics["total_calls"],
            total_llm_cost=llm_metrics["total_cost"],
            total_trading_cost=0.0,
            llm_cost_per_trade=0.0,
        )

    def _mock_to_trade_records(
        self,
        mock_trades: list["_MockTrade"],
        symbol: str,
        timeframe: str,
        config_name: str,
    ) -> list[TradeRecord]:
        """将 _MockTrade 列表转换为 TradeRecord 列表 (用于成本计算)。"""
        tf_minutes = _timeframe_to_minutes(timeframe)
        records: list[TradeRecord] = []

        for i, mt in enumerate(mock_trades):
            if mt.decision == "HOLD":
                continue

            holding_minutes = tf_minutes * max(1, _estimate_holding(timeframe))
            exit_time = mt.timestamp + timedelta(minutes=holding_minutes)

            records.append(
                TradeRecord(
                    trade_id=f"{config_name}_{symbol}_{i}",
                    symbol=symbol,
                    decision=mt.decision,
                    entry_price=mt.entry_price,
                    exit_price=mt.exit_price,
                    position_size=mt.position_size,
                    entry_time=mt.timestamp,
                    exit_time=exit_time,
                    is_dry_run=True,
                )
            )

        return records


# ============================================================================
# 内部数据结构
# ============================================================================


class _MockTrade:
    """用于构造模拟交易的内部数据结构。"""

    __slots__ = (
        "timestamp", "decision", "entry_price", "exit_price",
        "position_size", "confidence", "is_winning", "disagreement",
    )

    def __init__(
        self,
        timestamp: datetime,
        decision: Literal["LONG", "SHORT", "HOLD"],
        entry_price: float,
        exit_price: float,
        position_size: float,
        confidence: float,
        is_winning: bool,
        disagreement: float | None = None,
    ) -> None:
        self.timestamp = timestamp
        self.decision = decision
        self.entry_price = entry_price
        self.exit_price = exit_price
        self.position_size = position_size
        self.confidence = confidence
        self.is_winning = is_winning
        self.disagreement = disagreement


# ============================================================================
# 内部工具函数
# ============================================================================


def _timeframe_to_minutes(tf: str) -> int:
    """将时间框架字符串转为分钟数。

    Args:
        tf: 时间框架，如 "1m", "5m", "1h", "4h", "1d"。

    Returns:
        对应的分钟数。
    """
    tf = tf.strip().lower()
    if tf.endswith("m"):
        return int(tf[:-1])
    elif tf.endswith("h"):
        return int(tf[:-1]) * 60
    elif tf.endswith("d"):
        return int(tf[:-1]) * 1440
    elif tf.endswith("w"):
        return int(tf[:-1]) * 10080
    else:
        # 默认 1h
        return 60


def _estimate_holding(timeframe: str) -> int:
    """估算默认持仓周期数 (基于时间框架的随机值)。

    Returns:
        持仓 K 线根数 (至少 1)。
    """
    tf_minutes = _timeframe_to_minutes(timeframe)
    if tf_minutes <= 5:
        return random.randint(3, 12)
    elif tf_minutes <= 60:
        return random.randint(3, 10)
    elif tf_minutes <= 1440:
        return random.randint(2, 8)
    else:
        return random.randint(1, 4)


def _compute_pnl_abs(mt: _MockTrade) -> float:
    """计算一笔交易的绝对盈亏 (quote currency)。"""
    if mt.decision == "LONG":
        return (mt.exit_price - mt.entry_price) * mt.position_size
    else:
        return (mt.entry_price - mt.exit_price) * mt.position_size


def _compute_pnl_pct(mt: _MockTrade) -> float:
    """计算一笔交易的收益率 (小数)。"""
    if mt.decision == "LONG":
        return (mt.exit_price - mt.entry_price) / mt.entry_price
    else:
        return (mt.entry_price - mt.exit_price) / mt.entry_price


def _compute_sharpe(daily_returns: list[float]) -> float:
    """计算年化 Sharpe Ratio (假设 252 个交易日)。"""
    if len(daily_returns) < 2:
        return 0.0
    mean_ret = statistics.mean(daily_returns)
    std_ret = statistics.stdev(daily_returns)
    if std_ret <= 0:
        return 0.0
    return round(mean_ret / std_ret * (252**0.5), 4)


def _compute_sortino(daily_returns: list[float]) -> float:
    """计算年化 Sortino Ratio (仅下行偏差)。"""
    if len(daily_returns) < 2:
        return 0.0
    downside = [r for r in daily_returns if r < 0]
    if not downside:
        return 0.0
    downside_std = statistics.stdev(downside) if len(downside) > 1 else 0.01
    if downside_std <= 0:
        return 0.0
    mean_ret = statistics.mean(daily_returns)
    return round(mean_ret / downside_std * (252**0.5), 4)
