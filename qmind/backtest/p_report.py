"""
P1-P6 Standard Report Generator — Alpha Illusion Compliance Protocol.

Implements the six-level reporting protocol defined in the Alpha Illusion paper
(arXiv 2605.16895, Ye et al. 2026).  Every level represents a methodological
safeguard that must be documented for a backtest result to be considered
deployable evidence rather than alpha hallucination.

Protocol Levels
---------------
P1 — Time Consistency
    Report train/val/test periods explicitly.  Use walk-forward or expanding
    window methodology.  No random shuffling.  Minimum 3-5 years of data
    spanning multiple market regimes.

P2 — Point-in-Time Data
    Confirm that every data point carries an ``as_of`` timestamp such that no
    information from the future leaks into the prompt.  The time guard must
    be enforced at query time.

P3 — Execution Timing
    Signal is generated at t close, execution occurs at t+1 close
    (next-close model).  Report the slippage model used.

P4 — Cost Realism
    Report Gross PnL and Net PnL side-by-side at 0 / 10 / 25 bps cost tiers.
    The gap (Gross - Net) is a quantitative measure of alpha hallucination.

P5 — Benchmark Comparison
    Compare strategy returns against buy-and-hold, equal-weight portfolio,
    and a relevant market index.

P6 — Ablation
    Break down single-agent baseline vs. multi-agent contributions.  Report
    debate cost in token usage and its impact on Net PnL.

Usage
-----
    from qmind.backtest.p_report import P1P6Report

    report = P1P6Report.generate(backtest_results, config)
    print(report.to_markdown())
    with open("report.html", "w") as f:
        f.write(report.to_html())
"""

from __future__ import annotations

import html
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from qmind.backtest.partition import TimeSplit

# ============================================================================
# P1 — Time Consistency
# ============================================================================


class P1Report(BaseModel):
    """P1: Time Consistency report.

    Documents the walk-forward / expanding-window methodology used to
    partition the backtest, including explicit train/val/test boundaries
    for every fold, the minimum training duration, and whether the protocol
    recommendation of >= 3-5 years spanning multiple regimes is met.

    Attributes:
        method: Partitioning strategy — ``"walk_forward"`` or
            ``"expanding_window"``.
        n_splits: Number of cross-validation folds.
        train_periods: ``(start, end)`` for each fold's training set.
        val_periods: ``(start, end)`` for each fold's validation set.
        test_periods: ``(start, end)`` for each fold's test set.
        min_train_days: Shortest training period across all folds, in
            calendar days.
        total_span_days: Total calendar span covered by the full dataset.
        spans_multiple_regimes: Whether the total span is >= 3 years
            (1095 days), the minimum recommended for meaningful multi-regime
            testing.
        passed: Whether basic P1 compliance is met — non-empty splits,
            train precedes val precedes test, and total span >= 365 days.
    """
    method: str = Field(
        description="Partitioning method: walk_forward | expanding_window"
    )
    n_splits: int = Field(ge=1, description="Number of folds")
    train_periods: list[tuple[datetime, datetime]] = Field(
        default_factory=list,
        description="(start, end) per fold for training set",
    )
    val_periods: list[tuple[datetime, datetime]] = Field(
        default_factory=list,
        description="(start, end) per fold for validation set",
    )
    test_periods: list[tuple[datetime, datetime]] = Field(
        default_factory=list,
        description="(start, end) per fold for test set",
    )
    min_train_days: int = Field(
        ge=0,
        description="Shortest training period across all folds (calendar days)",
    )
    total_span_days: int = Field(
        ge=0,
        description="Total calendar span of the full dataset (days)",
    )
    spans_multiple_regimes: bool = Field(
        default=False,
        description="True if total_span_days >= 1095 (3 years)",
    )
    passed: bool = Field(
        default=False,
        description="P1 compliance: splits exist and span >= 365 days",
    )


# ============================================================================
# P2 — Point-in-Time Data
# ============================================================================


class P2Report(BaseModel):
    """P2: Point-in-Time Data report.

    Confirms that every data point consumed during the backtest carries an
    ``as_of`` timestamp that prevents look-ahead bias.  The time-guard
    mechanism must be enforced at query time, and any detected bias is
    reported.

    Attributes:
        as_of_timestamps: Whether all market data included an ``as_of``
            field.
        time_guard_enforced: Whether the time-guard module was active
            during evaluation.
        max_look_ahead_bias_days: Worst-case look-ahead detected (0 if
            perfect).
        data_sources: List of data-source names (e.g. ``"yfinance"``,
            ``"tushare"``, ``"akshare"``).
        n_data_points: Total market-data points consumed.
        passed: True when ``max_look_ahead_bias_days == 0``.
    """
    as_of_timestamps: bool = Field(
        default=False,
        description="All data carried as_of timestamps",
    )
    time_guard_enforced: bool = Field(
        default=False,
        description="TimeGuard was active during evaluation",
    )
    max_look_ahead_bias_days: float = Field(
        ge=0.0,
        description="Worst detected look-ahead in days (0 = none)",
    )
    data_sources: list[str] = Field(
        default_factory=list,
        description="Data sources used in the backtest",
    )
    n_data_points: int = Field(
        ge=0,
        description="Total market-data points consumed",
    )
    passed: bool = Field(
        default=False,
        description="P2 compliance: no look-ahead detected",
    )


# ============================================================================
# P3 — Execution Timing
# ============================================================================


class P3Report(BaseModel):
    """P3: Execution Timing report.

    Documents the signal-to-execution model.  The protocol requires that a
    signal produced at bar t close is executed at bar t+1 close
    (``next_close``), not at the close of the same bar (``same_close``) which
    would constitute look-ahead.

    Attributes:
        signal_execution_model: ``"next_close"`` (compliant) or
            ``"same_close"`` (non-compliant).
        slippage_model: Description of the slippage model used (e.g.
            ``"fixed_10bps"``, ``"volume_weighted_5bps"``).
        avg_slippage_bps: Average realised slippage across all trades in
            basis points.
        fill_rate: Fraction of signals that resulted in a fill
            (``0.0`` to ``1.0``).
        passed: True when ``signal_execution_model == "next_close"``.
    """
    signal_execution_model: str = Field(
        description="next_close | same_close",
    )
    slippage_model: str = Field(
        default="",
        description="Description of the slippage model",
    )
    avg_slippage_bps: float = Field(
        ge=0.0,
        description="Average slippage in basis points across all trades",
    )
    fill_rate: float = Field(
        ge=0.0,
        le=1.0,
        description="Fraction of signals that filled",
    )
    passed: bool = Field(
        default=False,
        description="P3 compliance: execution_model == next_close",
    )


# ============================================================================
# P4 — Cost Realism
# ============================================================================


class CostTierResult(BaseModel):
    """P&L and risk metrics at a single cost tier.

    Attributes:
        gross_pnl_pct: Gross return before costs (percentage).
        net_pnl_pct: Net return after costs (percentage).
        sharpe: Annualised Sharpe ratio of net returns.
        max_dd_pct: Maximum drawdown of net returns (percentage).
        total_cost_pct: Total frictional cost as a percentage of
            notional traded.
    """
    gross_pnl_pct: float = Field(description="Gross return (%)")
    net_pnl_pct: float = Field(description="Net return after costs (%)")
    sharpe: float = Field(description="Annualised Sharpe ratio (net)")
    max_dd_pct: float = Field(description="Maximum drawdown of net returns (%)")
    total_cost_pct: float = Field(
        ge=0.0,
        description="Total cost as % of total notional",
    )


class P4Report(BaseModel):
    """P4: Cost Realism report.

    Presents Gross and Net PnL at three cost tiers (0, 10, 25 bps) plus
    the detailed layer-by-layer cost breakdown.  The gap between Gross and
    Net PnL is a quantitative measure of alpha hallucination (Alpha Illusion
    §4.2).

    Attributes:
        cost_tiers: Mapping from tier label (e.g. ``"0bps"``) to
            ``CostTierResult``.
        detailed_commission_bps: Commission cost in bps of notional.
        detailed_spread_bps: Bid-ask spread cost in bps.
        detailed_slippage_bps: Slippage cost in bps.
        detailed_gas_bps: Gas fee cost in bps (0 for CEX).
        gross_minus_net_gap_bps: The gap (Gross - Net returns) in bps
            at the realistic tier (10 bps).
        passed: True when at least one cost tier is reported.
    """
    cost_tiers: dict[str, CostTierResult] = Field(
        description='Cost tiers, e.g. {"0bps": ..., "10bps": ..., "25bps": ...}',
    )
    detailed_commission_bps: float = Field(default=0.0, ge=0.0)
    detailed_spread_bps: float = Field(default=0.0, ge=0.0)
    detailed_slippage_bps: float = Field(default=0.0, ge=0.0)
    detailed_gas_bps: float = Field(default=0.0, ge=0.0)
    gross_minus_net_gap_bps: float = Field(
        default=0.0,
        description="Gross - Net gap at the 10 bps tier (bps)",
    )
    passed: bool = Field(
        default=False,
        description="P4 compliance: at least one cost tier reported",
    )


# ============================================================================
# P5 — Benchmark Comparison
# ============================================================================


class BenchmarkResult(BaseModel):
    """Comparison result for a single benchmark.

    Attributes:
        total_return_pct: Total return of the benchmark over the same
            period (percentage).
        annualised_return_pct: Annualised return (percentage).
        sharpe: Annualised Sharpe ratio.
        max_dd_pct: Maximum drawdown (percentage).
        volatility_pct: Annualised volatility (percentage).
    """
    total_return_pct: float = Field(description="Total return (%)")
    annualised_return_pct: float = Field(description="Annualised return (%)")
    sharpe: float = Field(default=0.0, description="Annualised Sharpe ratio")
    max_dd_pct: float = Field(default=0.0, description="Maximum drawdown (%)")
    volatility_pct: float = Field(default=0.0, description="Annualised volatility (%)")


class P5Report(BaseModel):
    """P5: Benchmark Comparison report.

    Compares the strategy's net return against three canonical benchmarks:

    - **Buy & Hold**: Holding the asset for the full backtest duration.
    - **Equal Weight**: An equal-weighted portfolio of the same assets.
    - **Market Index**: A relevant broad-market index (e.g. S&P 500 for
      US equities, CSI 300 for A-shares, or the asset's spot return for
      crypto).

    The protocol requires reporting outperformance in basis points and
    stating whether the strategy beats every benchmark on a net-of-costs
    basis.

    Attributes:
        strategy: ``BenchmarkResult`` for the evaluated strategy (net of
            realistic costs).
        benchmarks: Mapping from benchmark name to ``BenchmarkResult``.
        outperformance_bps: Strategy net return minus best benchmark
            return, in basis points.  Positive means the strategy won.
        beats_all_benchmarks: True if strategy outperforms every
            benchmark.
        passed: True when at least strategy and one benchmark are
            reported.
    """
    strategy: BenchmarkResult | None = Field(
        default=None,
        description="Strategy net-of-costs performance",
    )
    benchmarks: dict[str, BenchmarkResult] = Field(
        default_factory=dict,
        description="Benchmark performance results",
    )
    outperformance_bps: float = Field(
        description="Strategy net return minus best benchmark (bps)",
    )
    beats_all_benchmarks: bool = Field(
        default=False,
        description="Strategy outperforms every benchmark",
    )
    passed: bool = Field(
        default=False,
        description="P5 compliance: strategy + at least one benchmark reported",
    )


# ============================================================================
# P6 — Ablation
# ============================================================================


class AblationResult(BaseModel):
    """Results for one ablation variant.

    Attributes:
        total_return_pct: Net total return of this variant (percentage).
        sharpe: Annualised Sharpe ratio.
        max_dd_pct: Maximum drawdown (percentage).
        n_trades: Number of trades executed.
        avg_holding_period: Average holding period description string.
        total_llm_calls: Total LLM inference calls made by this variant.
        total_token_cost_usd: Estimated total token cost in USD.
    """
    total_return_pct: float = Field(description="Net total return (%)")
    sharpe: float = Field(default=0.0, description="Annualised Sharpe ratio")
    max_dd_pct: float = Field(default=0.0, description="Maximum drawdown (%)")
    n_trades: int = Field(ge=0, description="Number of trades executed")
    avg_holding_period: str = Field(default="", description="Average holding period")
    total_llm_calls: int = Field(ge=0, description="Total LLM inference calls")
    total_token_cost_usd: float = Field(
        ge=0.0,
        description="Estimated total token cost in USD",
    )


class P6Report(BaseModel):
    """P6: Ablation report.

    Decomposes performance by variant to isolate the contribution of each
    component.  The protocol mandates at minimum three columns:

    1. **Single Agent** — a lone LLM making decisions without debate/risk.
    2. **Multi Agent, no debate** — full pipeline with debate silenced.
    3. **Full pipeline** — all agents, debate, and risk active.

    The difference between (3) and (1) is the total multi-agent contribution.
    The difference between (3) and (2) is the marginal debate contribution.
    The cost-performance ratio converts LLM token spend into PnL impact.

    Attributes:
        results: Mapping from variant name (e.g. ``"single_agent"``,
            ``"multi_no_debate"``, ``"full"``) to ``AblationResult``.
        debate_contribution_bps: Net return difference (full - no_debate)
            in basis points attributable to the debate mechanism.
        multi_agent_contribution_bps: Net return difference
            (full - single_agent) in basis points.
        cost_performance_ratio: LLM cost as a fraction of net PnL
            for each variant (``{variant: cost_frac}``).  A value > 1.0
            means LLM costs exceeded trading profits.
        passed: True when at least two variants are reported.
    """
    results: dict[str, AblationResult] = Field(
        default_factory=dict,
        description="Ablation variant results",
    )
    debate_contribution_bps: float = Field(
        default=0.0,
        description="Net return contribution of debate mechanism (bps)",
    )
    multi_agent_contribution_bps: float = Field(
        default=0.0,
        description="Net return contribution of full multi-agent system (bps)",
    )
    cost_performance_ratio: dict[str, float] = Field(
        default_factory=dict,
        description="LLM cost / Net PnL ratio per variant",
    )
    passed: bool = Field(
        default=False,
        description="P6 compliance: at least two variants reported",
    )


# ============================================================================
# Aggregated Report
# ============================================================================


class P1P6ReportResult(BaseModel):
    """Complete P1-P6 compliance report for one backtest run.

    This is the top-level result produced by ``P1P6Report.generate()``.

    Attributes:
        title: Human-readable title for the report.
        timestamp: When this report was generated (UTC).
        symbol: Traded symbol (e.g. ``"BTC/USDT"``, ``"AAPL"``).
        timeframe: Resolution of the data (e.g. ``"1h"``, ``"1d"``).
        p1: Time consistency report.
        p2: Point-in-time data report.
        p3: Execution timing report.
        p4: Cost realism report.
        p5: Benchmark comparison report.
        p6: Optional ablation report (only present when ablation was run).
        summary: One-line summary describing overall compliance.
        n_passed: Number of P levels that passed.
        all_passed: True when all applicable P levels passed.
    """
    title: str = Field(default="P1-P6 Compliance Report")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
    )
    symbol: str = ""
    timeframe: str = ""
    p1: P1Report = Field(default_factory=P1Report)
    p2: P2Report = Field(default_factory=P2Report)
    p3: P3Report = Field(default_factory=P3Report)
    p4: P4Report = Field(default_factory=P4Report)
    p5: P5Report = Field(default_factory=P5Report)
    p6: P6Report | None = Field(
        default=None,
        description="Ablation report (None if ablation not run)",
    )
    summary: str = Field(default="")
    n_passed: int = Field(ge=0, le=6, default=0)
    all_passed: bool = Field(default=False)


# ============================================================================
# Generator
# ============================================================================


class P1P6Report:
    """Factory for producing P1-P6 compliance reports from backtest results.

    Usage
    -----
    .. code-block:: python

        from qmind.backtest.p_report import P1P6Report

        report_result = P1P6Report.generate(backtest_results, config)
        print(report_result.to_markdown())
        # or
        html = report_result.to_html()
    """

    # ------------------------------------------------------------------
    # Builders — each produces one P-level sub-report
    # ------------------------------------------------------------------

    @staticmethod
    def _build_p1(
        results: dict[str, Any],
        config: dict[str, Any],
    ) -> P1Report:
        """Build the P1 report from backtest partition metadata.

        Parameters
        ----------
        results:
            Backtest results dict.  Expected keys:
            - ``"splits"``: list[TimeSplit] from WalkForwardPartition.
            - ``"total_span_days"``: optional int.
        config:
            Backtest config dict.  Expected keys:
            - ``"n_splits"``: int.
            - ``"test_size"``: float.
            - ``"validation_size"``: float.
            - ``"window_mode"``: str, ``"walk_forward"`` or
              ``"expanding_window"``.

        Returns
        -------
        P1Report
        """
        splits: list[TimeSplit] = results.get("splits", [])
        method = config.get("window_mode", "walk_forward")
        # Normalise to protocol vocabulary
        method_label = (
            "expanding_window"
            if method in ("expanding", "expanding_window")
            else "walk_forward"
        )

        n_splits = len(splits) if splits else config.get("n_splits", 1)

        train_periods: list[tuple[datetime, datetime]] = []
        val_periods: list[tuple[datetime, datetime]] = []
        test_periods: list[tuple[datetime, datetime]] = []
        min_train_days = 0

        if splits:
            for s in splits:
                train_periods.append(
                    (s.train_start.to_pydatetime(), s.train_end.to_pydatetime())
                )
                val_periods.append(
                    (s.val_start.to_pydatetime(), s.val_end.to_pydatetime())
                )
                test_periods.append(
                    (s.test_start.to_pydatetime(), s.test_end.to_pydatetime())
                )

            min_train_days = min(
                (s.train_end - s.train_start).days for s in splits
            )

        total_span_days = results.get(
            "total_span_days",
            (
                (splits[-1].test_end - splits[0].train_start).days
                if splits
                else 0
            ),
        )
        spans_multiple_regimes = total_span_days >= 1095
        passed = bool(splits) and total_span_days >= 365

        return P1Report(
            method=method_label,
            n_splits=n_splits,
            train_periods=train_periods,
            val_periods=val_periods,
            test_periods=test_periods,
            min_train_days=min_train_days,
            total_span_days=total_span_days,
            spans_multiple_regimes=spans_multiple_regimes,
            passed=passed,
        )

    @staticmethod
    def _build_p2(
        results: dict[str, Any],
        _config: dict[str, Any],
    ) -> P2Report:
        """Build the P2 report from time-guard and data-source metadata.

        Parameters
        ----------
        results:
            Expected keys:
            - ``"as_of_timestamps"``: bool, default False.
            - ``"time_guard_enforced"``: bool, default False.
            - ``"max_look_ahead_bias_days"``: float, default 0.
            - ``"data_sources"``: list[str], default [].
            - ``"n_data_points"``: int, default 0.
        _config:
            Unused for P2 currently.

        Returns
        -------
        P2Report
        """
        as_of = results.get("as_of_timestamps", False)
        guarded = results.get("time_guard_enforced", False)
        max_bias = float(results.get("max_look_ahead_bias_days", 0.0))
        sources: list[str] = results.get("data_sources", [])
        n_points = int(results.get("n_data_points", 0))

        passed = max_bias == 0.0
        return P2Report(
            as_of_timestamps=as_of,
            time_guard_enforced=guarded,
            max_look_ahead_bias_days=max_bias,
            data_sources=sources,
            n_data_points=n_points,
            passed=passed,
        )

    @staticmethod
    def _build_p3(
        results: dict[str, Any],
        config: dict[str, Any],
    ) -> P3Report:
        """Build the P3 report from execution metadata.

        Parameters
        ----------
        results:
            Expected keys:
            - ``"signal_execution_model"``: str, ``"next_close"`` or
              ``"same_close"``.  Default ``"next_close"``.
            - ``"slippage_model"``: str.
            - ``"avg_slippage_bps"``: float.
            - ``"fill_rate"``: float (0..1).
        config:
            Unused for P3 currently.

        Returns
        -------
        P3Report
        """
        exec_model = results.get(
            "signal_execution_model",
            config.get("signal_execution_model", "next_close"),
        )
        slip_model = results.get("slippage_model", "")
        avg_slip = float(results.get("avg_slippage_bps", 0.0))
        fill_rate = float(results.get("fill_rate", 1.0))

        passed = exec_model == "next_close"
        return P3Report(
            signal_execution_model=exec_model,
            slippage_model=slip_model,
            avg_slippage_bps=avg_slip,
            fill_rate=fill_rate,
            passed=passed,
        )

    @staticmethod
    def _build_p4(
        results: dict[str, Any],
        _config: dict[str, Any],
    ) -> P4Report:
        """Build the P4 report from cost-model results.

        The cost tiers dict inside *results* is expected to have keys like
        ``"0bps"``, ``"10bps"``, ``"25bps"``, each mapping to a dict with
        fields matching ``CostTierResult``.  If the field is absent a default
        is returned.

        Alternatively, the caller can use integer keys ``0.0``, ``10.0``,
        ``25.0`` which are automatically converted to string labels.

        Parameters
        ----------
        results:
            Expected keys:
            - ``"cost_tiers"``: dict[str | float, dict], optional.
            - ``"detailed_commission_bps"``: float.
            - ``"detailed_spread_bps"``: float.
            - ``"detailed_slippage_bps"``: float.
            - ``"detailed_gas_bps"``: float.
        _config:
            Unused for P4 currently.

        Returns
        -------
        P4Report
        """
        raw_tiers: dict = results.get("cost_tiers", {})
        cost_tiers: dict[str, CostTierResult] = {}

        for key, value in raw_tiers.items():
            label = f"{key}" if isinstance(key, str) else f"{int(key)}bps"
            if isinstance(value, dict):
                cost_tiers[label] = CostTierResult(**value)
            elif isinstance(value, CostTierResult):
                cost_tiers[label] = value

        # Fallback: create empty tiers so the report always has structure.
        for default_label in ("0bps", "10bps", "25bps"):
            if default_label not in cost_tiers:
                cost_tiers[default_label] = CostTierResult(
                    gross_pnl_pct=0.0,
                    net_pnl_pct=0.0,
                    sharpe=0.0,
                    max_dd_pct=0.0,
                    total_cost_pct=0.0,
                )

        # Gap at 10 bps tier (the "realistic" default).
        ten_bps = cost_tiers.get("10bps", cost_tiers.get("10.0"))
        gap_bps = (
            (ten_bps.gross_pnl_pct - ten_bps.net_pnl_pct) * 100
            if ten_bps
            else 0.0
        )

        passed = len(cost_tiers) > 0
        return P4Report(
            cost_tiers=dict(cost_tiers),
            detailed_commission_bps=float(
                results.get("detailed_commission_bps", 0.0)
            ),
            detailed_spread_bps=float(
                results.get("detailed_spread_bps", 0.0)
            ),
            detailed_slippage_bps=float(
                results.get("detailed_slippage_bps", 0.0)
            ),
            detailed_gas_bps=float(results.get("detailed_gas_bps", 0.0)),
            gross_minus_net_gap_bps=round(gap_bps, 4),
            passed=passed,
        )

    @staticmethod
    def _build_p5(
        results: dict[str, Any],
        _config: dict[str, Any],
    ) -> P5Report:
        """Build the P5 report from benchmark comparison metadata.

        Parameters
        ----------
        results:
            Expected keys:
            - ``"strategy_result"``: dict matching ``BenchmarkResult``,
              or a ``BenchmarkResult`` instance.
            - ``"benchmarks"``: dict[str, dict | BenchmarkResult].
        _config:
            Unused for P5 currently.

        Returns
        -------
        P5Report
        """
        # Strategy
        raw_strat = results.get("strategy_result")
        strategy: BenchmarkResult | None = None
        if isinstance(raw_strat, BenchmarkResult):
            strategy = raw_strat
        elif isinstance(raw_strat, dict):
            strategy = BenchmarkResult(**raw_strat)

        # Benchmarks
        raw_benches: dict = results.get("benchmarks", {})
        benchmarks: dict[str, BenchmarkResult] = {}
        for name, val in raw_benches.items():
            if isinstance(val, BenchmarkResult):
                benchmarks[name] = val
            elif isinstance(val, dict):
                benchmarks[name] = BenchmarkResult(**val)

        # Outperformance: strategy return minus best benchmark return.
        outperf = 0.0
        beats_all = False
        if strategy is not None and benchmarks:
            best_bench = max(
                b.total_return_pct for b in benchmarks.values()
            )
            outperf = strategy.total_return_pct - best_bench
            beats_all = all(
                strategy.total_return_pct > b.total_return_pct
                for b in benchmarks.values()
            )

        passed = strategy is not None and len(benchmarks) > 0
        return P5Report(
            strategy=strategy,
            benchmarks=benchmarks,
            outperformance_bps=round(outperf * 100, 2),
            beats_all_benchmarks=beats_all,
            passed=passed,
        )

    @staticmethod
    def _build_p6(
        results: dict[str, Any],
        _config: dict[str, Any],
    ) -> P6Report | None:
        """Build the P6 ablation report.

        Returns ``None`` if no ablation data is available.

        Parameters
        ----------
        results:
            Expected keys:
            - ``"ablation"``: dict[str, dict | AblationResult] — mapping
              from variant name to performance data.  Canonical variants:
              ``"single_agent"``, ``"multi_no_debate"``, ``"full"``.
        _config:
            Unused for P6 currently.

        Returns
        -------
        P6Report or None
        """
        raw_ablation: dict = results.get("ablation", {})
        if not raw_ablation:
            return None

        ablation_results: dict[str, AblationResult] = {}
        for name, val in raw_ablation.items():
            if isinstance(val, AblationResult):
                ablation_results[name] = val
            elif isinstance(val, dict):
                ablation_results[name] = AblationResult(**val)

        # Compute contributions.
        full = ablation_results.get("full")
        no_debate = ablation_results.get("multi_no_debate")
        single = ablation_results.get("single_agent")

        debate_contrib = 0.0
        if full is not None and no_debate is not None:
            debate_contrib = (
                full.total_return_pct - no_debate.total_return_pct
            ) * 100  # to bps

        multi_contrib = 0.0
        if full is not None and single is not None:
            multi_contrib = (
                full.total_return_pct - single.total_return_pct
            ) * 100  # to bps

        # Cost-performance ratio: LLM cost / abs(Net PnL).
        cost_perf: dict[str, float] = {}
        for name, ab in ablation_results.items():
            if ab.total_return_pct != 0:
                # Approximate: assume total_return_pct is on a standard
                # notional; we compare token cost as a fraction of a
                # hypothetical 1000-unit notional for ratio purposes.
                # This is a dimensionless ratio suitable for comparison
                # across variants.
                cost_perf[name] = round(
                    ab.total_token_cost_usd / abs(ab.total_return_pct),
                    6,
                )
            else:
                cost_perf[name] = float("inf")

        passed = len(ablation_results) >= 2
        return P6Report(
            results=ablation_results,
            debate_contribution_bps=round(debate_contrib, 2),
            multi_agent_contribution_bps=round(multi_contrib, 2),
            cost_performance_ratio=cost_perf,
            passed=passed,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def generate(
        backtest_results: dict[str, Any] | None = None,
        config: dict[str, Any] | None = None,
    ) -> P1P6ReportResult:
        """Generate a complete P1-P6 compliance report.

        Parameters
        ----------
        backtest_results:
            Dictionary of backtest output data.  Each P-level builder
            extracts its required keys (see individual builder docstrings).
            If ``None``, a skeleton report with all defaults is produced.
        config:
            Backtest configuration dictionary.  If ``None``, empty defaults
            are used.

        Returns
        -------
        P1P6ReportResult
            Fully populated compliance report.  Every P-level sub-report is
            guaranteed to be present and populated (P6 may be ``None`` if no
            ablation data was provided).
        """
        results = backtest_results or {}
        cfg = config or {}

        # Determine symbol and timeframe from results or config.
        symbol = results.get("symbol", cfg.get("symbol", ""))
        timeframe = results.get("timeframe", cfg.get("timeframe", ""))

        # Build each P-level.
        p1 = P1P6Report._build_p1(results, cfg)
        p2 = P1P6Report._build_p2(results, cfg)
        p3 = P1P6Report._build_p3(results, cfg)
        p4 = P1P6Report._build_p4(results, cfg)
        p5 = P1P6Report._build_p5(results, cfg)
        p6 = P1P6Report._build_p6(results, cfg)

        # Count passes.
        applicable = [p1, p2, p3, p4, p5]
        if p6 is not None:
            applicable.append(p6)
        n_passed = sum(1 for p in applicable if p.passed)
        all_passed = n_passed == len(applicable)

        # Summary line.
        summary_parts: list[str] = []
        labels = ["P1", "P2", "P3", "P4", "P5"]
        if p6 is not None:
            labels.append("P6")
        for label, p in zip(labels, applicable, strict=True):
            status = "PASS" if p.passed else "FAIL"
            summary_parts.append(f"{label}={status}")
        summary = (
            f"{symbol}/{timeframe} - "
            f"{n_passed}/{len(applicable)} passed "
            f"({' '.join(summary_parts)})"
        )

        return P1P6ReportResult(
            title=f"P1-P6 Compliance Report - {symbol} ({timeframe})",
            symbol=symbol,
            timeframe=timeframe,
            p1=p1,
            p2=p2,
            p3=p3,
            p4=p4,
            p5=p5,
            p6=p6,
            summary=summary,
            n_passed=n_passed,
            all_passed=all_passed,
        )


# ============================================================================
# Display
# ============================================================================


def _pass_fail(passed: bool) -> str:
    """Return a coloured pass/fail indicator for terminal output."""
    return "PASS" if passed else "FAIL"


def _fmt_pct(value: float) -> str:
    """Format a percentage value with sign and 2 decimal places."""
    return f"{value:+.2f}%"


def _fmt_pct_plain(value: float) -> str:
    """Format a percentage value without forced sign."""
    return f"{value:.2f}%"


def _fmt_bps(value: float) -> str:
    """Format a value in basis points."""
    return f"{value:+.2f} bps" if value != 0 else "0.00 bps"


def _fmt_date(dt: datetime) -> str:
    """Format a datetime as YYYY-MM-DD."""
    return dt.strftime("%Y-%m-%d")


def _indent(text: str, level: int = 0) -> str:
    """Indent every line of *text* by 2 spaces per *level*."""
    prefix = "  " * level
    return "\n".join(f"{prefix}{line}" for line in text.splitlines())


# ------------------------------------------------------------------
# Markdown Rendering
# ------------------------------------------------------------------


def _p1_to_markdown(p1: P1Report, level: int = 0) -> str:
    """Render P1 report as Markdown."""
    lines = [
        f"{'#' * (2 + level)} P1: Time Consistency [{_pass_fail(p1.passed)}]",
        "",
        f"- **Method:** {p1.method}",
        f"- **Folds:** {p1.n_splits}",
        f"- **Total span:** {p1.total_span_days} days"
        f" {'(>= 3 years -- OK)' if p1.spans_multiple_regimes else '(< 3 years -- warning)'}",
        f"- **Min train days:** {p1.min_train_days}",
        "",
    ]

    if p1.train_periods:
        lines.append("| Fold | Train | Val | Test |")
        lines.append("|------|-------|-----|------|")
        for i in range(len(p1.train_periods)):
            def _range_str(periods, idx):
                return (
                    f"{_fmt_date(periods[idx][0])} - {_fmt_date(periods[idx][1])}"
                )
            tr = _range_str(p1.train_periods, i)
            vl = _range_str(p1.val_periods, i)
            te = _range_str(p1.test_periods, i)
            lines.append(f"| {i} | {tr} | {vl} | {te} |")

        lines.append("")

    return "\n".join(lines)


def _p2_to_markdown(p2: P2Report, level: int = 0) -> str:
    """Render P2 report as Markdown."""
    lines = [
        f"{'#' * (2 + level)} P2: Point-in-Time Data [{_pass_fail(p2.passed)}]",
        "",
        f"- **as_of timestamps:** {'Yes' if p2.as_of_timestamps else 'No'}",
        f"- **Time-guard enforced:** {'Yes' if p2.time_guard_enforced else 'No'}",
        f"- **Max look-ahead bias:** {p2.max_look_ahead_bias_days:.2f} days",
        f"- **Data sources:** {', '.join(p2.data_sources) if p2.data_sources else 'N/A'}",
        f"- **Data points consumed:** {p2.n_data_points:,}",
        "",
    ]
    return "\n".join(lines)


def _p3_to_markdown(p3: P3Report, level: int = 0) -> str:
    """Render P3 report as Markdown."""
    lines = [
        f"{'#' * (2 + level)} P3: Execution Timing [{_pass_fail(p3.passed)}]",
        "",
        f"- **Signal -> Execution model:** ``{p3.signal_execution_model}``",
        f"- **Slippage model:** {p3.slippage_model or 'N/A'}",
        f"- **Avg slippage:** {p3.avg_slippage_bps:.2f} bps",
        f"- **Fill rate:** {p3.fill_rate * 100:.1f}%",
        "",
    ]
    return "\n".join(lines)


def _p4_to_markdown(p4: P4Report, level: int = 0) -> str:
    """Render P4 report as Markdown."""
    lines = [
        f"{'#' * (2 + level)} P4: Cost Realism [{_pass_fail(p4.passed)}]",
        "",
        "| Tier | Gross PnL | Net PnL | Sharpe | Max DD | Cost % |",
        "|------|-----------|---------|--------|--------|--------|",
    ]

    for label in ("0bps", "10bps", "25bps"):
        tier = p4.cost_tiers.get(label)
        if tier is None:
            continue
        lines.append(
            f"| {label} | {_fmt_pct(tier.gross_pnl_pct)} | "
            f"{_fmt_pct(tier.net_pnl_pct)} | {tier.sharpe:.2f} | "
            f"{_fmt_pct_plain(tier.max_dd_pct)} | "
            f"{_fmt_pct_plain(tier.total_cost_pct)} |"
        )

    lines.append("")
    lines.append("**Detailed cost breakdown:**")
    lines.append(f"- Commission: {p4.detailed_commission_bps:.2f} bps")
    lines.append(f"- Bid-ask spread: {p4.detailed_spread_bps:.2f} bps")
    lines.append(f"- Slippage: {p4.detailed_slippage_bps:.2f} bps")
    lines.append(f"- Gas fee: {p4.detailed_gas_bps:.2f} bps")
    lines.append("")
    lines.append(
        f"> Gross - Net gap at 10 bps tier: "
        f"**{_fmt_bps(p4.gross_minus_net_gap_bps)}** "
        f"(= alpha hallucination, Alpha Illusion S4.2)"
    )
    lines.append("")

    return "\n".join(lines)


def _p5_to_markdown(p5: P5Report, level: int = 0) -> str:
    """Render P5 report as Markdown."""
    lines = [
        f"{'#' * (2 + level)} P5: Benchmark Comparison "
        f"[{_pass_fail(p5.passed)}]",
        "",
    ]

    if p5.strategy:
        lines.append("### Strategy (net of costs)")
        lines.append(
            f"- Return: {_fmt_pct(p5.strategy.total_return_pct)}  |  "
            f"Annualised: {_fmt_pct(p5.strategy.annualised_return_pct)}"
        )
        lines.append(
            f"- Sharpe: {p5.strategy.sharpe:.2f}  |  "
            f"Max DD: {_fmt_pct_plain(p5.strategy.max_dd_pct)}  |  "
            f"Vol: {_fmt_pct_plain(p5.strategy.volatility_pct)}"
        )
        lines.append("")

    if p5.benchmarks:
        lines.append("### Benchmarks")
        lines.append(
            "| Benchmark | Return | Ann. Return | Sharpe | Max DD | Vol |"
        )
        lines.append(
            "|-----------|--------|-------------|--------|--------|-----|"
        )
        for name, bench in p5.benchmarks.items():
            lines.append(
                f"| {name} | {_fmt_pct(bench.total_return_pct)} | "
                f"{_fmt_pct(bench.annualised_return_pct)} | "
                f"{bench.sharpe:.2f} | "
                f"{_fmt_pct_plain(bench.max_dd_pct)} | "
                f"{_fmt_pct_plain(bench.volatility_pct)} |"
            )
        lines.append("")

    lines.append(
        f"> **Outperformance:** {_fmt_bps(p5.outperformance_bps)} vs best benchmark"
    )
    if p5.beats_all_benchmarks:
        lines.append("> **Result:** Strategy beats every benchmark.")
    else:
        lines.append(
            "> **Result:** Strategy does NOT beat all benchmarks."
        )
    lines.append("")

    return "\n".join(lines)


def _p6_to_markdown(p6: P6Report, level: int = 0) -> str:
    """Render P6 ablation report as Markdown."""
    if p6 is None:
        return ""

    lines = [
        f"{'#' * (2 + level)} P6: Ablation "
        f"[{_pass_fail(p6.passed)}]",
        "",
        "| Variant | Return | Sharpe | Max DD | Trades | LLM Calls | LLM Cost $ |",
        "|---------|--------|--------|--------|--------|-----------|------------|",
    ]

    for name, ab in p6.results.items():
        lines.append(
            f"| {name} | {_fmt_pct(ab.total_return_pct)} | "
            f"{ab.sharpe:.2f} | {_fmt_pct_plain(ab.max_dd_pct)} | "
            f"{ab.n_trades} | {ab.total_llm_calls} | "
            f"${ab.total_token_cost_usd:.2f} |"
        )

    lines.append("")
    lines.append("**Marginal contributions:**")
    lines.append(
        f"- Debate mechanism: {_fmt_bps(p6.debate_contribution_bps)}"
    )
    lines.append(
        f"- Full multi-agent vs single: "
        f"{_fmt_bps(p6.multi_agent_contribution_bps)}"
    )
    lines.append("")
    lines.append("**Cost-performance ratio (LLM cost / |Return|):**")
    for name, ratio in p6.cost_performance_ratio.items():
        ratio_str = f"{ratio:.6f}" if ratio != float("inf") else "Inf (zero return)"
        lines.append(f"- {name}: {ratio_str}")
    lines.append("")

    return "\n".join(lines)


# ============================================================================
# to_markdown  /  to_html  on P1P6ReportResult
# ============================================================================


def _to_markdown_inner(report: P1P6ReportResult) -> str:
    """Render the full P1-P6 report as Markdown."""
    sections: list[str] = []

    # Title
    sections.append(f"# {report.title}")
    sections.append("")
    sections.append(
        f"**Generated:** {report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"**Symbol:** {report.symbol}  |  "
        f"**Timeframe:** {report.timeframe}"
    )
    sections.append("")

    # Summary
    sections.append("## Summary")
    sections.append("")
    sections.append(report.summary)
    sections.append("")
    sections.append(
        f"> **{report.n_passed}/6** protocol levels passed  |  "
        f"{'All PASS' if report.all_passed else 'Some FAILs -- see below'}"
    )
    sections.append("")

    # Per-P sections
    sections.append(_p1_to_markdown(report.p1))
    sections.append(_p2_to_markdown(report.p2))
    sections.append(_p3_to_markdown(report.p3))
    sections.append(_p4_to_markdown(report.p4))
    sections.append(_p5_to_markdown(report.p5))
    if report.p6 is not None:
        sections.append(_p6_to_markdown(report.p6))

    return "\n".join(sections)


# ============================================================================
# HTML Rendering
# ============================================================================


def _html_escape(text: str) -> str:
    """HTML-escape a plain-text string."""
    return html.escape(text, quote=True)


def _html_pass_fail(passed: bool) -> str:
    """Return an HTML badge for pass/fail."""
    badge = (
        '<span style="background:#27ae60;color:#fff;padding:2px 8px;'
        'border-radius:3px;font-size:0.85em">PASS</span>'
        if passed
        else '<span style="background:#e74c3c;color:#fff;padding:2px 8px;'
        'border-radius:3px;font-size:0.85em">FAIL</span>'
    )
    return badge


def _p1_to_html(p1: P1Report, level: int = 3) -> str:
    """Render P1 report as HTML."""
    h = level
    lines = [
        f"<h{h}>P1: Time Consistency {_html_pass_fail(p1.passed)}</h{h}>",
        "<table>",
        f"<tr><td>Method</td><td>{_html_escape(p1.method)}</td></tr>",
        f"<tr><td>Folds</td><td>{p1.n_splits}</td></tr>",
        f"<tr><td>Total span</td><td>{p1.total_span_days} days"
        f"{' (>= 3 years &#10004;)' if p1.spans_multiple_regimes else ' (< 3 years &#9888;)'}"
        f"</td></tr>",
        f"<tr><td>Min train days</td><td>{p1.min_train_days}</td></tr>",
        "</table>",
    ]

    if p1.train_periods:
        lines.append("<table>")
        lines.append(
            "<tr><th>Fold</th><th>Train</th><th>Val</th><th>Test</th></tr>"
        )
        for i in range(len(p1.train_periods)):
            def _html_range(periods, idx):
                return (
                    f"{_fmt_date(periods[idx][0])} &ndash; "
                    f"{_fmt_date(periods[idx][1])}"
                )
            tr = _html_range(p1.train_periods, i)
            vl = _html_range(p1.val_periods, i)
            te = _html_range(p1.test_periods, i)
            lines.append(
                f"<tr><td>{i}</td><td>{tr}</td><td>{vl}</td><td>{te}</td></tr>"
            )
        lines.append("</table>")

    return "\n".join(lines)


def _p2_to_html(p2: P2Report, level: int = 3) -> str:
    h = level
    return (
        f"<h{h}>P2: Point-in-Time Data {_html_pass_fail(p2.passed)}</h{h}>"
        "<table>"
        f"<tr><td>as_of timestamps</td><td>{'Yes' if p2.as_of_timestamps else 'No'}</td></tr>"
        f"<tr><td>Time-guard enforced</td><td>{'Yes' if p2.time_guard_enforced else 'No'}</td></tr>"
        f"<tr><td>Max look-ahead bias</td><td>{p2.max_look_ahead_bias_days:.2f} days</td></tr>"
        f"<tr><td>Data sources</td><td>{', '.join(p2.data_sources) if p2.data_sources else 'N/A'}</td></tr>"
        f"<tr><td>Data points consumed</td><td>{p2.n_data_points:,}</td></tr>"
        "</table>"
    )


def _p3_to_html(p3: P3Report, level: int = 3) -> str:
    h = level
    return (
        f"<h{h}>P3: Execution Timing {_html_pass_fail(p3.passed)}</h{h}>"
        "<table>"
        f"<tr><td>Signal &rarr; Execution model</td>"
        f"<td><code>{_html_escape(p3.signal_execution_model)}</code></td></tr>"
        f"<tr><td>Slippage model</td><td>{_html_escape(p3.slippage_model) or 'N/A'}</td></tr>"
        f"<tr><td>Avg slippage</td><td>{p3.avg_slippage_bps:.2f} bps</td></tr>"
        f"<tr><td>Fill rate</td><td>{p3.fill_rate * 100:.1f}%</td></tr>"
        "</table>"
    )


def _p4_to_html(p4: P4Report, level: int = 3) -> str:
    h = level
    lines = [
        f"<h{h}>P4: Cost Realism {_html_pass_fail(p4.passed)}</h{h}>",
        "<table>",
        "<tr><th>Tier</th><th>Gross PnL</th><th>Net PnL</th>"
        "<th>Sharpe</th><th>Max DD</th><th>Cost %</th></tr>",
    ]
    for label in ("0bps", "10bps", "25bps"):
        tier = p4.cost_tiers.get(label)
        if tier is None:
            continue
        lines.append(
            f"<tr><td>{label}</td>"
            f"<td>{_fmt_pct(tier.gross_pnl_pct)}</td>"
            f"<td>{_fmt_pct(tier.net_pnl_pct)}</td>"
            f"<td>{tier.sharpe:.2f}</td>"
            f"<td>{_fmt_pct_plain(tier.max_dd_pct)}</td>"
            f"<td>{_fmt_pct_plain(tier.total_cost_pct)}</td></tr>"
        )
    lines.append("</table>")

    lines.append("<p><strong>Detailed cost breakdown:</strong><br>")
    lines.append(f"Commission: {p4.detailed_commission_bps:.2f} bps<br>")
    lines.append(f"Bid-ask spread: {p4.detailed_spread_bps:.2f} bps<br>")
    lines.append(f"Slippage: {p4.detailed_slippage_bps:.2f} bps<br>")
    lines.append(f"Gas fee: {p4.detailed_gas_bps:.2f} bps</p>")
    lines.append(
        f"<blockquote>Gross &minus; Net gap at 10 bps tier: "
        f"<strong>{_fmt_bps(p4.gross_minus_net_gap_bps)}</strong> "
        f"(= alpha hallucination, Alpha Illusion &sect;4.2)</blockquote>"
    )

    return "\n".join(lines)


def _p5_to_html(p5: P5Report, level: int = 3) -> str:
    h = level
    lines = [
        f"<h{h}>P5: Benchmark Comparison {_html_pass_fail(p5.passed)}</h{h}>",
    ]

    if p5.strategy:
        s = p5.strategy
        lines.append("<h4>Strategy (net of costs)</h4>")
        lines.append(
            f"<p>Return: {_fmt_pct(s.total_return_pct)}  |  "
            f"Annualised: {_fmt_pct(s.annualised_return_pct)}<br>"
            f"Sharpe: {s.sharpe:.2f}  |  "
            f"Max DD: {_fmt_pct_plain(s.max_dd_pct)}  |  "
            f"Vol: {_fmt_pct_plain(s.volatility_pct)}</p>"
        )

    if p5.benchmarks:
        lines.append("<h4>Benchmarks</h4>")
        lines.append(
            "<table><tr><th>Benchmark</th><th>Return</th>"
            "<th>Ann. Return</th><th>Sharpe</th><th>Max DD</th>"
            "<th>Vol</th></tr>"
        )
        for name, b in p5.benchmarks.items():
            lines.append(
                f"<tr><td>{_html_escape(name)}</td>"
                f"<td>{_fmt_pct(b.total_return_pct)}</td>"
                f"<td>{_fmt_pct(b.annualised_return_pct)}</td>"
                f"<td>{b.sharpe:.2f}</td>"
                f"<td>{_fmt_pct_plain(b.max_dd_pct)}</td>"
                f"<td>{_fmt_pct_plain(b.volatility_pct)}</td></tr>"
            )
        lines.append("</table>")

    lines.append(
        f"<blockquote>Outperformance: {_fmt_bps(p5.outperformance_bps)} "
        f"vs best benchmark<br>"
    )
    if p5.beats_all_benchmarks:
        lines.append("<strong>Result:</strong> Strategy beats every benchmark.")
    else:
        lines.append(
            "<strong>Result:</strong> Strategy does NOT beat all "
            "benchmarks."
        )
    lines.append("</blockquote>")

    return "\n".join(lines)


def _p6_to_html(p6: P6Report, level: int = 3) -> str:
    if p6 is None:
        return ""

    h = level
    lines = [
        f"<h{h}>P6: Ablation {_html_pass_fail(p6.passed)}</h{h}>",
        "<table>",
        "<tr><th>Variant</th><th>Return</th><th>Sharpe</th><th>Max DD</th>"
        "<th>Trades</th><th>LLM Calls</th><th>LLM Cost $</th></tr>",
    ]

    for name, ab in p6.results.items():
        lines.append(
            f"<tr><td>{_html_escape(name)}</td>"
            f"<td>{_fmt_pct(ab.total_return_pct)}</td>"
            f"<td>{ab.sharpe:.2f}</td>"
            f"<td>{_fmt_pct_plain(ab.max_dd_pct)}</td>"
            f"<td>{ab.n_trades}</td>"
            f"<td>{ab.total_llm_calls}</td>"
            f"<td>${ab.total_token_cost_usd:.2f}</td></tr>"
        )
    lines.append("</table>")

    lines.append("<p><strong>Marginal contributions:</strong><br>")
    lines.append(f"Debate mechanism: {_fmt_bps(p6.debate_contribution_bps)}<br>")
    lines.append(
        f"Full multi-agent vs single: "
        f"{_fmt_bps(p6.multi_agent_contribution_bps)}</p>"
    )

    lines.append("<p><strong>Cost-performance ratio (LLM cost / |Return|):</strong><br>")
    for name, ratio in p6.cost_performance_ratio.items():
        ratio_str = f"{ratio:.6f}" if ratio != float("inf") else "Inf (zero return)"
        lines.append(f"{name}: {ratio_str}<br>")
    lines.append("</p>")

    return "\n".join(lines)


def _to_html_inner(report: P1P6ReportResult) -> str:
    """Render the full P1-P6 report as a standalone HTML page."""
    css = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto,
                     'Helvetica Neue', sans-serif;
        max-width: 960px; margin: 2em auto; padding: 0 1em;
        color: #222; line-height: 1.6;
    }
    h1 { border-bottom: 2px solid #2c3e50; padding-bottom: 0.3em; }
    h2 { color: #2c3e50; margin-top: 1.5em; }
    h3 { color: #34495e; margin-top: 1.2em; }
    table {
        border-collapse: collapse; width: 100%; margin: 0.5em 0 1em 0;
    }
    th, td {
        border: 1px solid #bdc3c7; padding: 6px 10px; text-align: left;
    }
    th { background: #f5f6fa; font-weight: 600; }
    blockquote {
        border-left: 4px solid #3498db;
        margin: 0.5em 0; padding: 0.5em 1em;
        background: #f8f9fa; border-radius: 3px;
    }
    code { background: #ecf0f1; padding: 1px 4px; border-radius: 3px; }
    .summary {
        font-size: 1.1em; padding: 0.8em; background: #f0f4f8;
        border-radius: 4px; border-left: 4px solid #2980b9;
    }
    """

    sections = [
        "<!DOCTYPE html>",
        '<html lang="en">',
        "<head>",
        '<meta charset="utf-8">',
        f"<title>{_html_escape(report.title)}</title>",
        f"<style>{css}</style>",
        "</head>",
        "<body>",
        f"<h1>{_html_escape(report.title)}</h1>",
        "<p>",
        f"<strong>Generated:</strong> "
        f"{report.timestamp.strftime('%Y-%m-%d %H:%M UTC')}  |  "
        f"<strong>Symbol:</strong> {_html_escape(report.symbol)}  |  "
        f"<strong>Timeframe:</strong> {_html_escape(report.timeframe)}",
        "</p>",
        "<h2>Summary</h2>",
        f'<p class="summary">{_html_escape(report.summary)}</p>',
        "<p>"
        f"<strong>{report.n_passed}/6</strong> protocol levels passed  |  "
        f"{'<strong>All PASS</strong>' if report.all_passed else 'Some FAILs'}"
        "</p>",
        # Per-P sections
        _p1_to_html(report.p1),
        _p2_to_html(report.p2),
        _p3_to_html(report.p3),
        _p4_to_html(report.p4),
        _p5_to_html(report.p5),
    ]

    if report.p6 is not None:
        sections.append(_p6_to_html(report.p6))

    sections.extend(["</body>", "</html>"])

    return "\n".join(sections)


# ============================================================================
# Monkey-patch display methods onto P1P6ReportResult
# ============================================================================


def to_markdown(self: P1P6ReportResult) -> str:
    """Render this report as a formatted Markdown string.

    The output includes a title header, summary line, and one section per
    P-level.  Each section starts with a PASS/FAIL indicator.

    Returns
    -------
    str
        Markdown-formatted report.
    """
    return _to_markdown_inner(self)


def to_html(self: P1P6ReportResult) -> str:
    """Render this report as a standalone HTML page.

    The page is self-contained with embedded CSS and no external
    dependencies.  Suitable for saving to a ``.html`` file and opening in a
    browser.

    Returns
    -------
    str
        HTML document as a single string.
    """
    return _to_html_inner(self)


# Attach methods to the result class.
P1P6ReportResult.to_markdown = to_markdown
P1P6ReportResult.to_html = to_html
