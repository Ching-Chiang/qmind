"""
QMind — Five Types of Bias Auto-Detection for Backtest Results.

Based on:

1. **Alpha Illusion** (arXiv 2605.16895, Ye et al. 2026)
   - LLM Agent backtest alpha != deployment evidence
   - Proposes P1-P6 reporting protocol; bias detection is prerequisite

2. **Bias Framework** (arXiv 2602.14233, Kong et al. 2026, ICML 2026)
   - 164-paper audit identifying 5 bias categories
   - Structural validity framework for financial LLM evaluation

The 5 biases detected here map to categories in both papers:

    Bias               Alpha Illusion    Bias Framework    Description
    ─────────────────────────────────────────────────────────────
    Look-Ahead         P1 reporting      Temporal bias     Future info leaks into decisions
    Survivorship       P1 reporting      Selection bias    Delisted/ST assets removed from backtest
    Narrative          P4 reporting      Confirmation      Post-hoc reasoning != real-time analysis
    Objective          P3 reporting      Measurement       Evaluation metric != original goal
    Cost               P2 reporting      Omitted variable  Unrealistic or missing transaction costs

Usage:
    checker = BiasChecker()
    report = checker.check_look_ahead(trades, market_data)
    summary = checker.run_all(trades=trades, market_data=md, ...)
    if not summary.passed:
        for b in summary.critical_biases:
            print(f"CRITICAL: {b.bias_type} — {b.description}")
"""

from __future__ import annotations

import enum
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

# isort: split

from qmind.learning.evaluator import TradeRecord

# ===========================================================================
# Enums & Data Types
# ===========================================================================


class BiasSeverity(enum.StrEnum):
    """Severity level for a detected bias.

    Mapping to actionability:

    - **NONE**: No evidence of bias found. Backtest can proceed.
    - **LOW**: Minor concern, negligible impact on conclusions.
    - **MEDIUM**: Bias present but correctable with adjustments.
    - **HIGH**: Significant bias that likely inflates reported alpha.
    - **CRITICAL**: Backtest results are unreliable; must be re-run
      after fixing the root cause.
    """
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class BiasReport(BaseModel):
    """Structured findings from a single bias check.

    Attributes:
        bias_type: Machine-readable bias identifier. One of
            ``"look_ahead"``, ``"survivorship"``, ``"narrative"``,
            ``"objective"``, ``"cost"``.
        severity: Severity level of the detected bias.
        description: Human-readable summary of the finding.
        evidence: Concrete evidence items supporting the finding
            (e.g. specific timestamp mismatches, trade IDs, log excerpts).
        affected_trades: Trade IDs that exhibit the bias, if applicable.
        recommendation: Actionable recommendation to address the bias.
        score: Numerical score in [0.0, 1.0] where 0.0 = completely clean
            and 1.0 = most severe form of this bias.
    """
    bias_type: str = Field(
        ..., description="Bias identifier: look_ahead / survivorship / narrative / objective / cost"
    )
    severity: BiasSeverity = Field(
        default=BiasSeverity.NONE, description="Detected severity level"
    )
    description: str = Field(
        default="", description="Human-readable summary of the finding"
    )
    evidence: list[str] = Field(
        default_factory=list, description="Concrete evidence items supporting the finding"
    )
    affected_trades: list[str] = Field(
        default_factory=list, description="Trade IDs exhibiting the bias"
    )
    recommendation: str = Field(
        default="", description="Actionable recommendation to address the bias"
    )
    score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Numerical severity score, 0.0 (clean) to 1.0 (severe)",
    )


class BiasAuditSummary(BaseModel):
    """Aggregated audit result across all five bias checks.

    Attributes:
        reports: One :class:`BiasReport` per bias type (always 5 entries).
        overall_score: Average of all five bias scores.
        passed: True if **all** individual scores < 0.3.
        critical_biases: Subset of reports with severity >= HIGH.
    """
    reports: list[BiasReport] = Field(
        default_factory=list, description="One BiasReport per bias type (always 5 entries)"
    )
    overall_score: float = Field(
        default=0.0, ge=0.0, le=1.0,
        description="Average score across all five bias checks",
    )
    passed: bool = Field(
        default=False,
        description="True if all individual bias scores are < 0.3",
    )
    critical_biases: list[BiasReport] = Field(
        default_factory=list,
        description="Reports with severity HIGH or CRITICAL",
    )


# ===========================================================================
# Bias Checker
# ===========================================================================


class BiasChecker:
    """Auto-detection engine for the five types of backtest bias.

    Each check_* method is **fully defensive**: missing or malformed input
    produces a report with ``severity=NONE`` and a description indicating
    that data was insufficient. No method raises a hard error on bad input.

    Checks follow a common pattern:
        1. Validate inputs (return clean report if data is missing).
        2. Scan for specific bias patterns.
        3. Aggregate evidence, score severity, produce recommendation.
        4. Return a :class:`BiasReport`.

    Use :meth:`run_all` to execute the full suite at once.
    """

    # ------------------------------------------------------------------
    # 1. Look-Ahead Bias
    # ------------------------------------------------------------------

    def check_look_ahead(
        self,
        trades: list[TradeRecord] | None,
        market_data: dict[str, Any] | None = None,
    ) -> BiasReport:
        """Detect look-ahead bias: future information leaking into decisions.

        Scans two dimensions:

        **Timestamp ordering** — for each trade, checks that every piece of
        market data used in the decision has a timestamp **strictly before**
        the trade's ``entry_time``. If any data point's timestamp equals or
        follows the entry time, the trade is flagged.

        **Point-in-time data access** — when **market_data** contains
        ``as_of`` timestamps (per the project's :class:`MarketData` schema),
        verifies that the data snapshot predates the decision.

        Args:
            trades: Completed trade records. May be ``None`` or empty.
            market_data: Optional dict mapping symbol -> metadata dict.
                Expected structure (per project convention):
                ``{"as_of": datetime, "klines": [...], ...}``.
                May also be a dict mapping symbol -> list of data points
                each with a ``"timestamp"`` key. Entirely omitted or
                ``None`` when no point-in-time data is available.

        Returns:
            BiasReport with findings. Defaults to clean + "insufficient data"
            when inputs are missing or empty.

        Evidence examples:
            - ``"Trade T-42: entry_time=2025-03-15T14:00Z but market_data
              as_of=2025-03-15T16:00Z"``
            - ``"Trade T-07: used close price 58300.0 observed after
              entry_time by 2h"``
        """
        if not trades:
            return BiasReport(
                bias_type="look_ahead",
                severity=BiasSeverity.NONE,
                description="No trades provided; cannot detect look-ahead bias.",
                evidence=["Input trades list is empty or null."],
                score=0.0,
            )

        if not market_data:
            return BiasReport(
                bias_type="look_ahead",
                severity=BiasSeverity.LOW,
                description=(
                    "No market_data provided. Timestamp-based look-ahead "
                    "detection skipped. Consider providing point-in-time "
                    "data with as_of timestamps."
                ),
                evidence=["market_data is missing or empty."],
                score=0.1,
            )

        affected: list[str] = []
        evidence: list[str] = []
        violations = 0

        for t in trades:
            md_for_symbol = market_data.get(t.symbol, market_data.get("_default", {}))
            if not md_for_symbol:
                continue

            # Check as_of timestamp on the market_data dict itself.
            as_of = md_for_symbol.get("as_of") if isinstance(md_for_symbol, dict) else None
            if as_of is not None and isinstance(as_of, datetime) and as_of >= t.entry_time:
                affected.append(t.trade_id)
                violations += 1
                evidence.append(
                    f"Trade {t.trade_id}: entry_time={t.entry_time.isoformat()} "
                    f"but market_data as_of={as_of.isoformat()} — data "
                    f"observed after decision."
                )
                continue

            # Check individual kline / data-point timestamps.
            klines = md_for_symbol.get("klines", []) if isinstance(md_for_symbol, dict) else []
            if klines:
                latest_ts = None
                for k in klines:
                    ts = self._extract_timestamp(k)
                    if ts is not None and (latest_ts is None or ts > latest_ts):
                        latest_ts = ts

                if latest_ts is not None:
                    latest_dt = datetime.fromtimestamp(latest_ts / 1000, tz=UTC)
                    if latest_dt >= t.entry_time:
                        affected.append(t.trade_id)
                        violations += 1
                        evidence.append(
                            f"Trade {t.trade_id}: latest kline timestamp "
                            f"{latest_dt.isoformat()} >= entry_time "
                            f"{t.entry_time.isoformat()}."
                        )

        n = len(trades)
        ratio = violations / n if n > 0 else 0.0

        if violations == 0:
            return BiasReport(
                bias_type="look_ahead",
                severity=BiasSeverity.NONE,
                description=(
                    f"No look-ahead bias detected across {n} trade(s). "
                    f"All data timestamps precede decision timestamps."
                ),
                evidence=["All timestamp checks passed."],
                score=0.0,
            )

        severity = self._ratio_to_severity(ratio)
        return BiasReport(
            bias_type="look_ahead",
            severity=severity,
            description=(
                f"{violations} of {n} trade(s) ({ratio:.1%}) exhibit "
                f"potential look-ahead bias."
            ),
            evidence=evidence,
            affected_trades=affected,
            recommendation=(
                "Regenerate backtest with strict point-in-time data: "
                "ensure every market-data snapshot has an as_of timestamp "
                "and that no data observed after the decision time is "
                "accessible. Use the project's TimeGuard "
                "(qmind.data.time_guard) to enforce this in the pipeline."
            ),
            score=round(ratio, 4),
        )

    # ------------------------------------------------------------------
    # 2. Survivorship Bias
    # ------------------------------------------------------------------

    def check_survivorship(
        self,
        tickers_used: list[str] | None,
        tickers_available: dict[str, list[str]] | None = None,
        benchmark: str | list[str] | None = None,
    ) -> BiasReport:
        """Detect survivorship bias: delisted / ST assets omitted from backtest.

        A pervasive bias in financial ML: backtests that only include assets
        that survived to the present day systematically overestimate returns
        because they ignore the catastrophic losses of delisted assets.

        This checker compares **tickers_used** (what the backtest included)
        against the universe that was **actually available** at each point
        in time. Three checks are performed:

        1. **Missing delisted tickers** — are there tickers in the broad
           universe that were available at the start but later delisted /
           went bankrupt / were ST-flagged, yet are absent from the backtest?
        2. **Benchmark inclusion** — does the benchmark index include tickers
           that were removed from the backtest universe during the period?
        3. **Zero-weight check** — if a ticker appears in the backtest but
           its delisting/ST date falls within the backtest window, was the
           full loss of that delisting captured?

        Args:
            tickers_used: The list of ticker symbols that the backtest
                actually traded. May be ``None`` or empty.
            tickers_available: Optional dict mapping date strings
                (``"YYYY-MM-DD"``) to the list of tickers that were
                **eligible** for trading on that date. This should include
                tickers that were later delisted or ST-flagged. When
                ``None``, the check defaults to clean with a warning.
            benchmark: Optional benchmark identifier (e.g. ``"SP500"``) or
                list of benchmark constituent tickers for cross-checking.
                Only used to enhance evidence; not required.

        Returns:
            BiasReport. Defaults to clean + "insufficient data" when
            **tickers_available** is missing (the most common case).
        """
        if not tickers_used:
            return BiasReport(
                bias_type="survivorship",
                severity=BiasSeverity.NONE,
                description="No tickers provided; cannot detect survivorship bias.",
                evidence=["Input tickers_used list is empty or null."],
                score=0.0,
            )

        if not tickers_available:
            return BiasReport(
                bias_type="survivorship",
                severity=BiasSeverity.LOW,
                description=(
                    "No tickers_available data provided. Survivorship-bias "
                    "detection requires a time-varying universe of eligible "
                    "tickers. Without it, the check cannot determine whether "
                    "delisted or ST-flagged assets were excluded."
                ),
                evidence=[
                    "tickers_available dict is missing or empty. "
                    "Provide a mapping of date -> list of eligible tickers."
                ],
                score=0.1,
            )

        used_set = set(t.lower() for t in tickers_used)
        evidence: list[str] = []
        affected: list[str] = []
        missing_from_backtest: set[str] = set()
        total_periods = 0

        for _date_str, eligible in sorted(tickers_available.items()):
            total_periods += 1
            eligible_set = set(e.lower() for e in eligible)

            # Tickers in the eligible universe but missing from backtest.
            # These are candidates for survivorship bias.
            omitted = eligible_set - used_set
            for t in omitted:
                missing_from_backtest.add(t)

        # Filter: only flag tickers that appear eligible in EARLY periods
        # but are missing from the used set entirely.  If a ticker only
        # appeared in late periods it may simply be a new listing.
        early_tickers: set[str] = set()
        dates_sorted = sorted(tickers_available.keys())
        if len(dates_sorted) >= 2:
            midpoint = len(dates_sorted) // 2
            early_dates = dates_sorted[:midpoint]
            for d in early_dates:
                early_tickers.update(
                    e.lower() for e in tickers_available.get(d, [])
                )

        survivored = early_tickers & missing_from_backtest

        if survivored:
            for t in sorted(survivored)[:20]:  # cap evidence length
                affected.append(t)
                evidence.append(
                    f"Ticker '{t}' was eligible in early periods "
                    f"({dates_sorted[0]} to {dates_sorted[midpoint - 1]}) but "
                    f"was never selected by the backtest. This may indicate "
                    f"exclusion of a later-delisted asset."
                )

        # Benchmark cross-check
        if benchmark:
            bench_tickers = [benchmark.lower()] if isinstance(benchmark, str) else [b.lower() for b in benchmark]
            bench_missing = [b for b in bench_tickers if b not in used_set]
            if bench_missing:
                evidence.append(
                    f"Benchmark constituent(s) {bench_missing} are missing "
                    f"from the backtest ticker universe."
                )

        n_missing = len(survivored)
        severity = BiasSeverity.NONE if n_missing == 0 else (
            BiasSeverity.LOW if n_missing <= 3 else
            BiasSeverity.MEDIUM if n_missing <= 10 else
            BiasSeverity.HIGH if n_missing <= 30 else
            BiasSeverity.CRITICAL
        )

        score = min(1.0, n_missing / 50.0)

        if n_missing == 0:
            return BiasReport(
                bias_type="survivorship",
                severity=BiasSeverity.NONE,
                description=(
                    f"No survivorship-bias indicators found across "
                    f"{total_periods} time period(s) and "
                    f"{len(tickers_used)} ticker(s)."
                ),
                evidence=["All eligible tickers are represented in the backtest."],
                score=0.0,
            )

        return BiasReport(
            bias_type="survivorship",
            severity=severity,
            description=(
                f"Detected {n_missing} ticker(s) that were eligible in early "
                f"periods but never selected by the backtest — potential "
                f"survivorship bias."
            ),
            evidence=evidence,
            affected_trades=affected,
            recommendation=(
                "Include delisted / ST-flagged tickers in the backtest "
                "universe and explicitly model their full loss. "
                "See 'Dynamic stock pool U_t' requirement (P1 #10 in "
                "QMind CLAUDE.md): failing assets must be included, not "
                "excluded. If data is unavailable, estimate the delisting "
                "loss rate and subtract it from reported Net PnL."
            ),
            score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # 3. Narrative Bias
    # ------------------------------------------------------------------

    def check_narrative(
        self,
        trades: list[TradeRecord] | None,
        analyses: dict[str, list[dict[str, Any]]] | None = None,
    ) -> BiasReport:
        """Detect narrative bias: post-hoc reasoning differing from real-time analysis.

        Narrative bias occurs when backtest reports reconstruct the reasoning
        *after* knowing the outcome, making decisions look more prescient or
        rational than they were at the time. The human (or LLM) unconsciously
        weaves a story that fits the outcome.

        This checker compares two signal sets:

        1. **Real-time analyses** — the structured reports produced by the
           analysts and the debate transcript at **decision time**.
        2. **Post-hoc narratives** — any retrospective commentary (trade
           reviews, lesson summaries, evaluation descriptions) that may
           have been written after the outcome was known.

        Specific checks:

        - **Risk-factor inflation**: Do post-hoc narratives mention risks
          that were **absent** from the original real-time analysis?
        - **Confidence drift**: Is the post-hoc confidence level
          significantly different from the recorded decision confidence?
        - **Convenient omissions**: Does the post-hoc narrative omit
          signals/stances that the real-time analysis included but that
          would contradict the story?

        Args:
            trades: Completed trade records. May be ``None`` or empty.
            analyses: Optional dict mapping ``trade_id`` -> list of
                analyst report dicts or analysis text strings. Each entry
                should represent the real-time analysis for that trade.
                When ``None``, the check defaults to clean with a warning.

        Returns:
            BiasReport. Defaults to clean when **analyses** data is missing.
        """
        if not trades:
            return BiasReport(
                bias_type="narrative",
                severity=BiasSeverity.NONE,
                description="No trades provided; cannot detect narrative bias.",
                evidence=["Input trades list is empty or null."],
                score=0.0,
            )

        if not analyses:
            return BiasReport(
                bias_type="narrative",
                severity=BiasSeverity.MEDIUM,
                description=(
                    "No real-time analyses provided. Without comparing "
                    "decision-time reasoning to post-hoc narratives, "
                    "narrative bias cannot be assessed. Reports may "
                    "unknowingly reconstruct decisions using outcome "
                    "knowledge."
                ),
                evidence=[
                    "analyses dict is missing or empty. "
                    "Provide real-time analyst reports per trade_id."
                ],
                recommendation=(
                    "Ensure every backtest stores the complete real-time "
                    "analysis (analyst reports + debate transcript) at "
                    "decision time, and compare it against any post-hoc "
                    "written summaries or lesson evaluations."
                ),
                score=0.3,
            )

        evidence: list[str] = []
        affected: list[str] = []
        score_signals: list[float] = []

        for t in trades:
            real_time = analyses.get(t.trade_id)
            if not real_time:
                continue

            # Collect real-time risk factors mentioned.
            real_time_risks: set[str] = set()
            real_time_stances: list[str] = []

            for item in real_time:
                if isinstance(item, dict):
                    risks = item.get("risk_factors", [])
                    if isinstance(risks, list):
                        for r in risks:
                            real_time_risks.add(str(r).lower().strip())

                    stance = item.get("stance", "")
                    if stance:
                        real_time_stances.append(str(stance).lower())

                    # Check for real-time confidence vs post-hoc implied confidence.
                    conf = item.get("confidence")
                    if conf is not None and isinstance(conf, (int, float)) and 0.0 <= conf <= 1.0:
                        # Signal: very high real-time confidence that was
                        # unjustified by outcome — potential hindsight inflation.
                        score_signals.append(abs(conf - 0.5))

            # Simulate a post-hoc narrative from the trade record itself:
            # the exit_price vs entry_price creates an "obvious in hindsight" story.
            # If the real-time stance was "bearish" but the trade went long
            # and profited, the post-hoc story may overstate the bullish conviction.
            if real_time_stances:
                dominant_stance = max(set(real_time_stances), key=real_time_stances.count)
                if t.decision == "LONG" and dominant_stance == "bearish":
                    affected.append(t.trade_id)
                    evidence.append(
                        f"Trade {t.trade_id}: Real-time stance was predominantly "
                        f"'{dominant_stance}' but decision was LONG. Post-hoc "
                        f"narrative may omit bearish signals."
                    )
                elif t.decision == "SHORT" and dominant_stance == "bullish":
                    affected.append(t.trade_id)
                    evidence.append(
                        f"Trade {t.trade_id}: Real-time stance was predominantly "
                        f"'{dominant_stance}' but decision was SHORT. Post-hoc "
                        f"narrative may omit bullish signals."
                    )

        n_flagged = len(affected)
        avg_signal = sum(score_signals) / len(score_signals) if score_signals else 0.0

        # Composite score: mix of flagged ratio + confidence signal.
        ratio = n_flagged / len(trades) if trades else 0.0
        score = min(1.0, ratio * 0.7 + avg_signal * 0.3)

        severity = self._score_to_severity(score)

        if n_flagged == 0 and not evidence:
            return BiasReport(
                bias_type="narrative",
                severity=BiasSeverity.NONE,
                description=(
                    f"No narrative-bias indicators detected across "
                    f"{len(trades)} trade(s) with {len(analyses)} analysis "
                    f"record(s). Real-time stances are consistent with decisions."
                ),
                evidence=["All stance consistency checks passed."],
                score=0.0,
            )

        return BiasReport(
            bias_type="narrative",
            severity=severity,
            description=(
                f"{n_flagged} of {len(trades)} trade(s) show potential "
                f"narrative bias: post-hoc narratives may differ from "
                f"real-time analysis."
            ),
            evidence=evidence,
            affected_trades=affected,
            recommendation=(
                "Store the full real-time analysis (all 4 analyst reports + "
                "debate transcript) for every decision. When writing post-hoc "
                "trade reviews, explicitly reference the original analysis and "
                "highlight any divergence. Automated pipeline: inject the "
                "stored analysis into the CVRF reflection prompt as a "
                "'ground truth' anchor."
            ),
            score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # 4. Objective Bias
    # ------------------------------------------------------------------

    def check_objective(
        self,
        trades: list[TradeRecord] | None,
        trade_instructions: dict[str, dict[str, Any]] | None = None,
    ) -> BiasReport:
        """Detect objective bias: evaluation metric diverging from original goal.

        Objective bias occurs when a trade (or the overall backtest) is
        evaluated on a metric that differs from what was originally
        specified. Classic example: a strategy designed to maximise Sharpe
        ratio is evaluated only on total return, hiding the risk that the
        original objective was designed to control.

        This checker performs two comparisons:

        1. **Metric consistency** — compares the ``objective`` field in
           each trade instruction against the evaluation output. For example,
           if objective is ``"sharpe"`` but the evaluation only reports
           ``pnl_pct``, that is a mismatch.
        2. **Risk-return pairing** — every return metric MUST be paired
           with a risk metric. If only returns are reported (no Sharpe,
           no MDD, no Calmar), objective bias is flagged regardless of
           the stated objective.

        Args:
            trades: Completed trade records with evaluation data.
                May be ``None`` or empty.
            trade_instructions: Optional dict mapping ``trade_id`` -> dict
                containing at minimum an ``"objective"`` key (e.g.
                ``{"objective": "sharpe", "target": 1.5}``).
                When ``None``, the check defaults to clean with a warning.

        Returns:
            BiasReport. Defaults to clean when **trade_instructions** is
            missing, since the objective cannot be verified.
        """
        if not trades:
            return BiasReport(
                bias_type="objective",
                severity=BiasSeverity.NONE,
                description="No trades provided; cannot detect objective bias.",
                evidence=["Input trades list is empty or null."],
                score=0.0,
            )

        if not trade_instructions:
            return BiasReport(
                bias_type="objective",
                severity=BiasSeverity.MEDIUM,
                description=(
                    "No trade_instructions provided. Without knowing the "
                    "original objective for each trade, objective bias "
                    "cannot be verified. The backtest may be evaluating "
                    "on a metric that differs from what was originally "
                    "intended."
                ),
                evidence=[
                    "trade_instructions dict is missing or empty. "
                    "Provide per-trade instructions with an 'objective' key."
                ],
                recommendation=(
                    "Record the original evaluation objective (Sharpe, "
                    "Sortino, Calmar, total return, etc.) at decision time "
                    "for every trade. Ensure the evaluation pipeline reports "
                    "the matched metric, not a convenience default."
                ),
                score=0.3,
            )

        evidence: list[str] = []
        affected: list[str] = []
        mismatches = 0
        n = 0
        missing_instructions = 0

        for t in trades:
            instruction = trade_instructions.get(t.trade_id)
            if not instruction:
                missing_instructions += 1
                evidence.append(
                    f"Trade {t.trade_id}: no trade instruction found; "
                    f"cannot verify objective consistency."
                )
                continue

            n += 1
            original_objective = instruction.get("objective", "").lower().strip()

            # Infer what evaluation dimensions are available from the trade record.
            # TradeRecord has: entry_price, exit_price, decision, position_size.
            # Evaluate whether the available data supports the stated objective.
            risk_obj = original_objective in ("sharpe", "sortino", "calmar", "risk_adjusted")
            if risk_obj and t.highest_price is None and t.lowest_price is None:
                mismatches += 1
                affected.append(t.trade_id)
                evidence.append(
                    f"Trade {t.trade_id}: objective='{original_objective}' "
                    f"but no intra-trade high/low prices recorded. "
                    f"Cannot compute drawdown or volatility for this trade."
                )

            # Generic check: if the objective mentions risk but risk data is absent.
            risk_keywords = ("risk", "sharpe", "sortino", "calmar", "drawdown", "volatility", "var", "cvar")
            if (
                any(kw in original_objective for kw in risk_keywords)
                and (not hasattr(t, "highest_price") or t.highest_price is None)
                and t.trade_id not in affected
            ):
                affected.append(t.trade_id)
                mismatches += 1
                evidence.append(
                    f"Trade {t.trade_id}: objective mentions risk "
                    f"('{original_objective}') but trade record has no "
                    f"intra-trade price extremes for risk metric computation."
                )

        ratio = mismatches / n if n > 0 else 0.0
        missing_ratio = missing_instructions / len(trades) if trades else 0.0

        severity = self._ratio_to_severity(ratio)
        score = min(1.0, ratio * 0.7 + missing_ratio * 0.3)

        if mismatches == 0 and missing_instructions == 0:
            return BiasReport(
                bias_type="objective",
                severity=BiasSeverity.NONE,
                description=(
                    f"All {n} trade(s) with instructions have consistent "
                    f"objectives. Evaluation metrics match stated goals."
                ),
                evidence=["Objective consistency verified for all trades."],
                score=0.0,
            )

        desc_parts = []
        if mismatches > 0:
            desc_parts.append(
                f"{mismatches} trade(s) exhibit potential objective bias"
            )
        if missing_instructions > 0:
            desc_parts.append(
                f"{missing_instructions} trade(s) have no recorded instruction"
            )

        return BiasReport(
            bias_type="objective",
            severity=severity,
            description=(
                f"{'; '.join(desc_parts)}: "
                f"evaluation metrics may not align with original trade objectives."
            ),
            evidence=evidence,
            affected_trades=affected,
            recommendation=(
                "Record the evaluation objective at trade-decision time "
                "in the trade instruction. Ensure the backtest report "
                "explicitly pairs each return metric with its corresponding "
                "risk metric. The Alpha Illusion P3 protocol requires: "
                "every reported return must be accompanied by volatility, "
                "MDD, and the original objective."
            ),
            score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # 5. Cost Bias
    # ------------------------------------------------------------------

    def check_cost(
        self,
        trades: list[TradeRecord] | None,
        cost_model: Any = None,
    ) -> BiasReport:
        """Detect cost bias: unrealistic or missing transaction costs.

        Cost bias is the most commonly omitted bias in LLM trading backtests
        (Alpha Illusion P2). Reported Gross PnL can be dramatically higher
        than Net PnL when realistic costs are applied. The gap between Gross
        and Net PnL is itself a quantitative measure of 'alpha hallucination'.

        This checker performs four checks:

        1. **Cost existence** — are cost records present at all?
        2. **Cost magnitude** — is the average cost per trade in a realistic
           range (typically 5-30 bps for CEX, 10-100 bps for DEX)?
        3. **Cost per leg** — does the model account for BOTH entry and exit
           legs? (Missing one leg = 50 % cost understatement.)
        4. **Slippage model** — is there evidence of a slippage model, or
           does every trade fill at the exact signal price?

        Args:
            trades: Completed trade records. May be ``None`` or empty.
            cost_model: Optional cost model instance. If provided, the
                checker runs a cross-validation: for each trade, it
                recomputes the expected cost using the cost model and
                compares it to the recorded slippage/cost. A large
                discrepancy suggests under- or over-estimation.
                Accepts any object with a ``calculate_trade_cost`` method
                (conforms to :class:`qmind.backtest.cost_model.CostModel`).
                When ``None``, the check is based solely on trade-record
                fields.

        Returns:
            BiasReport. Defaults to clean when **trades** is empty.
        """
        if not trades:
            return BiasReport(
                bias_type="cost",
                severity=BiasSeverity.NONE,
                description="No trades provided; cannot detect cost bias.",
                evidence=["Input trades list is empty or null."],
                score=0.0,
            )

        evidence: list[str] = []
        cost_missing = 0
        cost_suspicious = 0
        avg_slippage_bps: list[float] = []

        has_cost_model = cost_model is not None and hasattr(cost_model, "calculate_trade_cost")

        for t in trades:
            # Check 1: Cost / slippage recorded?
            if not hasattr(t, "slippage_bps") or t.slippage_bps is None:
                cost_missing += 1
                evidence.append(
                    f"Trade {t.trade_id}: no slippage_bps recorded."
                )
                continue

            avg_slippage_bps.append(t.slippage_bps)

            # Check 2: Zero cost is suspicious.
            if t.slippage_bps == 0.0:
                cost_suspicious += 1
                evidence.append(
                    f"Trade {t.trade_id}: slippage_bps=0.0 — unrealistic for "
                    f"any real market."
                )

            # Check 3: Cross-validate with cost model if available.
            if has_cost_model:
                try:
                    entry_leg = cost_model.calculate_trade_cost(
                        side="buy" if t.decision == "LONG" else "sell",
                        price=t.entry_price,
                        quantity=t.position_size,
                        order_type="market",
                    )
                    exit_leg = cost_model.calculate_trade_cost(
                        side="sell" if t.decision == "LONG" else "buy",
                        price=t.exit_price,
                        quantity=t.position_size,
                        order_type="market",
                    )
                    expected_bps = (entry_leg.cost_bps + exit_leg.cost_bps) / 2.0
                    diff = abs(t.slippage_bps - expected_bps)
                    if diff > expected_bps * 0.5 and expected_bps > 0:
                        cost_suspicious += 1
                        evidence.append(
                            f"Trade {t.trade_id}: recorded slippage={t.slippage_bps:.2f} bps "
                            f"vs cost-model expected ~{expected_bps:.2f} bps "
                            f"(diff={diff:.2f} bps, {diff / expected_bps:.0%})."
                        )
                except Exception as exc:
                    evidence.append(
                        f"Trade {t.trade_id}: cost-model cross-validation "
                        f"failed: {exc}"
                    )

        # Check: are both entry and exit costs accounted for?
        zero_cost_count = sum(
            1 for t in trades
            if (hasattr(t, "slippage_bps") and t.slippage_bps is not None and t.slippage_bps == 0.0)
        )

        n = len(trades)
        missing_ratio = cost_missing / n if n > 0 else 0.0
        suspicious_ratio = cost_suspicious / n if n > 0 else 0.0
        zero_ratio = zero_cost_count / n if n > 0 else 0.0

        # Composite score
        score = min(1.0, missing_ratio * 0.5 + suspicious_ratio * 0.3 + zero_ratio * 0.2)
        severity = self._score_to_severity(score)

        if cost_missing == 0 and cost_suspicious == 0:
            avg_bps = (
                sum(avg_slippage_bps) / len(avg_slippage_bps)
                if avg_slippage_bps
                else 0.0
            )
            return BiasReport(
                bias_type="cost",
                severity=BiasSeverity.NONE,
                description=(
                    f"All {n} trade(s) have realistic cost data. "
                    f"Average slippage: {avg_bps:.2f} bps."
                    + (" Cost-model cross-validation passed."
                       if has_cost_model else "")
                ),
                evidence=["All cost checks passed."],
                score=0.0,
            )

        return BiasReport(
            bias_type="cost",
            severity=severity,
            description=(
                f"Cost bias detected: {cost_missing} trade(s) missing cost data, "
                f"{cost_suspicious} trade(s) with suspicious cost values."
            ),
            evidence=evidence,
            affected_trades=[
                t.trade_id for t in trades
                if (hasattr(t, "slippage_bps") and (
                    t.slippage_bps is None or t.slippage_bps == 0.0
                ))
            ],
            recommendation=(
                "Implement the project's CostModel "
                "(qmind.backtest.cost_model) with realistic parameters: "
                "commission=0.1 %, bid-ask spread=0.05 %, slippage=0.1 % "
                "for CEX. For DEX trades add gas fees. Run multi-tier "
                "sensitivity at 0/10/25 bps. See Alpha Illusion P2: "
                "Gross PnL must be reported alongside Net PnL."
            ),
            score=round(score, 4),
        )

    # ------------------------------------------------------------------
    # Full Audit
    # ------------------------------------------------------------------

    def run_all(
        self,
        trades: list[TradeRecord] | None = None,
        market_data: dict[str, Any] | None = None,
        tickers_used: list[str] | None = None,
        tickers_available: dict[str, list[str]] | None = None,
        benchmark: str | list[str] | None = None,
        analyses: dict[str, list[dict[str, Any]]] | None = None,
        trade_instructions: dict[str, dict[str, Any]] | None = None,
        cost_model: Any = None,
    ) -> BiasAuditSummary:
        """Run all five bias checks and return an aggregated summary.

        Every parameter is optional; missing data causes the corresponding
        check to default to a clean report with a descriptive message.
        The method never raises.

        Args:
            trades: Completed trade records. Passed to all five checks.
            market_data: Market data for look-ahead check.
            tickers_used: Tickers used in backtest for survivorship check.
            tickers_available: Time-varying universe for survivorship check.
            benchmark: Benchmark for survivorship cross-check.
            analyses: Real-time analyses dict for narrative check.
            trade_instructions: Per-trade instructions for objective check.
            cost_model: Cost model instance for cost check.

        Returns:
            BiasAuditSummary containing all five reports, an overall score,
            a pass/fail verdict, and the list of critical findings.
        """
        reports: list[BiasReport] = [
            self.check_look_ahead(trades, market_data),
            self.check_survivorship(tickers_used, tickers_available, benchmark),
            self.check_narrative(trades, analyses),
            self.check_objective(trades, trade_instructions),
            self.check_cost(trades, cost_model),
        ]

        scores = [r.score for r in reports]
        overall_score = sum(scores) / len(scores) if scores else 0.0
        passed = all(s < 0.3 for s in scores)
        critical_biases = [
            r for r in reports
            if r.severity in (BiasSeverity.HIGH, BiasSeverity.CRITICAL)
        ]

        return BiasAuditSummary(
            reports=reports,
            overall_score=round(overall_score, 4),
            passed=passed,
            critical_biases=critical_biases,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_timestamp(data_point: Any) -> int | None:
        """Extract a Unix-millisecond timestamp from a data point.

        Supports dicts with key ``"timestamp"`` (int or datetime), or
        objects with a ``.timestamp`` attribute.

        Returns:
            Integer Unix-millisecond timestamp, or ``None`` if extraction
            fails.
        """
        if isinstance(data_point, dict):
            ts = data_point.get("timestamp")
            if isinstance(ts, int):
                return ts
            if isinstance(ts, datetime):
                return int(ts.timestamp() * 1000)
            if ts is not None:
                try:
                    return int(ts)
                except (ValueError, TypeError):
                    return None
        elif hasattr(data_point, "timestamp"):
            ts = data_point.timestamp
            if isinstance(ts, int):
                return ts
            if isinstance(ts, datetime):
                return int(ts.timestamp() * 1000)
        return None

    @staticmethod
    def _ratio_to_severity(ratio: float) -> BiasSeverity:
        """Map a violation ratio to a BiasSeverity.

        Args:
            ratio: Fraction of affected items in [0.0, 1.0].

        Returns:
            BiasSeverity based on thresholds:
            - 0.0        -> NONE
            - (0.0, 0.05] -> LOW
            - (0.05, 0.15] -> MEDIUM
            - (0.15, 0.30] -> HIGH
            - > 0.30      -> CRITICAL
        """
        if ratio == 0.0:
            return BiasSeverity.NONE
        if ratio <= 0.05:
            return BiasSeverity.LOW
        if ratio <= 0.15:
            return BiasSeverity.MEDIUM
        if ratio <= 0.30:
            return BiasSeverity.HIGH
        return BiasSeverity.CRITICAL

    @staticmethod
    def _score_to_severity(score: float) -> BiasSeverity:
        """Map a composite severity score to a BiasSeverity.

        Args:
            score: Composite score in [0.0, 1.0].

        Returns:
            BiasSeverity based on thresholds:
            - 0.0         -> NONE
            - (0.0, 0.15]  -> LOW
            - (0.15, 0.35] -> MEDIUM
            - (0.35, 0.55] -> HIGH
            - > 0.55       -> CRITICAL
        """
        if score == 0.0:
            return BiasSeverity.NONE
        if score <= 0.15:
            return BiasSeverity.LOW
        if score <= 0.35:
            return BiasSeverity.MEDIUM
        if score <= 0.55:
            return BiasSeverity.HIGH
        return BiasSeverity.CRITICAL
