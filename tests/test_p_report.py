"""Tests for qmind.backtest.p_report — P1-P6 Alpha Illusion compliance report."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import pytest

from qmind.backtest.p_report import (
    AblationResult,
    BenchmarkResult,
    CostTierResult,
    P1P6Report,
    P1P6ReportResult,
    P1Report,
    P2Report,
    P3Report,
    P4Report,
    P5Report,
    P6Report,
)


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def mock_backtest_results() -> dict[str, Any]:
    base = pd.Timestamp(2024, 1, 1, tzinfo=UTC)
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "splits": [
            _make_split(0, base, 365, 90, 90),
            _make_split(1, base, 365, 90, 90, shift_days=180),
        ],
        "total_span_days": 730,
        "as_of_timestamps": True,
        "time_guard_enforced": True,
        "max_look_ahead_bias_days": 0.0,
        "data_sources": ["yfinance", "binance"],
        "n_data_points": 50000,
        "signal_execution_model": "next_close",
        "slippage_model": "fixed_10bps",
        "avg_slippage_bps": 8.5,
        "fill_rate": 0.97,
        "cost_tiers": {
            "0bps": {
                "gross_pnl_pct": 45.2,
                "net_pnl_pct": 45.2,
                "sharpe": 1.82,
                "max_dd_pct": -12.3,
                "total_cost_pct": 0.0,
            },
            "10bps": {
                "gross_pnl_pct": 45.2,
                "net_pnl_pct": 38.7,
                "sharpe": 1.56,
                "max_dd_pct": -13.1,
                "total_cost_pct": 6.5,
            },
            "25bps": {
                "gross_pnl_pct": 45.2,
                "net_pnl_pct": 28.4,
                "sharpe": 1.14,
                "max_dd_pct": -14.8,
                "total_cost_pct": 16.8,
            },
        },
        "detailed_commission_bps": 5.0,
        "detailed_spread_bps": 3.0,
        "detailed_slippage_bps": 2.0,
        "detailed_gas_bps": 0.0,
        "strategy_result": {
            "total_return_pct": 38.7,
            "annualised_return_pct": 19.2,
            "sharpe": 1.56,
            "max_dd_pct": -13.1,
            "volatility_pct": 18.5,
        },
        "benchmarks": {
            "BuyHold": {
                "total_return_pct": 22.0,
                "annualised_return_pct": 11.0,
                "sharpe": 0.95,
                "max_dd_pct": -18.2,
                "volatility_pct": 22.0,
            },
            "EqualWeight": {
                "total_return_pct": 18.5,
                "annualised_return_pct": 9.2,
                "sharpe": 0.82,
                "max_dd_pct": -20.1,
                "volatility_pct": 21.5,
            },
        },
        "ablation": {
            "single_agent": {
                "total_return_pct": 25.0,
                "sharpe": 1.10,
                "max_dd_pct": -15.0,
                "n_trades": 45,
                "avg_holding_period": "6h",
                "total_llm_calls": 90,
                "total_token_cost_usd": 4.50,
            },
            "multi_no_debate": {
                "total_return_pct": 32.0,
                "sharpe": 1.35,
                "max_dd_pct": -14.0,
                "n_trades": 52,
                "avg_holding_period": "6h",
                "total_llm_calls": 180,
                "total_token_cost_usd": 9.00,
            },
            "full": {
                "total_return_pct": 38.7,
                "sharpe": 1.56,
                "max_dd_pct": -13.1,
                "n_trades": 58,
                "avg_holding_period": "6h",
                "total_llm_calls": 320,
                "total_token_cost_usd": 16.00,
            },
        },
    }


@pytest.fixture
def mock_config() -> dict[str, Any]:
    return {
        "symbol": "BTC/USDT",
        "timeframe": "1h",
        "window_mode": "walk_forward",
        "n_splits": 2,
        "test_size": 0.2,
        "validation_size": 0.2,
        "signal_execution_model": "next_close",
    }


def _make_split(
    fold: int,
    base: pd.Timestamp,
    train_days: int,
    val_days: int,
    test_days: int,
    shift_days: int = 0,
) -> Any:
    """Create a mock TimeSplit using pd.Timestamp (matches real production flow)."""
    from qmind.backtest.partition import TimeSplit

    offset = shift_days * fold
    return TimeSplit(
        fold=fold,
        train_start=base if fold == 0 else _shift(base, offset),
        train_end=_shift(base, offset + train_days),
        val_start=_shift(base, offset + train_days),
        val_end=_shift(base, offset + train_days + val_days),
        test_start=_shift(base, offset + train_days + val_days),
        test_end=_shift(base, offset + train_days + val_days + test_days),
    )


def _shift(ts: pd.Timestamp, days: int) -> pd.Timestamp:
    return ts + timedelta(days=days)


# ===========================================================================
# 1. Generate P1-P6 report with mock data
# ===========================================================================


class TestGenerate:
    def test_generate_with_full_data(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert isinstance(report, P1P6ReportResult)
        assert report.symbol == "BTC/USDT"
        assert report.timeframe == "1h"
        assert report.title == "P1-P6 Compliance Report - BTC/USDT (1h)"

    def test_generate_with_none_inputs(self) -> None:
        report = P1P6Report.generate(None, None)
        assert isinstance(report, P1P6ReportResult)
        assert report.symbol == ""
        assert report.timeframe == ""

    def test_generate_with_empty_dicts(self) -> None:
        report = P1P6Report.generate({}, {})
        assert isinstance(report, P1P6ReportResult)


# ===========================================================================
# 2. P1 report sections
# ===========================================================================


class TestP1:
    def test_sections_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p1 = report.p1
        assert isinstance(p1, P1Report)
        assert p1.method == "walk_forward"
        assert p1.n_splits == 2
        assert len(p1.train_periods) == 2
        assert len(p1.val_periods) == 2
        assert len(p1.test_periods) == 2
        assert p1.total_span_days > 0
        assert p1.min_train_days > 0

    def test_spans_multiple_regimes(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p1.spans_multiple_regimes is False  # 730 < 1095

    def test_spans_multiple_regimes_true(self, mock_config: dict[str, Any]) -> None:
        base = pd.Timestamp(2023, 1, 1, tzinfo=UTC)
        results: dict[str, Any] = {
            "splits": [_make_split(0, base, 1100, 90, 90)],
            "total_span_days": 1280,
        }
        report = P1P6Report.generate(results, mock_config)
        assert report.p1.spans_multiple_regimes is True

    def test_passed_when_splits_exist_and_span_ge_365(
        self, mock_config: dict[str, Any],
    ) -> None:
        base = pd.Timestamp(2024, 1, 1, tzinfo=UTC)
        results: dict[str, Any] = {
            "splits": [_make_split(0, base, 200, 50, 50)],
            "total_span_days": 400,
        }
        report = P1P6Report.generate(results, mock_config)
        assert report.p1.passed is True

    def test_failed_when_no_splits(self, mock_config: dict[str, Any]) -> None:
        results: dict[str, Any] = {"splits": []}
        report = P1P6Report.generate(results, mock_config)
        assert report.p1.passed is False

    def test_expanding_window_detection(self, mock_config: dict[str, Any]) -> None:
        cfg = dict(mock_config, window_mode="expanding")
        base = pd.Timestamp(2024, 1, 1, tzinfo=UTC)
        results: dict[str, Any] = {
            "splits": [_make_split(0, base, 200, 50, 50)],
            "total_span_days": 400,
        }
        report = P1P6Report.generate(results, cfg)
        assert report.p1.method == "expanding_window"


# ===========================================================================
# 3. P2 Point-in-Time section
# ===========================================================================


class TestP2:
    def test_sections_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p2 = report.p2
        assert isinstance(p2, P2Report)
        assert p2.as_of_timestamps is True
        assert p2.time_guard_enforced is True
        assert p2.max_look_ahead_bias_days == 0.0
        assert "yfinance" in p2.data_sources
        assert "binance" in p2.data_sources
        assert p2.n_data_points == 50000

    def test_passed_when_no_look_ahead(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p2.passed is True

    def test_failed_when_look_ahead_present(self, mock_config: dict[str, Any]) -> None:
        results: dict[str, Any] = {
            "max_look_ahead_bias_days": 2.5,
            "as_of_timestamps": False,
        }
        report = P1P6Report.generate(results, mock_config)
        assert report.p2.passed is False
        assert report.p2.max_look_ahead_bias_days == 2.5

    def test_missing_data_defaults(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        assert report.p2.as_of_timestamps is False
        assert report.p2.time_guard_enforced is False
        assert report.p2.n_data_points == 0
        assert report.p2.data_sources == []


# ===========================================================================
# 4. P3 execution timing
# ===========================================================================


class TestP3:
    def test_sections_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p3 = report.p3
        assert isinstance(p3, P3Report)
        assert p3.signal_execution_model == "next_close"
        assert p3.slippage_model == "fixed_10bps"
        assert p3.avg_slippage_bps == pytest.approx(8.5)
        assert p3.fill_rate == pytest.approx(0.97)

    def test_passed_when_next_close(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p3.passed is True

    def test_failed_when_same_close(
        self, mock_config: dict[str, Any],
    ) -> None:
        results: dict[str, Any] = {"signal_execution_model": "same_close"}
        report = P1P6Report.generate(results, mock_config)
        assert report.p3.passed is False

    def test_default_model_from_config(
        self, mock_backtest_results: dict[str, Any],
    ) -> None:
        cfg: dict[str, Any] = {"signal_execution_model": "next_close"}
        report = P1P6Report.generate({}, cfg)
        assert report.p3.signal_execution_model == "next_close"


# ===========================================================================
# 5. P4 cost tiers (3 tiers)
# ===========================================================================


class TestP4:
    def test_three_cost_tiers_present(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p4 = report.p4
        assert isinstance(p4, P4Report)
        assert "0bps" in p4.cost_tiers
        assert "10bps" in p4.cost_tiers
        assert "25bps" in p4.cost_tiers

    def test_cost_tier_values(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p4 = report.p4

        # 0bps tier: gross == net
        t0 = p4.cost_tiers["0bps"]
        assert t0.gross_pnl_pct == 45.2
        assert t0.net_pnl_pct == 45.2
        assert t0.sharpe == 1.82
        assert t0.total_cost_pct == 0.0

        # 10bps tier
        t10 = p4.cost_tiers["10bps"]
        assert t10.net_pnl_pct == 38.7
        assert t10.sharpe == 1.56

        # 25bps tier
        t25 = p4.cost_tiers["25bps"]
        assert t25.net_pnl_pct == 28.4

    def test_detailed_cost_breakdown(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p4 = report.p4
        assert p4.detailed_commission_bps == 5.0
        assert p4.detailed_spread_bps == 3.0
        assert p4.detailed_slippage_bps == 2.0
        assert p4.detailed_gas_bps == 0.0

    def test_gross_minus_net_gap(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        # gap = (gross - net) * 100 = (45.2 - 38.7) * 100 = 650 bps
        expected_gap = (45.2 - 38.7) * 100
        assert report.p4.gross_minus_net_gap_bps == pytest.approx(expected_gap)

    def test_fallback_tiers_when_missing(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        p4 = report.p4
        assert "0bps" in p4.cost_tiers
        assert "10bps" in p4.cost_tiers
        assert "25bps" in p4.cost_tiers
        # All zero-filled
        for label in ("0bps", "10bps", "25bps"):
            assert p4.cost_tiers[label].gross_pnl_pct == 0.0
            assert p4.cost_tiers[label].net_pnl_pct == 0.0

    def test_passed_when_tiers_exist(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p4.passed is True


# ===========================================================================
# 6. P5 benchmark comparison
# ===========================================================================


class TestP5:
    def test_sections_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p5 = report.p5
        assert isinstance(p5, P5Report)
        assert p5.strategy is not None
        assert p5.strategy.total_return_pct == 38.7
        assert p5.strategy.annualised_return_pct == 19.2
        assert p5.strategy.sharpe == 1.56

    def test_benchmarks_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p5 = report.p5
        assert "BuyHold" in p5.benchmarks
        assert "EqualWeight" in p5.benchmarks
        bh = p5.benchmarks["BuyHold"]
        assert bh.total_return_pct == 22.0
        assert bh.sharpe == 0.95

    def test_outperformance_calculation(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p5 = report.p5
        # Strategy (38.7) - Best benchmark (BuyHold 22.0) = 16.7 -> 1670 bps
        expected_bps = (38.7 - 22.0) * 100
        assert p5.outperformance_bps == pytest.approx(expected_bps)
        assert p5.beats_all_benchmarks is True

    def test_strategy_does_not_beat_all(
        self, mock_config: dict[str, Any],
    ) -> None:
        results: dict[str, Any] = {
            "strategy_result": {
                "total_return_pct": 15.0,
                "annualised_return_pct": 7.5,
                "sharpe": 0.8,
                "max_dd_pct": -15.0,
                "volatility_pct": 20.0,
            },
            "benchmarks": {
                "SP500": {
                    "total_return_pct": 22.0,
                    "annualised_return_pct": 11.0,
                    "sharpe": 1.2,
                    "max_dd_pct": -10.0,
                    "volatility_pct": 15.0,
                },
            },
        }
        report = P1P6Report.generate(results, mock_config)
        assert report.p5.beats_all_benchmarks is False
        assert report.p5.outperformance_bps < 0

    def test_missing_strategy_not_passed(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        assert report.p5.passed is False
        assert report.p5.strategy is None

    def test_passed_with_strategy_and_benchmarks(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p5.passed is True


# ===========================================================================
# 7. P6 ablation results (can be None)
# ===========================================================================


class TestP6:
    def test_ablation_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.p6 is not None
        p6 = report.p6
        assert isinstance(p6, P6Report)
        assert "single_agent" in p6.results
        assert "multi_no_debate" in p6.results
        assert "full" in p6.results

    def test_ablation_result_values(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p6 = report.p6
        assert p6 is not None

        single = p6.results["single_agent"]
        assert single.total_return_pct == 25.0
        assert single.sharpe == 1.10
        assert single.n_trades == 45
        assert single.total_llm_calls == 90
        assert single.total_token_cost_usd == 4.50

        full = p6.results["full"]
        assert full.total_return_pct == 38.7
        assert full.sharpe == 1.56
        assert full.n_trades == 58
        assert full.total_llm_calls == 320

    def test_debate_contribution_calculated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p6 = report.p6
        assert p6 is not None
        # full (38.7) - multi_no_debate (32.0) = 6.7 -> 670 bps
        assert p6.debate_contribution_bps == pytest.approx(670.0)

    def test_multi_agent_contribution_calculated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p6 = report.p6
        assert p6 is not None
        # full (38.7) - single_agent (25.0) = 13.7 -> 1370 bps
        assert p6.multi_agent_contribution_bps == pytest.approx(1370.0)

    def test_cost_performance_ratio(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p6 = report.p6
        assert p6 is not None
        # single: 4.5 / 25.0 = 0.18
        assert p6.cost_performance_ratio["single_agent"] == pytest.approx(0.18)
        # full: 16.0 / 38.7
        assert p6.cost_performance_ratio["full"] == pytest.approx(
            16.0 / 38.7
        )

    def test_ablation_can_be_none(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        assert report.p6 is None

    def test_passed_with_two_variants(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        p6 = report.p6
        assert p6 is not None
        assert p6.passed is True

    def test_not_passed_with_single_variant(
        self, mock_config: dict[str, Any],
    ) -> None:
        results: dict[str, Any] = {
            "ablation": {
                "single_agent": {
                    "total_return_pct": 25.0, "sharpe": 1.1,
                    "max_dd_pct": -15.0, "n_trades": 45,
                    "avg_holding_period": "6h", "total_llm_calls": 90,
                    "total_token_cost_usd": 4.5,
                },
            },
        }
        report = P1P6Report.generate(results, mock_config)
        p6 = report.p6
        assert p6 is not None
        assert p6.passed is False

    def test_zero_return_cost_perf_is_inf(
        self, mock_config: dict[str, Any],
    ) -> None:
        results: dict[str, Any] = {
            "ablation": {
                "single_agent": {
                    "total_return_pct": 0.0, "sharpe": 0.0,
                    "max_dd_pct": 0.0, "n_trades": 1,
                    "avg_holding_period": "1h", "total_llm_calls": 5,
                    "total_token_cost_usd": 1.0,
                },
                "full": {
                    "total_return_pct": 10.0, "sharpe": 0.5,
                    "max_dd_pct": -5.0, "n_trades": 10,
                    "avg_holding_period": "1h", "total_llm_calls": 50,
                    "total_token_cost_usd": 5.0,
                },
            },
        }
        report = P1P6Report.generate(results, mock_config)
        p6 = report.p6
        assert p6 is not None
        assert p6.cost_performance_ratio["single_agent"] == float("inf")


# ===========================================================================
# 8. to_markdown() produces non-empty output
# ===========================================================================


class TestToMarkdown:
    def test_to_markdown_non_empty(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        md = report.to_markdown()
        assert isinstance(md, str)
        assert len(md) > 0

    def test_to_markdown_contains_title(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        md = report.to_markdown()
        assert "P1-P6 Compliance Report" in md
        assert "BTC/USDT" in md

    def test_to_markdown_contains_sections(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        md = report.to_markdown()
        assert "P1: Time Consistency" in md
        assert "P2: Point-in-Time Data" in md
        assert "P3: Execution Timing" in md
        assert "P4: Cost Realism" in md
        assert "P5: Benchmark Comparison" in md
        assert "P6: Ablation" in md

    def test_to_markdown_without_ablation(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        md = report.to_markdown()
        assert len(md) > 0
        # The P6 section heading should not appear when ablation data is missing.
        # "P1-P6 Compliance Report" in the title is fine; check for the section header.
        assert "P6: Ablation" not in md

    def test_to_markdown_contains_summary(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        md = report.to_markdown()
        assert "Summary" in md
        assert report.summary in md

    def test_to_markdown_contains_pass_fail_indicators(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        md = report.to_markdown()
        assert "PASS" in md or "FAIL" in md


# ===========================================================================
# 9. Summary string is generated
# ===========================================================================


class TestSummary:
    def test_summary_is_populated(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_summary_contains_symbol(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert "BTC/USDT" in report.summary

    def test_summary_contains_pass_count(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert "/" in report.summary  # e.g. "5/6"
        assert "passed" in report.summary.lower()

    def test_summary_with_no_data(self) -> None:
        report = P1P6Report.generate(None, None)
        assert isinstance(report.summary, str)
        assert len(report.summary) > 0

    def test_summary_p1_through_p5(
        self, mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate({}, mock_config)
        # Without ablation, we expect 5 levels
        assert "P1" in report.summary
        assert "P2" in report.summary
        assert "P3" in report.summary
        assert "P4" in report.summary
        assert "P5" in report.summary
        assert "P6" not in report.summary

    def test_n_passed_count(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert report.n_passed >= 0
        assert report.n_passed <= 6

    def test_all_passed_flag(
        self, mock_backtest_results: dict[str, Any],
        mock_config: dict[str, Any],
    ) -> None:
        report = P1P6Report.generate(mock_backtest_results, mock_config)
        assert isinstance(report.all_passed, bool)


# ===========================================================================
# Data type default / boundary tests
# ===========================================================================


class TestDataTypes:
    def test_p1_report_defaults(self) -> None:
        p1 = P1Report(
            method="walk_forward", n_splits=4,
            min_train_days=365, total_span_days=730,
        )
        assert p1.method == "walk_forward"
        assert p1.n_splits == 4
        assert p1.total_span_days == 730
        assert p1.passed is False

    def test_p2_report_defaults(self) -> None:
        p2 = P2Report(max_look_ahead_bias_days=0.0, n_data_points=0)
        assert p2.as_of_timestamps is False
        assert p2.time_guard_enforced is False
        assert p2.max_look_ahead_bias_days == 0.0
        assert p2.passed is False

    def test_p3_report_defaults(self) -> None:
        p3 = P3Report(
            signal_execution_model="next_close",
            avg_slippage_bps=0.0, fill_rate=0.0,
        )
        assert p3.signal_execution_model == "next_close"
        assert p3.avg_slippage_bps == 0.0
        assert p3.fill_rate == 0.0
        # `passed` is stored, not computed; must be set explicitly
        assert p3.passed is False

    def test_cost_tier_result_validation(self) -> None:
        c = CostTierResult(
            gross_pnl_pct=10.0, net_pnl_pct=8.0,
            sharpe=1.2, max_dd_pct=-5.0, total_cost_pct=2.0,
        )
        assert c.total_cost_pct >= 0.0

    def test_benchmark_result_defaults(self) -> None:
        b = BenchmarkResult(total_return_pct=10.0, annualised_return_pct=5.0)
        assert b.sharpe == 0.0
        assert b.max_dd_pct == 0.0
        assert b.volatility_pct == 0.0

    def test_ablation_result_defaults(self) -> None:
        a = AblationResult(
            total_return_pct=10.0, sharpe=0.5, max_dd_pct=-5.0,
            n_trades=10, total_llm_calls=50, total_token_cost_usd=2.5,
        )
        assert a.avg_holding_period == ""
        assert a.total_token_cost_usd >= 0.0
