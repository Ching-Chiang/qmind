"""Tests for qmind.audit.bias_checker — 5-category backtest bias detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest

from qmind.audit.bias_checker import (
    BiasAuditSummary,
    BiasChecker,
    BiasReport,
    BiasSeverity,
)
from qmind.learning.evaluator import TradeRecord


# ===========================================================================
# Fixtures
# ===========================================================================


@pytest.fixture
def checker() -> BiasChecker:
    return BiasChecker()


@pytest.fixture
def clean_trades() -> list[TradeRecord]:
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
    return [
        TradeRecord(
            trade_id="T-01",
            symbol="BTC/USDT",
            decision="LONG",
            entry_price=85000.0,
            exit_price=87000.0,
            position_size=0.1,
            entry_time=base,
            exit_time=base + timedelta(hours=6),
            highest_price=87500.0,
            lowest_price=84800.0,
            slippage_bps=8.0,
        ),
        TradeRecord(
            trade_id="T-02",
            symbol="ETH/USDT",
            decision="SHORT",
            entry_price=3200.0,
            exit_price=3100.0,
            position_size=1.0,
            entry_time=base + timedelta(hours=12),
            exit_time=base + timedelta(hours=24),
            highest_price=3220.0,
            lowest_price=3080.0,
            slippage_bps=10.0,
        ),
    ]


@pytest.fixture
def clean_market_data() -> dict[str, Any]:
    base = datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC)
    return {
        "BTC/USDT": {
            "as_of": base - timedelta(minutes=5),
            "klines": [
                {"timestamp": int((base - timedelta(hours=2)).timestamp() * 1000)}
            ],
        },
        "ETH/USDT": {
            "as_of": base + timedelta(hours=11, minutes=55),
            "klines": [
                {"timestamp": int((base + timedelta(hours=10)).timestamp() * 1000)}
            ],
        },
    }


# ===========================================================================
# 1. Look-Ahead Bias: future data in trades -> flag
# ===========================================================================


class TestLookAhead:
    def test_clean_data_no_bias(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
        clean_market_data: dict[str, Any],
    ) -> None:
        report = checker.check_look_ahead(clean_trades, clean_market_data)
        assert report.bias_type == "look_ahead"
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0
        assert len(report.affected_trades) == 0

    def test_as_of_after_entry_time_flags_bias(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
    ) -> None:
        t = clean_trades[0]
        bad_md = {
            "BTC/USDT": {
                "as_of": t.entry_time + timedelta(hours=1),
                "klines": [],
            },
        }
        report = checker.check_look_ahead(clean_trades, bad_md)
        assert report.score > 0.0
        assert report.severity in (BiasSeverity.HIGH, BiasSeverity.CRITICAL)
        assert "T-01" in report.affected_trades

    def test_late_kline_timestamp_flags_bias(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
    ) -> None:
        t = clean_trades[0]
        bad_md = {
            "BTC/USDT": {
                "klines": [
                    {
                        "timestamp": int(
                            (t.entry_time + timedelta(minutes=30)).timestamp() * 1000
                        )
                    },
                ],
            },
        }
        report = checker.check_look_ahead(clean_trades, bad_md)
        assert report.score > 0.0
        assert "T-01" in report.affected_trades

    def test_empty_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_look_ahead([], {"dummy": {}})
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0
        assert "no trades provided" in report.description.lower()

    def test_none_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_look_ahead(None)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_missing_market_data_low_severity(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
    ) -> None:
        report = checker.check_look_ahead(clean_trades, None)
        assert report.severity == BiasSeverity.LOW
        assert report.score == 0.1

    def test_default_key_in_market_data(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
    ) -> None:
        t = clean_trades[0]
        bad_md = {
            "_default": {
                "as_of": t.entry_time + timedelta(hours=1),
            },
        }
        report = checker.check_look_ahead(clean_trades, bad_md)
        assert report.score > 0.0


# ===========================================================================
# 2. Survivorship Bias: missing delisted symbols -> flag
# ===========================================================================


class TestSurvivorship:
    def test_no_missing_tickers(
        self, checker: BiasChecker,
    ) -> None:
        tickers = ["AAPL", "MSFT", "GOOGL"]
        available = {
            "2024-01-01": ["AAPL", "MSFT", "GOOGL"],
            "2024-06-01": ["AAPL", "MSFT", "GOOGL"],
        }
        report = checker.check_survivorship(tickers, available)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_missing_delisted_flags_bias(
        self, checker: BiasChecker,
    ) -> None:
        tickers = ["AAPL", "MSFT"]
        available = {
            "2024-01-01": ["AAPL", "MSFT", "GOOGL", "ENRNQ"],
            "2024-06-01": ["AAPL", "MSFT", "GOOGL", "ENRNQ"],
        }
        report = checker.check_survivorship(tickers, available)
        assert report.score > 0.0
        assert len(report.affected_trades) > 0
        assert report.severity != BiasSeverity.NONE

    def test_empty_tickers_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_survivorship([], {"2024-01-01": ["AAPL"]})
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_none_tickers_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_survivorship(None)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_missing_available_data_low_severity(
        self, checker: BiasChecker,
    ) -> None:
        report = checker.check_survivorship(["AAPL"], None)
        assert report.severity == BiasSeverity.LOW
        assert report.score == 0.1

    def test_benchmark_cross_check_adds_evidence(
        self, checker: BiasChecker,
    ) -> None:
        tickers = ["AAPL"]
        available = {"2024-01-01": ["AAPL", "MSFT"]}
        report = checker.check_survivorship(tickers, available, benchmark="msft")
        assert len(report.evidence) > 0

    def test_case_insensitive_matching(
        self, checker: BiasChecker,
    ) -> None:
        tickers = ["aapl", "MSFT"]
        available = {"2024-01-01": ["AAPL", "MSFT", "GOOGL"]}
        report = checker.check_survivorship(tickers, available)
        assert report.severity == BiasSeverity.NONE

    def test_all_tickers_missing(self, checker: BiasChecker) -> None:
        tickers = ["AAPL"]
        available = {
            "2024-01-01": ["GOOGL", "MSFT"],
            "2024-06-01": ["GOOGL"],
        }
        report = checker.check_survivorship(tickers, available)
        assert report.score > 0.0


# ===========================================================================
# 3. Narrative Bias: post-hoc vs real-time reasoning mismatch
# ===========================================================================


class TestNarrative:
    def test_consistent_stance_no_bias(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        analyses = {
            "T-01": [
                {"stance": "bullish", "risk_factors": ["volatility"], "confidence": 0.7},
                {"stance": "bullish", "risk_factors": ["liquidity"], "confidence": 0.6},
            ],
        }
        report = checker.check_narrative(trades, analyses)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_bearish_stance_but_long_decision(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        analyses = {
            "T-01": [
                {"stance": "bearish", "risk_factors": ["overbought"], "confidence": 0.8},
                {"stance": "bearish", "risk_factors": ["resistance"], "confidence": 0.7},
            ],
        }
        report = checker.check_narrative(trades, analyses)
        assert report.score > 0.0
        assert len(report.affected_trades) > 0

    def test_bullish_stance_but_short_decision(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="SHORT",
                entry_price=85000.0, exit_price=83000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        analyses = {
            "T-01": [
                {"stance": "bullish", "risk_factors": [], "confidence": 0.7},
            ],
        }
        report = checker.check_narrative(trades, analyses)
        assert report.score > 0.0
        assert "T-01" in report.affected_trades

    def test_empty_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_narrative([], {})
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_none_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_narrative(None)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_missing_analyses_medium_severity(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        report = checker.check_narrative(trades, None)
        assert report.severity == BiasSeverity.MEDIUM
        assert report.score == pytest.approx(0.3)


# ===========================================================================
# 4. Objective Bias: evaluation metrics mismatch
# ===========================================================================


class TestObjective:
    def test_consistent_objective_no_bias(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                highest_price=87500.0, lowest_price=84800.0,
            ),
        ]
        instructions = {"T-01": {"objective": "total_return", "target": 5.0}}
        report = checker.check_objective(trades, instructions)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_risk_objective_missing_high_low(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        instructions = {"T-01": {"objective": "sharpe", "target": 1.5}}
        report = checker.check_objective(trades, instructions)
        assert report.score > 0.0
        assert "T-01" in report.affected_trades

    def test_risk_objective_with_high_low_passes(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                highest_price=87500.0, lowest_price=84800.0,
            ),
        ]
        instructions = {"T-01": {"objective": "sharpe", "target": 1.5}}
        report = checker.check_objective(trades, instructions)
        assert report.severity == BiasSeverity.NONE

    def test_risk_keywords_in_objective_missing_data(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        instructions = {"T-01": {"objective": "max_drawdown", "target": 0.05}}
        report = checker.check_objective(trades, instructions)
        assert report.score > 0.0

    def test_empty_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_objective([], {})
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_none_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_objective(None)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_missing_instructions_medium_severity(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        report = checker.check_objective(trades, None)
        assert report.severity == BiasSeverity.MEDIUM
        assert report.score == pytest.approx(0.3)

    def test_trade_missing_instruction(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                highest_price=87500.0, lowest_price=84800.0,
            ),
        ]
        instructions: dict[str, dict[str, Any]] = {}
        report = checker.check_objective(trades, instructions)
        assert len(report.evidence) > 0


# ===========================================================================
# 5. Cost Bias: unrealistic costs -> flag
# ===========================================================================


class TestCost:
    def test_all_trades_realistic_costs_no_bias(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=8.0,
            ),
            TradeRecord(
                trade_id="T-02", symbol="ETH/USDT", decision="SHORT",
                entry_price=3200.0, exit_price=3100.0, position_size=1.0,
                entry_time=datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 18, 0, 0, tzinfo=UTC),
                slippage_bps=12.0,
            ),
        ]
        report = checker.check_cost(trades)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_zero_slippage_flags_bias(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=0.0,
            ),
        ]
        report = checker.check_cost(trades)
        assert report.score > 0.0
        assert "T-01" in report.affected_trades

    def test_missing_slippage_attribute(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            ),
        ]
        del trades[0].slippage_bps
        report = checker.check_cost(trades)
        assert report.score > 0.0

    def test_non_zero_but_low_slippage_not_flagged(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=0.01,
            ),
        ]
        # 0.01 bps is non-zero, so the checker does not flag it
        report = checker.check_cost(trades)
        assert report.score == 0.0
        assert report.severity == BiasSeverity.NONE

    def test_some_missing_some_present(
        self, checker: BiasChecker,
    ) -> None:
        t1 = TradeRecord(
            trade_id="T-01", symbol="BTC/USDT", decision="LONG",
            entry_price=85000.0, exit_price=87000.0, position_size=0.1,
            entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
            exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
            slippage_bps=8.0,
        )
        t2 = TradeRecord(
            trade_id="T-02", symbol="ETH/USDT", decision="SHORT",
            entry_price=3200.0, exit_price=3100.0, position_size=1.0,
            entry_time=datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC),
            exit_time=datetime(2025, 6, 1, 18, 0, 0, tzinfo=UTC),
            slippage_bps=0.0,
        )
        report = checker.check_cost([t1, t2])
        assert report.score > 0.0
        assert "T-02" in report.affected_trades

    def test_empty_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_cost([])
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_none_trades_returns_clean(self, checker: BiasChecker) -> None:
        report = checker.check_cost(None)
        assert report.severity == BiasSeverity.NONE
        assert report.score == 0.0

    def test_cost_model_cross_validation(
        self, checker: BiasChecker,
    ) -> None:
        class FakeCostModel:
            @staticmethod
            def calculate_trade_cost(
                side, price, quantity, order_type,
            ):
                from pydantic import BaseModel

                class LegCost(BaseModel):
                    cost_bps: float = 10.0
                    total_cost: float = 0.0
                    gross_pnl: float = 0.0
                    commission: float = 0.0
                    spread_cost: float = 0.0
                    slippage_cost: float = 0.0
                    gas_fee: float = 0.0
                    net_pnl: float = 0.0

                return LegCost(
                    cost_bps=10.0, total_cost=price * quantity * 0.001
                )

        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=8.0,
            ),
        ]
        report = checker.check_cost(trades, FakeCostModel())
        assert report.score >= 0.0


# ===========================================================================
# 6. All clean data -> no biases flagged
# ===========================================================================


class TestAllClean:
    def test_run_all_with_clean_data(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
        clean_market_data: dict[str, Any],
    ) -> None:
        summary = checker.run_all(
            trades=clean_trades,
            market_data=clean_market_data,
            tickers_used=["BTC/USDT", "ETH/USDT"],
            tickers_available={
                "2025-01-01": ["BTC/USDT", "ETH/USDT"],
                "2025-06-01": ["BTC/USDT", "ETH/USDT"],
            },
            analyses={
                "T-01": [{"stance": "bullish", "risk_factors": [], "confidence": 0.7}],
                "T-02": [{"stance": "bearish", "risk_factors": [], "confidence": 0.6}],
            },
            trade_instructions={
                "T-01": {"objective": "total_return"},
                "T-02": {"objective": "total_return"},
            },
        )
        assert isinstance(summary, BiasAuditSummary)
        assert summary.passed
        assert summary.overall_score == 0.0
        assert len(summary.critical_biases) == 0
        assert len(summary.reports) == 5
        for r in summary.reports:
            assert r.severity == BiasSeverity.NONE


# ===========================================================================
# 7. Overall audit summary
# ===========================================================================


class TestAuditSummary:
    def test_run_all_with_biases(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=0.0,
            ),
        ]
        summary = checker.run_all(
            trades=trades,
            market_data={
                "BTC/USDT": {
                    "as_of": datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC),
                }
            },
        )
        assert not summary.passed
        assert summary.overall_score > 0.0
        assert len(summary.reports) == 5
        scores = [r.score for r in summary.reports]
        assert all(0.0 <= s <= 1.0 for s in scores)

    def test_report_types_in_summary(
        self, checker: BiasChecker,
    ) -> None:
        summary = checker.run_all()
        types = [r.bias_type for r in summary.reports]
        assert types == [
            "look_ahead", "survivorship", "narrative", "objective", "cost",
        ]

    def test_critical_biases_subset(
        self, checker: BiasChecker,
    ) -> None:
        trades = [
            TradeRecord(
                trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                slippage_bps=0.0,
            ),
        ]
        summary = checker.run_all(
            trades=trades,
            market_data={
                "BTC/USDT": {
                    "as_of": datetime(2025, 6, 1, 12, 0, 0, tzinfo=UTC),
                }
            },
        )
        for r in summary.critical_biases:
            assert r.severity in (BiasSeverity.HIGH, BiasSeverity.CRITICAL)

    def test_overall_score_is_average(
        self, checker: BiasChecker,
    ) -> None:
        summary = checker.run_all()
        expected = sum(r.score for r in summary.reports) / len(summary.reports)
        assert summary.overall_score == pytest.approx(expected)


# ===========================================================================
# 8. Missing data -> no crash, reports insufficient data
# ===========================================================================


class TestDefensive:
    def test_run_all_with_no_data(
        self, checker: BiasChecker,
    ) -> None:
        summary = checker.run_all()
        assert summary.passed
        assert summary.overall_score == 0.0
        assert isinstance(summary, BiasAuditSummary)

    def test_run_all_with_partial_data(
        self, checker: BiasChecker,
    ) -> None:
        summary = checker.run_all(
            trades=[
                TradeRecord(
                    trade_id="T-01", symbol="BTC/USDT", decision="LONG",
                    entry_price=85000.0, exit_price=87000.0, position_size=0.1,
                    entry_time=datetime(2025, 6, 1, 10, 0, 0, tzinfo=UTC),
                    exit_time=datetime(2025, 6, 1, 16, 0, 0, tzinfo=UTC),
                ),
            ],
            market_data=None,
            tickers_used=["BTC/USDT"],
            tickers_available=None,
        )
        assert isinstance(summary, BiasAuditSummary)
        assert len(summary.reports) == 5
        for r in summary.reports:
            assert isinstance(r, BiasReport)
            assert r.bias_type in (
                "look_ahead", "survivorship", "narrative", "objective", "cost",
            )

    def test_every_check_accepts_none(
        self, checker: BiasChecker,
    ) -> None:
        checker.check_look_ahead(None, None)
        checker.check_survivorship(None, None)
        checker.check_narrative(None, None)
        checker.check_objective(None, None)
        checker.check_cost(None)

    def test_every_check_with_empty_lists(
        self, checker: BiasChecker,
    ) -> None:
        checker.check_look_ahead([], {})
        checker.check_survivorship([], {})
        checker.check_narrative([], {})
        checker.check_objective([], {})
        checker.check_cost([])

    def test_missing_symbol_in_market_data(
        self, checker: BiasChecker, clean_trades: list[TradeRecord],
    ) -> None:
        md = {"OTHER": {"as_of": datetime(2025, 6, 1, 9, 0, 0, tzinfo=UTC)}}
        report = checker.check_look_ahead(clean_trades, md)
        assert report.severity == BiasSeverity.NONE


# ===========================================================================
# Data type tests
# ===========================================================================


class TestBiasReport:
    def test_default_severity_is_none(
        self,
    ) -> None:
        r = BiasReport(bias_type="look_ahead")
        assert r.severity == BiasSeverity.NONE
        assert r.score == 0.0
        assert r.evidence == []
        assert r.affected_trades == []

    def test_score_bounds(self) -> None:
        with pytest.raises(ValueError):
            BiasReport(bias_type="cost", score=-0.1)
        with pytest.raises(ValueError):
            BiasReport(bias_type="cost", score=1.5)


class TestBiasAuditSummary:
    def test_defaults(self) -> None:
        s = BiasAuditSummary()
        assert s.overall_score == 0.0
        assert s.passed is False
        assert s.reports == []
        assert s.critical_biases == []

    def test_passed_all_under_03(self) -> None:
        reports = [
            BiasReport(bias_type="la", score=0.1),
            BiasReport(bias_type="co", score=0.2),
            BiasReport(bias_type="na", score=0.0),
            BiasReport(bias_type="ob", score=0.05),
            BiasReport(bias_type="su", score=0.15),
        ]
        scores = [r.score for r in reports]
        s = BiasAuditSummary(
            reports=reports,
            overall_score=sum(scores) / len(scores),
            passed=all(sc < 0.3 for sc in scores),
        )
        assert s.passed
        assert s.overall_score < 0.3

    def test_failed_when_one_high(self) -> None:
        reports = [
            BiasReport(bias_type="la", score=0.5),
            BiasReport(bias_type="co", score=0.0),
            BiasReport(bias_type="na", score=0.0),
            BiasReport(bias_type="ob", score=0.0),
            BiasReport(bias_type="su", score=0.0),
        ]
        scores = [r.score for r in reports]
        s = BiasAuditSummary(
            reports=reports,
            overall_score=sum(scores) / len(scores),
            passed=False,
            critical_biases=[r for r in reports if r.score >= 0.3],
        )
        assert not s.passed
        assert len(s.critical_biases) == 1


class TestBiasSeverity:
    def test_values(self) -> None:
        assert BiasSeverity.NONE.value == "none"
        assert BiasSeverity.LOW.value == "low"
        assert BiasSeverity.MEDIUM.value == "medium"
        assert BiasSeverity.HIGH.value == "high"
        assert BiasSeverity.CRITICAL.value == "critical"


class TestHelpers:
    def test_extract_timestamp_from_dict_int(
        self, checker: BiasChecker,
    ) -> None:
        ts = checker._extract_timestamp({"timestamp": 1700000000000})
        assert ts == 1700000000000

    def test_extract_timestamp_from_dict_datetime(
        self, checker: BiasChecker,
    ) -> None:
        dt = datetime(2025, 1, 1, tzinfo=UTC)
        ts = checker._extract_timestamp({"timestamp": dt})
        assert ts is not None
        assert isinstance(ts, int)

    def test_extract_timestamp_missing_key(
        self, checker: BiasChecker,
    ) -> None:
        ts = checker._extract_timestamp({})
        assert ts is None

    def test_extract_timestamp_object_with_attr(
        self, checker: BiasChecker,
    ) -> None:
        class Obj:
            timestamp = 1700000000000

        ts = checker._extract_timestamp(Obj())
        assert ts == 1700000000000

    def test_ratio_to_severity_thresholds(
        self, checker: BiasChecker,
    ) -> None:
        assert checker._ratio_to_severity(0.0) == BiasSeverity.NONE
        assert checker._ratio_to_severity(0.02) == BiasSeverity.LOW
        assert checker._ratio_to_severity(0.10) == BiasSeverity.MEDIUM
        assert checker._ratio_to_severity(0.20) == BiasSeverity.HIGH
        assert checker._ratio_to_severity(0.50) == BiasSeverity.CRITICAL

    def test_score_to_severity_thresholds(
        self, checker: BiasChecker,
    ) -> None:
        assert checker._score_to_severity(0.0) == BiasSeverity.NONE
        assert checker._score_to_severity(0.10) == BiasSeverity.LOW
        assert checker._score_to_severity(0.25) == BiasSeverity.MEDIUM
        assert checker._score_to_severity(0.45) == BiasSeverity.HIGH
        assert checker._score_to_severity(0.75) == BiasSeverity.CRITICAL
