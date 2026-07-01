"""backtest/cost_model.py 单元测试

Covers:
- Basic trade cost calculation (long, short)
- Commission: maker/taker different rates
- Bid-ask spread added correctly
- Slippage: percentage and fixed models
- Multi-tier cost report (0/10/25 bps)
- Gas fee (for DEX)
- Net PnL = Gross PnL - Total Cost
- Cost in basis points
- Edge cases: zero price, zero quantity, negative values
- Calculator handles empty trade list
"""

from __future__ import annotations

from datetime import datetime

import pytest

from qmind.backtest.cost_model import CostConfig, CostModel, TotalCost, TradeCost
from qmind.learning.evaluator import TradeRecord


# ── Helpers ──

_LONG_TRADE = TradeRecord(
    trade_id="t1",
    symbol="BTC/USDT",
    decision="LONG",
    entry_price=100.0,
    exit_price=110.0,
    position_size=2.0,
    entry_time=datetime(2026, 6, 1, 10, 0, 0),
    exit_time=datetime(2026, 6, 1, 14, 0, 0),
)

_SHORT_TRADE = TradeRecord(
    trade_id="t2",
    symbol="BTC/USDT",
    decision="SHORT",
    entry_price=110.0,
    exit_price=100.0,
    position_size=2.0,
    entry_time=datetime(2026, 6, 1, 10, 0, 0),
    exit_time=datetime(2026, 6, 1, 14, 0, 0),
)

_LOSS_TRADE = TradeRecord(
    trade_id="t3",
    symbol="BTC/USDT",
    decision="LONG",
    entry_price=100.0,
    exit_price=95.0,
    position_size=2.0,
    entry_time=datetime(2026, 6, 1, 10, 0, 0),
    exit_time=datetime(2026, 6, 1, 14, 0, 0),
)


def default_model() -> CostModel:
    return CostModel()


# ── Test suite ──


class TestTradeCostBasic:
    """1. Basic trade cost calculation (single leg)."""

    def test_buy_market_cost(self):
        model = default_model()
        cost = model.calculate_trade_cost("buy", 100.0, 2.0, "market")
        notional = 200.0
        expected_commission = notional * 0.001  # 0.20
        expected_spread = notional * 0.0005  # 0.10
        expected_slippage = notional * 0.001  # 0.20
        expected_total = round(expected_commission + expected_spread + expected_slippage, 2)
        expected_bps = expected_total / notional * 10000

        assert cost.gross_pnl == 0.0
        assert cost.commission == pytest.approx(expected_commission, 0.01)
        assert cost.spread_cost == pytest.approx(expected_spread, 0.01)
        assert cost.slippage_cost == pytest.approx(expected_slippage, 0.01)
        assert cost.gas_fee == 0.0
        assert cost.total_cost == pytest.approx(expected_total, 0.01)
        assert cost.net_pnl == pytest.approx(-expected_total, 0.01)
        assert cost.cost_bps == pytest.approx(expected_bps, 0.01)

    def test_sell_market_cost(self):
        model = default_model()
        cost = model.calculate_trade_cost("sell", 110.0, 2.0, "market")
        assert cost.total_cost > 0
        assert cost.gross_pnl == 0.0
        assert cost.net_pnl < 0

    def test_buy_limit_cost(self):
        """Limit orders use maker fee (lower)."""
        config = CostConfig(commission_maker=0.0005, commission_taker=0.002)
        model = CostModel(config)
        cost_market = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        cost_limit = model.calculate_trade_cost("buy", 100.0, 1.0, "limit")
        assert cost_limit.commission < cost_market.commission
        assert cost_limit.commission == pytest.approx(100.0 * 0.0005, 0.01)

    def test_invalid_side_raises(self):
        model = default_model()
        with pytest.raises(ValueError, match="side must be 'buy' or 'sell'"):
            model.calculate_trade_cost("hold", 100.0, 1.0)

    def test_invalid_order_type_raises(self):
        model = default_model()
        with pytest.raises(ValueError, match="order_type must be 'market' or 'limit'"):
            model.calculate_trade_cost("buy", 100.0, 1.0, "stop")


class TestCommission:
    """2. Commission: maker/taker different rates."""

    def test_maker_vs_taker(self):
        config = CostConfig(commission_maker=0.0002, commission_taker=0.001)
        model = CostModel(config)
        maker = model.calculate_trade_cost("buy", 100.0, 1.0, "limit")
        taker = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert maker.commission == 0.02
        assert taker.commission == 0.10

    def test_zero_commission(self):
        config = CostConfig(commission_maker=0.0, commission_taker=0.0)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.commission == 0.0


class TestBidAskSpread:
    """3. Bid-ask spread added correctly."""

    def test_spread_applied(self):
        config = CostConfig(bid_ask_spread=0.001)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.spread_cost == 0.10  # 100 * 0.001

    def test_zero_spread(self):
        config = CostConfig(bid_ask_spread=0.0)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.spread_cost == 0.0


class TestSlippage:
    """4. Slippage: percentage and fixed models."""

    def test_percentage_slippage(self):
        config = CostConfig(slippage_pct=0.002)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.slippage_cost == 0.20  # 100 * 0.002

    def test_custom_slippage_fn(self):
        def custom_slippage(price: float, qty: float) -> float:
            return price * qty * 0.01 + 5.0  # 1% + fixed 5

        config = CostConfig(slippage_fn=custom_slippage)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 2.0, "market")
        expected = 100.0 * 2.0 * 0.01 + 5.0  # 7.0
        assert cost.slippage_cost == pytest.approx(expected, 0.01)
        # slippage_pct should be ignored when slippage_fn is set
        assert cost.slippage_cost != 200.0 * 0.001

    def test_zero_slippage(self):
        config = CostConfig(slippage_pct=0.0)
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.slippage_cost == 0.0


class TestGasFee:
    """5. Gas fee (for DEX)."""

    def test_gas_fee_applied(self):
        config = CostConfig(gas_fee=0.50)  # 0.50 quote currency per leg
        model = CostModel(config)
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.gas_fee == 0.50
        assert cost.total_cost > cost.commission + cost.spread_cost + cost.slippage_cost

    def test_gas_fee_zero_by_default(self):
        cost = default_model().calculate_trade_cost("buy", 100.0, 1.0, "market")
        assert cost.gas_fee == 0.0


class TestCalculateTotalCost:
    """6. Aggregate across trades (round-trip)."""

    def test_long_profit_total(self):
        model = default_model()
        total = model.calculate_total_cost([_LONG_TRADE])
        # Gross: (110 - 100) * 2 = 20
        assert total.gross_pnl == pytest.approx(20.0, 0.01)
        assert total.total_cost > 0
        assert total.net_pnl == pytest.approx(total.gross_pnl - total.total_cost, 0.01)
        assert total.trades == 1

    def test_short_profit_total(self):
        model = default_model()
        total = model.calculate_total_cost([_SHORT_TRADE])
        # Gross: (110 - 100) * 2 = 20
        assert total.gross_pnl == pytest.approx(20.0, 0.01)
        assert total.total_cost > 0
        assert total.net_pnl == pytest.approx(total.gross_pnl - total.total_cost, 0.01)

    def test_net_pnl_is_gross_minus_cost(self):
        """Net PnL = Gross PnL - Total Cost."""
        model = default_model()
        total = model.calculate_total_cost([_LONG_TRADE])
        assert total.net_pnl == pytest.approx(total.gross_pnl - total.total_cost, 0.01)

    def test_net_pnl_loss_bigger_than_gross(self):
        """Loss trade: net is even more negative after costs."""
        model = default_model()
        total = model.calculate_total_cost([_LOSS_TRADE])
        # Gross: (95 - 100) * 2 = -10
        assert total.gross_pnl == pytest.approx(-10.0, 0.01)
        assert total.total_cost > 0
        assert total.net_pnl < total.gross_pnl  # more negative

    def test_multiple_trades_aggregate(self):
        model = default_model()
        trades = [_LONG_TRADE, _SHORT_TRADE]
        total = model.calculate_total_cost(trades)
        assert total.trades == 2
        assert total.gross_pnl == pytest.approx(40.0, 0.01)  # 20 + 20
        assert total.total_cost > 0
        assert total.net_pnl < total.gross_pnl

    def test_empty_trade_list(self):
        model = default_model()
        total = model.calculate_total_cost([])
        assert total.trades == 0
        assert total.gross_pnl == 0.0
        assert total.total_cost == 0.0
        assert total.net_pnl == 0.0
        assert total.avg_cost_bps == 0.0
        assert total.cost_pct_of_gross == 0.0


class TestInvalidInputs:
    """7. Edge cases and invalid inputs."""

    def test_zero_position_size_raises(self):
        bad = TradeRecord(
            trade_id="bad",
            symbol="BTC/USDT",
            decision="LONG",
            entry_price=100.0,
            exit_price=110.0,
            position_size=0,
            entry_time=datetime(2026, 6, 1, 10, 0, 0),
            exit_time=datetime(2026, 6, 1, 14, 0, 0),
        )
        model = default_model()
        with pytest.raises(ValueError, match="position_size must be > 0"):
            model.calculate_total_cost([bad])

    def test_negative_position_size_raises(self):
        bad = TradeRecord(
            trade_id="bad",
            symbol="BTC/USDT",
            decision="LONG",
            entry_price=100.0,
            exit_price=110.0,
            position_size=-1.0,
            entry_time=datetime(2026, 6, 1, 10, 0, 0),
            exit_time=datetime(2026, 6, 1, 14, 0, 0),
        )
        model = default_model()
        with pytest.raises(ValueError, match="position_size must be > 0"):
            model.calculate_total_cost([bad])

    def test_zero_price_single_leg(self):
        """Zero price should not crash; cost bps should be 0 since notional is 0."""
        model = default_model()
        cost = model.calculate_trade_cost("buy", 0.0, 1.0, "market")
        assert cost.commission == 0.0
        assert cost.spread_cost == 0.0
        assert cost.slippage_cost == 0.0
        assert cost.total_cost == 0.0
        assert cost.cost_bps == 0.0

    def test_zero_quantity_single_leg(self):
        """Zero quantity should not crash."""
        model = default_model()
        cost = model.calculate_trade_cost("buy", 100.0, 0.0, "market")
        assert cost.commission == 0.0
        assert cost.total_cost == 0.0
        assert cost.cost_bps == 0.0

    def test_negative_price(self):
        """Negative price should still compute; notional is negative."""
        model = default_model()
        cost = model.calculate_trade_cost("buy", -10.0, 1.0, "market")
        # notional = -10, commission = -10 * 0.001 = -0.01, etc.
        assert cost.commission < 0  # mathematically correct
        assert cost.total_cost < 0


class TestCostConfigValidation:
    """8. CostConfig validation."""

    def test_negative_commission_raises(self):
        with pytest.raises(ValueError, match="commission_maker"):
            CostConfig(commission_maker=-0.001)

    def test_commission_over_100_pct_raises(self):
        with pytest.raises(ValueError, match="commission_maker"):
            CostConfig(commission_maker=1.5)

    def test_negative_spread_raises(self):
        with pytest.raises(ValueError, match="bid_ask_spread"):
            CostConfig(bid_ask_spread=-0.001)

    def test_negative_gas_fee_raises(self):
        with pytest.raises(ValueError, match="gas_fee"):
            CostConfig(gas_fee=-1.0)

    def test_boundary_values_accepted(self):
        """Exactly 0 and 1 are valid rates."""
        config = CostConfig(commission_maker=0.0, commission_taker=1.0,
                            bid_ask_spread=0.0, slippage_pct=1.0)
        assert config.commission_maker == 0.0
        assert config.commission_taker == 1.0


class TestCostInBasisPoints:
    """9. Cost in basis points."""

    def test_single_leg_bps(self):
        model = default_model()
        cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
        # Commission 0.10 + spread 0.05 + slippage 0.10 = 0.25 on 100 notional
        # 0.25 / 100 * 10000 = 25 bps
        assert cost.cost_bps == pytest.approx(25.0, 0.1)

    def test_aggregated_avg_bps(self):
        """avg_cost_bps is weighted by notional."""
        model = default_model()
        total = model.calculate_total_cost([_LONG_TRADE])
        assert total.avg_cost_bps > 0

    def test_cost_pct_of_gross(self):
        """cost_pct_of_gross should be total_cost / abs(gross_pnl) * 100."""
        model = default_model()
        total = model.calculate_total_cost([_LONG_TRADE])
        if total.gross_pnl != 0:
            expected_pct = total.total_cost / abs(total.gross_pnl) * 100
            assert total.cost_pct_of_gross == pytest.approx(expected_pct, 0.01)


class TestSensitivityTier:
    """10. sensitivity_tier_cost method."""

    def test_tier_10_bps(self):
        model = default_model()
        total = model.sensitivity_tier_cost([_LONG_TRADE], tier_bps=10.0)
        # Notional: (100 + 110) * 2 = 420
        # Cost: 420 * 10 / 10000 = 0.42
        assert total.total_cost == pytest.approx(0.42, 0.01)
        assert total.trades == 1
        assert total.gross_pnl == pytest.approx(20.0, 0.01)

    def test_tier_0_bps(self):
        model = default_model()
        total = model.sensitivity_tier_cost([_LONG_TRADE], tier_bps=0.0)
        assert total.total_cost == 0.0
        assert total.net_pnl == total.gross_pnl

    def test_tier_25_bps(self):
        model = default_model()
        total = model.sensitivity_tier_cost([_LONG_TRADE], tier_bps=25.0)
        # Notional: 420, Cost: 420 * 25 / 10000 = 1.05
        assert total.total_cost == pytest.approx(1.05, 0.01)

    def test_tier_empty(self):
        model = default_model()
        total = model.sensitivity_tier_cost([], tier_bps=10.0)
        assert total.trades == 0
        assert total.gross_pnl == 0.0

    def test_tier_avg_cost_bps_equals_tier(self):
        model = default_model()
        total = model.sensitivity_tier_cost([_LONG_TRADE], tier_bps=10.0)
        assert total.avg_cost_bps == 10.0


class TestMultiTierReport:
    """11. Multi-tier cost report (generate_report)."""

    def test_report_without_trades(self):
        model = default_model()
        report = model.generate_report()
        assert "Configured cost rates" in report
        assert "Provide a list of TradeRecord" in report

    def test_report_with_trades(self):
        model = default_model()
        report = model.generate_report([_LONG_TRADE])
        assert "Portfolio: 1 trade(s)" in report
        assert "0 bps" in report or "0 bps" in report
        assert "Detailed Cost Breakdown" in report
        assert "Gross PnL" in report
        assert "Net PnL" in report
        assert "Summary" in report

    def test_report_tiers_present(self):
        """Report should include all three sensitivity tiers."""
        model = default_model()
        report = model.generate_report([_LONG_TRADE, _SHORT_TRADE])
        for label, _ in model.SENSITIVITY_TIERS:
            assert label in report

    def test_report_empty_trades_no_crash(self):
        model = default_model()
        # Empty list is falsy but not None — should show "no trades" path
        report = model.generate_report([])
        assert "Configured cost rates" in report


class TestNetPnLMath:
    """12. Net PnL = Gross PnL - Total Cost."""

    def test_net_pnl_formula_holds(self):
        model = default_model()
        for trades in ([_LONG_TRADE], [_SHORT_TRADE], [_LOSS_TRADE]):
            total = model.calculate_total_cost(trades)
            assert total.net_pnl == pytest.approx(total.gross_pnl - total.total_cost, 0.01)

    def test_net_pnl_with_gas(self):
        config = CostConfig(gas_fee=0.30)
        model = CostModel(config)
        total = model.calculate_total_cost([_LONG_TRADE])
        # Gas: 0.30 * 2 legs = 0.60
        assert total.net_pnl == pytest.approx(total.gross_pnl - total.total_cost, 0.01)


class TestInferOrderType:
    """13. _infer_order_type helper."""

    def test_default_is_market(self):
        assert CostModel._infer_order_type(_LONG_TRADE, "entry") == "market"
        assert CostModel._infer_order_type(_LONG_TRADE, "exit") == "market"

    def test_both_legs_market(self):
        assert CostModel._infer_order_type(_SHORT_TRADE) == "market"


class TestGasFeeInTotalCost:
    """14. Gas fee multiplies by number of legs in total cost."""

    def test_gas_entry_and_exit(self):
        config = CostConfig(gas_fee=0.50)
        model = CostModel(config)
        total = model.calculate_total_cost([_LONG_TRADE])
        # Entry leg gas: 0.50, Exit leg gas: 0.50
        # Minimal commission/spread/slippage on top
        # The gas contributes 1.00 to total_cost
        entry = model.calculate_trade_cost("buy", _LONG_TRADE.entry_price, _LONG_TRADE.position_size)
        exit = model.calculate_trade_cost("sell", _LONG_TRADE.exit_price, _LONG_TRADE.position_size)
        expected_leg_total = entry.total_cost + exit.total_cost
        assert total.total_cost == pytest.approx(expected_leg_total, 0.01)

    def test_no_gas_fee_by_default(self):
        model = default_model()
        total = model.calculate_total_cost([_LONG_TRADE])
        # No gas fee component
        entry = model.calculate_trade_cost("buy", _LONG_TRADE.entry_price, _LONG_TRADE.position_size)
        exit = model.calculate_trade_cost("sell", _LONG_TRADE.exit_price, _LONG_TRADE.position_size)
        assert total.total_cost == pytest.approx(entry.total_cost + exit.total_cost, 0.01)
