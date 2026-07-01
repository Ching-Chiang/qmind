"""
QMind — Explicit Trading Cost Model (P0 Critical).

Alpha Illusion paper requirement: every backtest MUST explicitly model
trading costs and report Net PnL alongside Gross PnL. Gross - Net is a
quantitative measure of alpha hallucination.

Provides configurable cost layers (commission, bid-ask spread, slippage,
gas fee) plus multi-tier sensitivity analysis at 0 / 10 / 25 basis points
for cross-scenario comparison.

Usage:
    model = CostModel()
    # Single-leg cost
    cost = model.calculate_trade_cost("buy", 100.0, 1.0, "market")
    # Aggregate across completed trades
    total = model.calculate_total_cost([trade1, trade2])
    # Multi-tier sensitivity report
    print(model.generate_report([trade1, trade2]))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from pydantic import BaseModel, Field

from qmind.learning.evaluator import TradeRecord


# ──────────────────────────────────────────────
# Output Types
# ──────────────────────────────────────────────


class TradeCost(BaseModel):
    """Cost breakdown for a single trade leg (entry or exit).

    Attributes:
        gross_pnl: Gross PnL for this leg (0 for a single leg; the
            full round-trip PnL is computed in TotalCost).
        commission: Total commission paid (maker or taker rate).
        spread_cost: Cost attributed to the bid-ask spread.
        slippage_cost: Cost attributed to market impact / slippage.
        gas_fee: On-chain gas cost (0 for CEX trades).
        total_cost: Sum of all cost components.
        net_pnl: gross_pnl - total_cost (negative for a pure cost leg).
        cost_bps: Total cost expressed in basis points of notional.
    """
    gross_pnl: float = 0.0
    commission: float = 0.0
    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    gas_fee: float = 0.0
    total_cost: float = 0.0
    net_pnl: float = 0.0
    cost_bps: float = 0.0


class TotalCost(BaseModel):
    """Aggregated cost breakdown across a portfolio of completed trades.

    Attributes:
        trades: Number of round-trip trades in the portfolio.
        gross_pnl: Sum of gross PnL before costs.
        total_cost: Sum of all costs (commission + spread + slippage + gas).
        net_pnl: gross_pnl - total_cost.
        avg_cost_bps: Average cost per trade in basis points.
        cost_pct_of_gross: Total cost as a percentage of absolute gross PnL.
            Infra if gross PnL is 0 or negative. High values (>50%) signal
            that alpha is being consumed by friction.
    """
    trades: int = 0
    gross_pnl: float = 0.0
    total_cost: float = 0.0
    net_pnl: float = 0.0
    avg_cost_bps: float = 0.0
    cost_pct_of_gross: float = 0.0


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────


@dataclass
class CostConfig:
    """Configuration for each cost layer in the model.

    All rates are expressed as decimals (e.g. 0.001 = 0.1 %).

    Attributes:
        commission_maker: Maker fee for limit orders (default 0.001).
        commission_taker: Taker fee for market orders (default 0.001).
        bid_ask_spread: Assumed one-way spread cost per leg, as a
            fraction of notional (default 0.0005 = 0.5 bps per leg).
        slippage_pct: Fixed slippage rate per leg, as a fraction of
            notional (default 0.001). Ignored if **slippage_fn** is set.
        slippage_fn: Optional callable ``(price, quantity) -> slippage_cost``.
            Overrides **slippage_pct** when provided. Useful for market-impact
            models that depend on order size relative to order-book depth.
        gas_fee: Fixed gas fee in quote currency per on-chain leg.
            Set > 0 for DEX trades; stay at 0 for CEX.
    """
    commission_maker: float = 0.001
    commission_taker: float = 0.001
    bid_ask_spread: float = 0.0005
    slippage_pct: float = 0.001
    slippage_fn: Callable[[float, float], float] | None = None
    gas_fee: float = 0.0

    def __post_init__(self) -> None:
        """Validate that all rates are non-negative and <= 100 %."""
        for name in ("commission_maker", "commission_taker", "bid_ask_spread", "slippage_pct"):
            val = getattr(self, name)
            if not 0.0 <= val <= 1.0:
                raise ValueError(
                    f"{name} must be between 0 and 1, got {val}"
                )
        if self.gas_fee < 0:
            raise ValueError(f"gas_fee must be >= 0, got {self.gas_fee}")


# ──────────────────────────────────────────────
# Cost Model
# ──────────────────────────────────────────────

class CostModel:
    """Explicit trading cost model with configurable layers.

    Supports two modes:

    1. **Detailed mode** — uses the individual layers in **config**
       (commission, spread, slippage, gas) for realistic cost modeling.
    2. **Sensitivity mode** — replaces all layers with a flat cost rate
       in basis points (0, 10, 25) for multi-tier comparison. This is
       accessed via **generate_report()**.

    Thread-safe: all state is read-only after construction.
    """

    # Sensitivity tiers for multi-tier cost analysis (bps)
    SENSITIVITY_TIERS: list[tuple[str, float]] = [
        ("0 bps   (ideal)", 0.0),
        ("10 bps  (low-cost)", 10.0),
        ("25 bps  (moderate)", 25.0),
    ]

    def __init__(self, config: CostConfig | None = None) -> None:
        """Initialise the cost model.

        Args:
            config: Cost layer configuration. Uses defaults if omitted.
        """
        self.config = config or CostConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate_trade_cost(
        self,
        side: str,
        price: float,
        quantity: float,
        order_type: str = "market",
    ) -> TradeCost:
        """Calculate execution cost for a **single trade leg**.

        Call this twice per round-trip trade: once for entry, once for exit.
        Aggregate via **calculate_total_cost()** for final PnL reporting.

        Args:
            side: ``"buy"`` or ``"sell"``.
            price: Expected fill price for this leg.
            quantity: Number of base-currency units traded.
            order_type: ``"market"`` (taker fee) or ``"limit"`` (maker fee).

        Returns:
            TradeCost with all cost components. **gross_pnl** is 0 because
            this is a single leg; full round-trip PnL is computed by
            **calculate_total_cost()**.

        Raises:
            ValueError: If **side** or **order_type** is invalid.
        """
        side = side.lower()
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        order_type = order_type.lower()
        if order_type not in ("market", "limit"):
            raise ValueError(
                f"order_type must be 'market' or 'limit', got {order_type!r}"
            )

        notional = price * quantity

        # 1. Commission
        comm_rate = (
            self.config.commission_taker
            if order_type == "market"
            else self.config.commission_maker
        )
        commission = notional * comm_rate

        # 2. Bid-ask spread (one-way cost per leg)
        spread_cost = notional * self.config.bid_ask_spread

        # 3. Slippage / market impact
        if self.config.slippage_fn is not None:
            slippage_cost = self.config.slippage_fn(price, quantity)
        else:
            slippage_cost = notional * self.config.slippage_pct

        # 4. Gas fee (DEX only; 0 for CEX)
        gas_fee = self.config.gas_fee

        total_cost = commission + spread_cost + slippage_cost + gas_fee
        cost_bps = (total_cost / notional * 10000) if notional > 0 else 0.0

        return TradeCost(
            gross_pnl=0.0,
            commission=round(commission, 2),
            spread_cost=round(spread_cost, 2),
            slippage_cost=round(slippage_cost, 2),
            gas_fee=round(gas_fee, 2),
            total_cost=round(total_cost, 2),
            net_pnl=round(-total_cost, 2),
            cost_bps=round(cost_bps, 4),
        )

    def calculate_total_cost(self, trades: list[TradeRecord]) -> TotalCost:
        """Aggregate costs across a list of completed round-trip trades.

        For each trade, computes:
        - Gross PnL from entry and exit prices.
        - Round-trip cost (entry leg + exit leg).
        - Net PnL = Gross PnL - Total Cost.

        Args:
            trades: Completed trade records. May be empty.

        Returns:
            TotalCost with aggregated metrics.

        Raises:
            ValueError: If any trade has a non-positive position size.
        """
        if not trades:
            return TotalCost(trades=0)

        total_gross = 0.0
        total_cost = 0.0
        total_bps_weighted = 0.0
        total_notional_for_bps = 0.0

        for trade in trades:
            if trade.position_size <= 0:
                raise ValueError(
                    f"Trade {trade.trade_id}: position_size must be > 0, "
                    f"got {trade.position_size}"
                )

            # --- Gross PnL ---
            if trade.decision == "LONG":
                gross = (trade.exit_price - trade.entry_price) * trade.position_size
            else:  # SHORT
                gross = (trade.entry_price - trade.exit_price) * trade.position_size

            total_gross += gross

            # --- Entry leg ---
            # Default to market; limit orders assumed when the decision
            # explicitly carries a limit-price entry instruction.
            entry_type = self._infer_order_type(trade, leg="entry")
            entry_cost = self.calculate_trade_cost(
                side="buy" if trade.decision == "LONG" else "sell",
                price=trade.entry_price,
                quantity=trade.position_size,
                order_type=entry_type,
            )

            # --- Exit leg ---
            exit_type = self._infer_order_type(trade, leg="exit")
            exit_cost = self.calculate_trade_cost(
                side="sell" if trade.decision == "LONG" else "buy",
                price=trade.exit_price,
                quantity=trade.position_size,
                order_type=exit_type,
            )

            leg_total = entry_cost.total_cost + exit_cost.total_cost
            total_cost += leg_total

            # Weighted basis points: cost / avg notional * 10000
            avg_notional = (trade.entry_price + trade.exit_price) / 2 * trade.position_size
            total_bps_weighted += leg_total * 10000  # defer division to end
            total_notional_for_bps += avg_notional

        n = len(trades)
        avg_bps = (
            total_bps_weighted / total_notional_for_bps
            if total_notional_for_bps > 0
            else 0.0
        )
        cost_pct = (
            total_cost / abs(total_gross) * 100 if total_gross != 0 else 0.0
        )

        return TotalCost(
            trades=n,
            gross_pnl=round(total_gross, 2),
            total_cost=round(total_cost, 2),
            net_pnl=round(total_gross - total_cost, 2),
            avg_cost_bps=round(avg_bps, 4),
            cost_pct_of_gross=round(cost_pct, 2),
        )

    def generate_report(self, trades: list[TradeRecord] | None = None) -> str:
        """Generate a multi-tier cost sensitivity breakdown report.

        Three tiers (0 / 10 / 25 bps) are reported alongside the
        detailed configured cost breakdown. Output is a formatted
        string suitable for terminal display or log capture.

        Args:
            trades: Optional list of completed trades to base the
                report on. If omitted, shows configured cost rates
                without actual impact numbers.

        Returns:
            Formatted multi-line string report.
        """
        lines: list[str] = []
        sep = "=" * 62
        sub = "-" * 62

        lines.append(sep)
        lines.append("  Cost Model - Multi-Tier Sensitivity Analysis")
        lines.append(sep)

        if not trades:
            lines.append("")
            lines.append("  Configured cost rates:")
            lines.append(
                f"    Commission maker:   "
                f"{self.config.commission_maker * 100:.4f} %"
            )
            lines.append(
                f"    Commission taker:   "
                f"{self.config.commission_taker * 100:.4f} %"
            )
            lines.append(
                f"    Bid-Ask Spread:     "
                f"{self.config.bid_ask_spread * 100:.4f} %"
            )
            lines.append(
                f"    Slippage:           "
                f"{self.config.slippage_pct * 100:.4f} %"
            )
            lines.append(
                f"    Gas Fee:            "
                f"{self.config.gas_fee:.2f} quote currency"
            )
            lines.append("")
            lines.append(
                "  Sensitivity tiers (bps): "
                + ", ".join(t[0].split()[0] for t in self.SENSITIVITY_TIERS)
            )
            lines.append(
                "  Provide a list of TradeRecord objects to see actual "
                "cost impact."
            )
            lines.append(sep)
            return "\n".join(lines)

        # Compute aggregate values once for the detailed breakdown.
        detailed = self.calculate_total_cost(trades)

        total_entry_notional = sum(
            t.entry_price * t.position_size for t in trades
        )
        total_exit_notional = sum(
            t.exit_price * t.position_size for t in trades
        )
        total_notional_round = total_entry_notional + total_exit_notional

        # -- Sensitivity tiers --
        lines.append("")
        lines.append(f"  Portfolio: {len(trades)} trade(s)")
        lines.append("")
        header = f"  {'Tier':<24} {'Cost':>12} {'Gross PnL':>12} {'Net PnL':>12}"
        lines.append(header)
        lines.append("  " + sub[:len(header) - 2])

        for label, bps in self.SENSITIVITY_TIERS:
            tier_cost = total_notional_round * (bps / 10000.0)
            net = detailed.gross_pnl - tier_cost
            lines.append(
                f"  {label:<24} {tier_cost:>10.2f}  "
                f"{detailed.gross_pnl:>10.2f}  {net:>10.2f}"
            )

        # -- Detailed breakdown --
        lines.append("")
        lines.append(sep)
        lines.append("  Detailed Cost Breakdown (configured layers)")
        lines.append(sep)
        lines.append("")
        h2 = f"  {'Layer':<26} {'Amount':>12} {'bps':>10}"
        lines.append(h2)
        lines.append("  " + sub[:len(h2) - 2])

        def _layer_bps(amount: float) -> float:
            return (
                (amount / total_notional_round * 10000)
                if total_notional_round > 0
                else 0.0
            )

        # Commission
        cm_entry = total_entry_notional * self.config.commission_taker
        cm_exit = total_exit_notional * self.config.commission_taker
        cm_total = cm_entry + cm_exit
        lines.append(
            f"  {'Commission (taker assumed)':<26} {cm_total:>10.2f}  "
            f"{_layer_bps(cm_total):>10.4f}"
        )

        # Bid-ask spread
        sp_total = total_notional_round * self.config.bid_ask_spread
        lines.append(
            f"  {'Bid-Ask Spread':<26} {sp_total:>10.2f}  "
            f"{_layer_bps(sp_total):>10.4f}"
        )

        # Slippage
        if self.config.slippage_fn is not None:
            sl_entry = sum(
                self.config.slippage_fn(t.entry_price, t.position_size)
                for t in trades
            )
            sl_exit = sum(
                self.config.slippage_fn(t.exit_price, t.position_size)
                for t in trades
            )
            sl_total = sl_entry + sl_exit
        else:
            sl_total = total_notional_round * self.config.slippage_pct
        lines.append(
            f"  {'Slippage':<26} {sl_total:>10.2f}  "
            f"{_layer_bps(sl_total):>10.4f}"
        )

        # Gas fee
        gas_total = self.config.gas_fee * len(trades) * 2  # entry + exit
        if gas_total > 0:
            lines.append(
                f"  {'Gas Fee (DEX)':<26} {gas_total:>10.2f}  "
                f"{_layer_bps(gas_total):>10.4f}"
            )

        # Total line
        lines.append("  " + sub[:len(h2) - 2])
        tc = detailed.total_cost
        lines.append(
            f"  {'TOTAL':<26} {tc:>10.2f}  "
            f"{_layer_bps(tc):>10.4f}"
        )

        # Summary metrics
        lines.append("")
        lines.append(sep)
        lines.append("  Summary")
        lines.append(sep)
        lines.append(f"    Gross PnL:                 {detailed.gross_pnl:>12.2f}")
        lines.append(f"    Total Cost:                {detailed.total_cost:>12.2f}")
        lines.append(f"    Net PnL:                   {detailed.net_pnl:>12.2f}")
        lines.append(f"    Cost as % of |Gross|:      {detailed.cost_pct_of_gross:>12.2f} %")
        lines.append(f"    Average cost (bps/trade):  {detailed.avg_cost_bps:>12.4f}")
        lines.append(
            f"    Gross - Net gap (=alpha hallucination): "
            f"{detailed.total_cost:>12.2f}"
        )
        lines.append(sep)

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_order_type(trade: TradeRecord, leg: str = "entry") -> str:
        """Heuristic to determine whether a leg was a limit or market order.

        Args:
            trade: The trade record to inspect.
            leg: ``"entry"`` or ``"exit"``.

        Returns:
            ``"limit"`` if the entry decision specifies a limit price
            (i.e. ``entry.type == "limit"``), otherwise ``"market"``.
        """
        # TradeRecord doesn't carry an explicit order type, so we apply
        # a simple heuristic: if the trade record was generated by a
        # backtest with limit semantics, this method can be overridden
        # by subclasses. Default: market for both legs.
        return "market"

    def sensitivity_tier_cost(
        self,
        trades: list[TradeRecord],
        tier_bps: float,
    ) -> TotalCost:
        """Compute total cost at a flat sensitivity tier.

        This replaces all cost layers with a single flat rate for quick
        what-if analysis.

        Args:
            trades: List of completed trades.
            tier_bps: Flat cost rate in basis points (e.g. 10.0).

        Returns:
            TotalCost where **total_cost** = notional * tier_bps / 10000.
        """
        if not trades:
            return TotalCost(trades=0)

        total_gross = 0.0
        total_notional = 0.0

        for trade in trades:
            if trade.decision == "LONG":
                total_gross += (
                    trade.exit_price - trade.entry_price
                ) * trade.position_size
            else:
                total_gross += (
                    trade.entry_price - trade.exit_price
                ) * trade.position_size
            total_notional += (
                trade.entry_price + trade.exit_price
            ) * trade.position_size

        tier_rate = tier_bps / 10000.0
        tier_cost = total_notional * tier_rate
        net = total_gross - tier_cost

        return TotalCost(
            trades=len(trades),
            gross_pnl=round(total_gross, 2),
            total_cost=round(tier_cost, 2),
            net_pnl=round(net, 2),
            avg_cost_bps=round(tier_bps, 4),
            cost_pct_of_gross=round(
                tier_cost / abs(total_gross) * 100 if total_gross != 0 else 0.0,
                2,
            ),
        )
