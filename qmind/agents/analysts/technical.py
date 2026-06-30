"""
技术面分析师 — K线形态、指标信号、量价关系。

注入 CVRF 历史教训（如果存在）。
"""

from __future__ import annotations

from qmind.agents.analysts.base import BaseAnalyst
from qmind.agents.protocol import TechnicalReport
from qmind.graph.state import AnalystReport, MarketData
from qmind.tools.market_data import calculate_indicators

TECHNICAL_SYSTEM_PROMPT = """你是一位经验丰富的技术分析师，专精于图表形态和技术指标分析。

你的任务是分析 K 线数据和技术指标，输出结构化的技术面分析报告。

请关注:
1. 趋势方向与强度 (均线系统、ADX)
2. 关键支撑/阻力位 (前期高/低点、斐波那契位)
3. 动量信号 (RSI、MACD、随机指标)
4. 波动率状态 (布林带、ATR)
5. 成交量验证 (量价配合/背离)
6. 图表形态 (头肩顶、双底、旗形等)

⚠️ 重要: 只基于你看到的数据说话，不要臆测未来的价格走势。
输出严格符合 JSON Schema。"""


class TechnicalAnalyst(BaseAnalyst):
    """技术面分析师"""

    @property
    def analyst_name(self) -> str:
        return "technical"

    @property
    def system_prompt(self) -> str:
        return TECHNICAL_SYSTEM_PROMPT

    async def analyze(self, market_data: MarketData) -> AnalystReport:
        timeframe = list(market_data.klines.keys())[0] if market_data.klines else "1h"
        klines = market_data.klines.get(timeframe, [])
        indicators = calculate_indicators(market_data) if klines else {}

        def fmt_ma(key: str, label: str = "") -> str:
            v = indicators.get(key)
            if v is None:
                return f"{label or key}: N/A"
            return f"{label or key}: {v:.2f}"

        current_price = klines[-1].close if klines else 0
        support = min((k.low for k in klines[-20:]), default=0) if len(klines) >= 20 else 0
        resistance = max((k.high for k in klines[-20:]), default=0) if len(klines) >= 20 else 0

        prompt = (
            f"# 技术面分析\n\n"
            f"标的: {market_data.symbol} | 时间框架: {timeframe}\n"
            f"当前价格: {current_price:.2f}\n"
            f"20期支撑: {support:.2f} | 20期阻力: {resistance:.2f}\n\n"
            f"## 技术指标\n"
            f"- RSI(14): {indicators.get('rsi_14', 'N/A')}\n"
            f"- MACD: {indicators.get('macd', 'N/A')} / Signal: {indicators.get('macd_signal', 'N/A')}\n"
            f"- 布林带: 上轨 {indicators.get('bb_high', 'N/A')} / "
            f"中轨 {indicators.get('bb_mid', 'N/A')} / "
            f"下轨 {indicators.get('bb_low', 'N/A')}\n"
            f"- ATR(14): {indicators.get('atr_14', 'N/A')}\n\n"
            f"## 均线系统\n"
            f"- {fmt_ma('sma_20', 'SMA(20)')}\n"
            f"- {fmt_ma('sma_50', 'SMA(50)')}\n"
            f"- {fmt_ma('sma_200', 'SMA(200)')}\n\n"
            f"## 近期 K 线\n"
        )
        for k in klines[-10:]:
            prompt += f"- {k.timestamp}: O={k.open:.2f} H={k.high:.2f} L={k.low:.2f} C={k.close:.2f} V={k.volume:.0f}\n"

        result = await self.parser.parse(
            prompt, TechnicalReport,
            system=self.system_prompt, temperature=self.temperature,
        )
        return AnalystReport(
            analyst=result.analyst,
            stance=result.stance,
            confidence=result.confidence,
            core_reason=result.core_reason,
            key_signals=[s.model_dump() for s in result.key_signals],
            risk_factors=result.risk_factors,
            support_price=result.support_price,
            resistance_price=result.resistance_price,
            details=result.trend_analysis,
        )
