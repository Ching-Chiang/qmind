"""
QMind — CLI 入口

用法:
    qmind analyze BTC/USDT
    qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06
    qmind watch BTC/USDT ETH/USDT
    qmind learn --from-log trades.log
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import click
from rich.console import Console
from rich.table import Table

from qmind.audit_log import AuditLogger
from qmind.config import Config
from qmind.execution.factory import ExchangeFactory
from qmind.graph.pipeline import QMindPipeline
from qmind.llm.client import LLMClient
from qmind.notification import Notifier
from qmind.scheduler import Scheduler
from qmind.strategies.registry import list_strategies

console = Console()


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="配置文件路径")
@click.option("--dry-run/--live", default=True, help="dryRun 模式（默认开启）")
@click.option("--source", default="auto", help="数据源: auto / yfinance / tushare / binance / mock")
@click.option("--verbose", "-v", is_flag=True, help="显示智能体详细思考过程")
@click.pass_context
def cli(ctx: click.Context, config: str | None, dry_run: bool, source: str, verbose: bool) -> None:
    """QMind — 量化交易多智能体系统"""
    ctx.ensure_object(dict)
    cfg = Config(path=Path(config) if config else None)
    ctx.obj["config"] = cfg
    ctx.obj["dry_run"] = dry_run
    ctx.obj["verbose"] = verbose
    ctx.obj["llm_client"] = LLMClient()

    # 初始化各模块
    exchange = ExchangeFactory.create(
        "dry_run" if dry_run else cfg.get("execution.default_exchange", "binance"),
        dry_run=dry_run,
    )
    ctx.obj["audit"] = AuditLogger(db_path=cfg.db_path)
    ctx.obj["pipeline"] = QMindPipeline(ctx.obj["llm_client"], exchange=exchange, data_source=source)
    ctx.obj["exchange"] = exchange
    ctx.obj["notifier"] = Notifier(
        feishu_webhook=cfg.get("notification.webhook_url", ""),
    )


@cli.command()
@click.argument("symbol")
@click.option("--timeframe", "-t", default="1h", help="时间框架")
@click.option("--output", "-o", type=click.Path(), help="输出 HTML 报告路径")
@click.pass_context
def analyze(ctx: click.Context, symbol: str, timeframe: str, output: str | None) -> None:
    """对标的执行一次完整分析"""
    dry_run = ctx.obj["dry_run"]

    async def _run():
        verbose = ctx.obj.get("verbose", False)
        console.print(f"\n[bold cyan]QMind 分析报告[/bold cyan] · {symbol}")
        console.print(f"[dim]时间框架: {timeframe} | dryRun: {'on' if dry_run else 'off'} | verbose: {'on' if verbose else 'off'}[/dim]")

        pipeline: QMindPipeline = ctx.obj["pipeline"]

        if verbose:
            console.print("\n[bold]阶段 1/5: 数据采集[/bold]")
            console.print(f"  [dim]获取 {symbol} 行情数据...[/dim]")

        result = await pipeline.run(symbol, timeframe)

        # 输出结果
        decision = result.get("decision")
        risk = result.get("risk")
        analyses = result.get("analyses", [])
        debate = result.get("debate")
        execution = result.get("execution_result")
        disagreement = result.get("disagreement", 0.0)
        errors = result.get("errors", [])

        # ── 分析师详细输出（verbose） ──
        if verbose and analyses:
            console.print("\n[bold]阶段 2/5: 多维分析[/bold]")
            for a in analyses:
                icon = {"bullish": "[green]+[/]", "neutral": "[yellow]~[/]", "bearish": "[red]-[/]"}
                console.print(f"  {icon.get(a.stance, '[yellow]~[/]')} [bold]{a.analyst}[/bold]: {a.stance} ({a.confidence:.0%})")
                if a.core_reason:
                    console.print(f"    [dim]逻辑:[/dim] {a.core_reason[:200]}")
                if a.key_signals:
                    for sig in a.key_signals[:3]:
                        if isinstance(sig, str):
                            console.print(f"    [dim]信号:[/dim] {sig}")
                        elif isinstance(sig, dict):
                            s = sig.get("signal") or sig.get("name", "")
                            v = sig.get("value", "")
                            console.print(f"    [dim]信号:[/dim] {s} {v}")
                        else:
                            console.print(f"    [dim]信号:[/dim] {sig.signal} {sig.value}")
                if a.risk_factors:
                    for rf in a.risk_factors[:2]:
                        console.print(f"    [red]风险:[/red] {rf}")

        # ── 辩论详细输出（verbose） ──
        if verbose and debate:
            console.print("\n[bold]阶段 3/5: 多空辩论[/bold]")
            console.print(f"  分歧度 δ={disagreement:.3f}")
            console.print(f"  收敛: {debate.get('converged')}")
            console.print(f"  置信度降级因子: {debate.get('consensus_confidence', 1.0):.2f}")
            if debate.get("agreement_points"):
                console.print(f"  [green]共识点:[/green]")
                for p in debate["agreement_points"][:3]:
                    console.print(f"    + {p}")
            if debate.get("disagreement_points"):
                console.print(f"  [red]分歧点:[/red]")
                for p in debate["disagreement_points"][:3]:
                    console.print(f"    - {p}")

        # ── 决策详细输出（verbose） ──
        if verbose:
            console.print("\n[bold]阶段 4/5: 交易决策[/bold]")

        # 分析师共识（简洁版）
        if not verbose:
            console.print("\n[bold]分析师共识[/bold]")
            for a in analyses:
                icon = {"bullish": "[green]+[/]", "neutral": "[yellow]~[/]", "bearish": "[red]-[/]"}
                console.print(f"  {icon.get(a.stance, '[yellow]~[/]')} [{a.analyst}] {a.stance} ({a.confidence:.0%})")
            console.print(f"\n  [dim]分歧度 δ={disagreement:.3f}")
            if debate:
                console.print(f"  辩论: 降级因子={debate.get('consensus_confidence', 1.0):.2f}")

        if decision:
            d = decision
            console.print(f"\n[bold cyan]决策: {d.decision}[/bold cyan]")
            console.print(f"  入场: {d.entry.get('price', 'N/A')}")
            console.print(f"  止损: {d.stop_loss.get('price', 'N/A') if d.stop_loss else 'N/A'}")
            console.print(f"  仓位: {d.position_size_pct:.1f}%")
            console.print(f"  置信度: {d.confidence:.2f}")
            console.print(f"  风险收益比: {d.risk_reward_ratio:.2f}")
            if verbose and d.reasoning_chain:
                rc = d.reasoning_chain
                if rc.get("data_cot"):
                    console.print(f"  [dim]Data-CoT:[/dim] {rc['data_cot'][:200]}")
                if rc.get("concept_cot"):
                    console.print(f"  [dim]Concept-CoT:[/dim] {rc['concept_cot'][:200]}")
                if rc.get("thesis_cot"):
                    console.print(f"  [dim]Thesis-CoT:[/dim] {rc['thesis_cot'][:200]}")

        # ── 风控详细输出（verbose） ──
        if verbose and risk:
            console.print("\n[bold]阶段 5/5: 三角风控审核[/bold]")
            for role in ("aggressive_review", "conservative_review", "neutral_review"):
                review = getattr(risk, role, None)
                if review:
                    marker = {"approve": "[green]PASS[/]", "reject": "[red]VETO[/]", "modify": "[yellow]MODIFY[/]"}
                    status = marker.get(review.decision, "[yellow]?[/]")
                    console.print(f"  {status} [bold]{review.role}[/bold]")
                    if review.reason:
                        console.print(f"    [dim]理由:[/dim] {review.reason[:200]}")
                    if review.risk_assessment:
                        console.print(f"    [dim]风险评估:[/dim] {review.risk_assessment[:200]}")
                    if review.concerns:
                        for c in review.concerns[:2]:
                            console.print(f"    [red]顾虑:[/red] {c}")

        # 风控简洁版
        if risk:
            status = "[green]PASS[/]" if risk.approved else "[red]VETO[/]"
            veto = f" (by: {', '.join(risk.vetoed_by)})" if risk.vetoed_by else ""
            console.print(f"\n[bold]风控审核: {status}{veto}[/bold]")

        # 执行
        if execution:
            exec_status = execution.get("status", "unknown")
            if exec_status == "dry_run":
                console.print("\n[dim]模拟执行 — Net PnL 已模拟: $0.00[/dim]")
            elif exec_status == "rejected":
                console.print(f"\n[yellow]交易已取消: {execution.get('reason', '')}[/yellow]")

        # 错误
        for err in errors:
            console.print(f"[red]! {err}[/red]")

        # 审计
        audit: AuditLogger = ctx.obj["audit"]
        cost_tracker = ctx.obj["llm_client"].cost_tracker
        audit.log_decision(
            symbol=symbol,
            decision=decision.decision if decision else "N/A",
            confidence=decision.confidence if decision else 0,
            position_size_pct=decision.position_size_pct if decision else 0,
            entry_price=decision.entry.get("price", 0) if decision else 0,
            stop_loss=decision.stop_loss.get("price", 0) if decision else 0,
            risk_reward_ratio=decision.risk_reward_ratio if decision else 0,
            approval=risk.approved if risk else False,
            vetoed_by=risk.vetoed_by if risk else None,
            token_usage=cost_tracker.summary(),
            analyses=analyses,
            debate=debate,
            risk=risk,
            error="; ".join(errors) if errors else "",
            session_id=f"{symbol}_{timeframe}",
        )

        console.print(f"\n[dim]Token 用量: {cost_tracker.total_tokens()}"
                      f" | 成本: ${cost_tracker.total_cost():.6f}[/dim]")
        console.print(f"\n[dim]提示: qmind analyze {symbol} --live 去掉 dryRun 模拟[/dim]")

    asyncio.run(_run())


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--timeframe", "-t", default="1h", help="时间框架")
@click.option("--interval", "-i", default=300, help="轮询间隔（秒）")
@click.pass_context
def watch(ctx: click.Context, symbols: tuple[str], timeframe: str, interval: int) -> None:
    """持续监控标的，有信号时推送通知"""
    dry_run = ctx.obj["dry_run"]
    console.print(f"[yellow]watch[/yellow] 模式启动 (dryRun={'on' if dry_run else 'off'})")
    console.print(f"监控标的: {', '.join(symbols)} | 时间框架: {timeframe}")

    async def _watch():
        pipeline: QMindPipeline = ctx.obj["pipeline"]
        notifier: Notifier = ctx.obj["notifier"]
        scheduler = Scheduler()

        async def handler(symbol: str, tf: str):
            result = await pipeline.run(symbol, tf)
            decision = result.get("decision")
            risk = result.get("risk")
            if decision and decision.decision != "HOLD" and risk and risk.approved:
                await notifier.send_trade_signal(
                    symbol=symbol,
                    decision=decision.decision,
                    confidence=decision.confidence,
                    position_pct=decision.position_size_pct,
                    reason=decision.reasoning_chain.get("thesis_cot", ""),
                )
                console.print(f"[green]信号推送: {symbol} {decision.decision}[/green]")

        for sym in symbols:
            scheduler.add_job(sym, timeframe, interval)

        try:
            await scheduler.start(handler)
        except asyncio.CancelledError:
            await scheduler.stop()

    try:
        asyncio.run(_watch())
    except KeyboardInterrupt:
        console.print("\n[yellow]监控已停止[/yellow]")


@cli.command()
@click.option("--strategy", "-s", default="ma_cross", help="策略名称")
@click.option("--start", required=True, help="回测开始日期 (YYYY-MM)")
@click.option("--end", required=True, help="回测结束日期 (YYYY-MM)")
@click.option("--output", "-o", type=click.Path(), help="输出 HTML 报告路径")
def backtest(strategy: str, start: str, end: str, output: str | None) -> None:
    """运行策略回测"""
    console.print(f"\n[bold cyan]QMind 回测[/bold cyan] · {strategy}")
    console.print(f"[dim]区间: {start} → {end}[/dim]")
    strategies = list_strategies()
    names = [s["name"] for s in strategies]
    if strategy not in names:
        console.print(f"[red]未知策略: {strategy}[/red]")
        console.print(f"可用策略: {', '.join(names)}")
        return
    console.print(f"[green]策略 {strategy} 已找到，回测引擎将在 Phase 1.3 后可用[/green]")


@cli.command()
@click.option("--from-log", "log_path", type=click.Path(exists=True), help="交易日志路径")
@click.pass_context
def learn(ctx: click.Context, log_path: str | None) -> None:
    """从交易日志中学习，更新 CVRF 经验库"""
    from qmind.learning.memory import MemoryStore

    console.print("[bold cyan]QMind CVRF 学习[/bold cyan]")

    async def _learn():
        memory = MemoryStore("qmind.db")

        console.print(f"当前记忆库: {memory.count()} 条教训")

        if log_path:
            console.print(f"[yellow]从 {log_path} 读取交易记录...[/yellow]")
            # TODO: 解析 trades.log → TradeRecord 列表 → pipeline.batch_process()
            console.print("[yellow]交易解析功能将在后续版本完成[/yellow]")
        else:
            recent = memory.get_recent(limit=10)
            if recent:
                console.print("\n[green]最近教训:[/green]")
                for entry in recent:
                    for lesson in entry.lessons:
                        console.print(f"  > {lesson.lesson} ({lesson.confidence:.0%})")
            else:
                console.print("[dim]暂无教训记录。运行分析交易后自动生成。[/dim]")

        console.print("\n[dim]qmind learn --from-log trades.log 从日志学习[/dim]")

    asyncio.run(_learn())


@cli.command(name="list")
@click.option("--strategies/--no-strategies", default=True, help="列出策略")
@click.option("--audit/--no-audit", default=False, help="查看审计日志")
@click.pass_context
def list_cmd(ctx: click.Context, strategies: bool, audit: bool) -> None:
    """列出系统信息"""
    if strategies:
        table = Table(title="注册策略")
        table.add_column("名称", style="cyan")
        table.add_column("描述")
        for s in list_strategies():
            table.add_row(s["name"], s["description"])
        console.print(table)

    if audit:
        audit_logger: AuditLogger = ctx.obj["audit"]
        summary = audit_logger.summary()
        table = Table(title="审计摘要")
        table.add_column("指标", style="cyan")
        table.add_column("数值")
        for k, v in summary.items():
            table.add_row(k, str(v))
        console.print(table)


@cli.command()
@click.argument("symbol")
def price(symbol: str) -> None:
    """测试获取标的实时价格"""
    async def _price():
        exchange = ExchangeFactory.create("dry_run", dry_run=True)
        from qmind.data.sources.factory import DataSourceFactory
        factory = DataSourceFactory()
        data = await factory.fetch_market_data(symbol)
        for klines in data.klines.values():
            if klines:
                price_val = klines[-1].close
                exchange.update_price(symbol, price_val)
                console.print(f"\n[bold cyan]{symbol} 实时价格[/bold cyan]")
                console.print(f"  当前: ${price_val:,.2f}")
                console.print(f"  开盘: ${klines[-1].open:,.2f}")
                console.print(f"  最高: ${klines[-1].high:,.2f}")
                console.print(f"  最低: ${klines[-1].low:,.2f}")
                console.print(f"  K线数: {len(klines)}")
                return
        console.print("[red]未能获取价格数据[/red]")
        console.print("[dim]可能原因: yfinance 速率限制、网络问题、无效标的[/dim]")
    asyncio.run(_price())


@cli.command()
def version() -> None:
    """显示版本信息"""
    from qmind import __version__
    console.print(f"[bold cyan]QMind[/bold cyan] v{__version__}")
    console.print("量化交易多智能体系统")


if __name__ == "__main__":
    cli()
