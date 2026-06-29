"""
QMind — CLI 入口

用法:
    qmind analyze BTC/USDT
    qmind backtest --strategy ma_cross --start 2024-01 --end 2025-06
    qmind watch BTC/USDT ETH/USDT
    qmind learn --from-log trades.log
"""

from __future__ import annotations

from pathlib import Path

import click

from qmind.config import Config


@click.group()
@click.option("--config", "-c", type=click.Path(exists=True), help="配置文件路径")
@click.option("--dry-run/--live", default=True, help="dryRun 模式（默认开启）")
@click.pass_context
def cli(ctx: click.Context, config: str | None, dry_run: bool) -> None:
    """QMind — 量化交易多智能体系统"""
    ctx.ensure_object(dict)
    ctx.obj["config"] = Config(path=Path(config) if config else None)
    ctx.obj["dry_run"] = dry_run


@cli.command()
@click.argument("symbols", nargs=-1, required=True)
@click.option("--timeframe", "-t", default="1h", help="时间框架 (1m, 5m, 1h, 1d)")
@click.option("--interval", "-i", default=300, help="轮询间隔（秒），默认 300")
@click.pass_context
def watch(ctx: click.Context, symbols: tuple[str], timeframe: str, interval: int) -> None:
    """持续监控标的，有信号时推送通知"""
    from rich.console import Console
    console = Console()
    dry_run = ctx.obj["dry_run"]
    console.print(f"[yellow]watch[/yellow] 模式启动 (dryRun={'on' if dry_run else 'off'})")
    console.print(f"监控标的: {', '.join(symbols)} | 时间框架: {timeframe}")
    if dry_run:
        console.print("[dim]提示: 去掉 --dry-run 以实盘模式运行[/dim]")


@cli.command()
@click.argument("symbol")
@click.option("--timeframe", "-t", default="1h", help="时间框架")
@click.option("--output", "-o", type=click.Path(), help="输出 HTML 报告路径")
@click.pass_context
def analyze(ctx: click.Context, symbol: str, timeframe: str, output: str | None) -> None:
    """对标的执行一次完整分析"""
    from rich.console import Console
    console = Console()
    dry_run = ctx.obj["dry_run"]

    console.print(f"\n[bold cyan]QMind 分析报告[/bold cyan] · {symbol}")
    console.print(f"[dim]时间框架: {timeframe} | dryRun: {'on' if dry_run else 'off'}[/dim]")
    console.print("[yellow]⏳ 分析中... 此功能将在 Phase 1 完成后可用[/yellow]\n")


@cli.command()
@click.option("--strategy", "-s", default="ma_cross", help="策略名称")
@click.option("--start", required=True, help="回测开始日期 (YYYY-MM)")
@click.option("--end", required=True, help="回测结束日期 (YYYY-MM)")
@click.option("--output", "-o", type=click.Path(), help="输出 HTML 报告路径")
def backtest(strategy: str, start: str, end: str, output: str | None) -> None:
    """运行策略回测"""
    from rich.console import Console
    console = Console()
    console.print(f"\n[bold cyan]QMind 回测[/bold cyan] · {strategy}")
    console.print(f"[dim]区间: {start} → {end}[/dim]")
    console.print("[yellow]⏳ 回测中... 此功能将在 Phase 1 完成后可用[/yellow]\n")


@cli.command()
@click.option("--from-log", "log_path", type=click.Path(exists=True), help="交易日志路径")
def learn(log_path: str | None) -> None:
    """从交易日志中学习，更新 CVRF 经验库"""
    from rich.console import Console
    console = Console()
    console.print("[bold cyan]QMind CVRF 学习[/bold cyan]")
    console.print("[yellow]⏳ 学习中... 此功能将在 Phase 4 完成后可用[/yellow]\n")


if __name__ == "__main__":
    cli()
