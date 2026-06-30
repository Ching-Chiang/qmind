"""
定时轮询调度器 — 从 fin-agent 迁入，支持分钟/小时/日级轮询。

支持:
- 多品种并行监控（各自独立决策）
- 可配置轮询间隔
- 飞书通知 + 异常告警
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class WatchJob:
    """监控任务"""
    symbol: str
    timeframe: str = "1h"
    interval_sec: int = 300
    last_run: datetime | None = None
    run_count: int = 0
    error_count: int = 0
    enabled: bool = True


class Scheduler:
    """定时调度器"""

    def __init__(self):
        self.jobs: dict[str, WatchJob] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    def add_job(self, symbol: str, timeframe: str = "1h", interval_sec: int = 300) -> None:
        """添加监控任务"""
        self.jobs[symbol] = WatchJob(
            symbol=symbol,
            timeframe=timeframe,
            interval_sec=interval_sec,
        )
        logger.info(f"已添加监控: {symbol} @ {timeframe} (间隔 {interval_sec}s)")

    def remove_job(self, symbol: str) -> None:
        """移除监控任务"""
        self.jobs.pop(symbol, None)

    async def _run_job(self, job: WatchJob, handler: Callable) -> None:
        """运行单个监控任务"""
        while self._running and job.enabled:
            try:
                job.last_run = datetime.utcnow()
                await handler(job.symbol, job.timeframe)
                job.run_count += 1
                job.error_count = 0
            except asyncio.CancelledError:
                break
            except Exception as e:
                job.error_count += 1
                logger.error(f"监控 {job.symbol} 异常: {e} (第 {job.error_count} 次)")

            # 等待下次轮询
            await asyncio.sleep(job.interval_sec)

    async def start(self, handler: Callable) -> None:
        """启动所有监控任务"""
        if self._running:
            logger.warning("调度器已在运行")
            return

        self._running = True
        self._tasks = []
        for job in self.jobs.values():
            if job.enabled:
                task = asyncio.create_task(self._run_job(job, handler))
                self._tasks.append(task)
                # 错开启动时间，避免同时请求
                await asyncio.sleep(1)

        logger.info(f"调度器已启动: {len(self._tasks)} 个任务")

    async def stop(self) -> None:
        """停止所有监控任务"""
        self._running = False
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()
        logger.info("调度器已停止")

    def status(self) -> list[dict[str, Any]]:
        """状态报告"""
        return [
            {
                "symbol": j.symbol,
                "timeframe": j.timeframe,
                "interval_sec": j.interval_sec,
                "run_count": j.run_count,
                "error_count": j.error_count,
                "last_run": j.last_run.isoformat() if j.last_run else None,
                "enabled": j.enabled,
            }
            for j in self.jobs.values()
        ]
