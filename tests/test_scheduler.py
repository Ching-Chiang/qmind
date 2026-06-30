"""scheduler.py 调度器 单元测试"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from qmind.scheduler import Scheduler


class TestScheduler:
    @pytest.fixture
    def scheduler(self):
        return Scheduler()

    def test_add_job(self, scheduler):
        scheduler.add_job("BTC/USDT", "1h", 60)
        assert len(scheduler.jobs) == 1
        assert scheduler.jobs["BTC/USDT"].timeframe == "1h"

    def test_remove_job(self, scheduler):
        scheduler.add_job("BTC/USDT")
        scheduler.remove_job("BTC/USDT")
        assert len(scheduler.jobs) == 0

    def test_status_empty(self, scheduler):
        assert scheduler.status() == []

    def test_status_with_jobs(self, scheduler):
        scheduler.add_job("BTC/USDT", "1h", 300)
        scheduler.add_job("ETH/USDT", "4h", 900)
        status = scheduler.status()
        assert len(status) == 2
        assert status[0]["symbol"] == "BTC/USDT"
        assert status[1]["interval_sec"] == 900

    async def test_start_stop(self, scheduler):
        handler = AsyncMock()
        scheduler.add_job("TEST", "1h", 1)  # 1 second interval

        # Start in background
        start_task = asyncio.create_task(scheduler.start(handler))
        await asyncio.sleep(0.1)  # Let it start
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False
        start_task.cancel()
