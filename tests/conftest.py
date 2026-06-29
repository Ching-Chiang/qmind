"""pytest 共享 fixtures"""

from __future__ import annotations

from datetime import datetime

import pytest

from qmind.data.time_guard import TimeGuard
from qmind.llm.client import CostTracker, LLMClient
from qmind.llm.router import LLMRouter
from qmind.llm.structured import StructuredParser


@pytest.fixture
def cost_tracker() -> CostTracker:
    return CostTracker()


@pytest.fixture
def llm_client(cost_tracker: CostTracker) -> LLMClient:
    return LLMClient(cost_tracker=cost_tracker)


@pytest.fixture
def llm_router(llm_client: LLMClient) -> LLMRouter:
    return LLMRouter(client=llm_client)


@pytest.fixture
def structured_parser(llm_client: LLMClient) -> StructuredParser:
    return StructuredParser(client=llm_client, model="claude-sonnet-4-6")


@pytest.fixture
def time_guard() -> TimeGuard:
    return TimeGuard(decision_time=datetime(2026, 6, 29, 12, 0, 0))
