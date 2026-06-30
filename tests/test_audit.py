"""audit_log.py 审计日志 集成测试"""

from __future__ import annotations

import pytest

from qmind.audit_log import AuditLogger


@pytest.fixture
def audit():
    return AuditLogger(":memory:")


class TestAuditLogger:
    def test_log_and_summary(self, audit):
        lid = audit.log_decision("BTC/USDT", "LONG", confidence=0.72,
                                  position_size_pct=12.5, approval=True)
        assert lid > 0
        summary = audit.summary()
        assert summary["total_decisions"] == 1
        assert summary["longs"] == 1
        assert summary["approved"] == 1

    def test_log_rejected(self, audit):
        audit.log_decision("ETH/USDT", "SHORT", confidence=0.55,
                            approval=False, vetoed_by=["conservative"])
        summary = audit.summary()
        assert summary["rejected"] == 1
        assert summary["shorts"] == 1

    def test_get_recent(self, audit):
        for i in range(5):
            audit.log_decision(f"PAIR{i}", "HOLD")
        recent = audit.get_recent(limit=3)
        assert len(recent) == 3

    def test_get_by_symbol(self, audit):
        audit.log_decision("BTC/USDT", "LONG")
        audit.log_decision("ETH/USDT", "SHORT")
        audit.log_decision("BTC/USDT", "HOLD")
        btc = audit.get_by_symbol("BTC/USDT")
        assert len(btc) == 2

    def test_token_usage_stored(self, audit):
        tok = {"total_tokens": 1500, "cost_usd": 0.0075}
        audit.log_decision("TEST", "LONG", token_usage=tok)
        recent = audit.get_recent(1)
        import json
        stored = json.loads(recent[0]["token_usage"])
        assert stored["total_tokens"] == 1500
