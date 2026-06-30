"""
全链路审计日志 — 每个决策的记录。

记录:
- 时间戳、模型版本、Token 用量
- 证据链（分析师报告、辩论纪要）
- 原始 LLM 响应
- 最终决策和执行结果

P1-P6 兼容: 每个记录含足够信息以复现决策过程。
"""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any


class AuditLogger:
    """审计日志记录器"""

    def __init__(self, db_path: str = "qmind.db"):
        self.db_path = db_path
        self._conn: sqlite3.Connection | None = None
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
            self._conn.row_factory = sqlite3.Row
        return self._conn

    def _init_db(self) -> None:
        conn = self._get_conn()
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS audit_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    session_id TEXT,
                    decision TEXT,          -- LONG / SHORT / HOLD
                    confidence REAL,
                    position_size_pct REAL,
                    entry_price REAL,
                    stop_loss REAL,
                    risk_reward_ratio REAL,
                    approval BOOLEAN,       -- 风控是否通过
                    vetoed_by TEXT,        -- 否决者列表 JSON
                    pnl_abs REAL,          -- 事后结果
                    pnl_pct REAL,
                    token_usage TEXT,      -- JSON: {total, cost, by_model}
                    analyses_json TEXT,    -- 分析师报告 JSON
                    debate_json TEXT,      -- 辩论纪要 JSON
                    risk_json TEXT,        -- 风控审核 JSON
                    llm_responses_json TEXT, -- LLM 原始响应
                    error TEXT,
                    created_at TEXT DEFAULT (datetime('now'))
                );

                CREATE INDEX IF NOT EXISTS idx_audit_symbol
                    ON audit_log(symbol);
                CREATE INDEX IF NOT EXISTS idx_audit_timestamp
                    ON audit_log(timestamp);
            """)

    def log_decision(
        self,
        symbol: str,
        decision: str,
        confidence: float = 0.0,
        position_size_pct: float = 0.0,
        entry_price: float = 0.0,
        stop_loss: float = 0.0,
        risk_reward_ratio: float = 0.0,
        approval: bool = False,
        vetoed_by: list[str] = None,
        token_usage: dict[str, Any] = None,
        analyses: list[Any] = None,
        debate: Any = None,
        risk: Any = None,
        llm_responses: list[Any] = None,
        error: str = "",
        session_id: str = "",
    ) -> int:
        """记录一条审计日志"""
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO audit_log
                   (timestamp, symbol, session_id, decision, confidence,
                    position_size_pct, entry_price, stop_loss,
                    risk_reward_ratio, approval, vetoed_by,
                    token_usage, analyses_json, debate_json,
                    risk_json, llm_responses_json, error)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    datetime.utcnow().isoformat(),
                    symbol,
                    session_id,
                    decision,
                    confidence,
                    position_size_pct,
                    entry_price,
                    stop_loss,
                    risk_reward_ratio,
                    int(approval) if approval is not None else None,
                    json.dumps(vetoed_by) if vetoed_by else None,
                    json.dumps(token_usage) if token_usage else None,
                    json.dumps(
                        [a.model_dump() if hasattr(a, 'model_dump') else a
                         for a in (analyses or [])],
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        debate.model_dump() if hasattr(debate, 'model_dump') else (debate or {}),
                        ensure_ascii=False,
                    ),
                    json.dumps(
                        risk.model_dump() if hasattr(risk, 'model_dump') else (risk or {}),
                        ensure_ascii=False,
                    ),
                    json.dumps(llm_responses or [], ensure_ascii=False),
                    error,
                ),
            )
            return cur.lastrowid or 0

    def get_recent(self, limit: int = 50) -> list[dict[str, Any]]:
        """最近审计记录"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [dict(r) for r in rows]

    def get_by_symbol(self, symbol: str, limit: int = 50) -> list[dict[str, Any]]:
        """某标的的交易历史"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def summary(self) -> dict[str, Any]:
        """审计摘要"""
        with self._get_conn() as conn:
            total = conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()["c"]
            approved = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE approval = 1").fetchone()["c"]
            rejected = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE approval = 0").fetchone()["c"]
            longs = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE decision = 'LONG'").fetchone()["c"]
            shorts = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE decision = 'SHORT'").fetchone()["c"]
            holds = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE decision = 'HOLD'").fetchone()["c"]
            return {
                "total_decisions": total,
                "approved": approved,
                "rejected": rejected,
                "longs": longs,
                "shorts": shorts,
                "holds": holds,
            }
