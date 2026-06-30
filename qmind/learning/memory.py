"""
CVRF 概念记忆库 — SQLite 持久化 + 向量相似度检索。

表结构:
- lessons: id, timestamp, symbol, market_condition(JSON), lessons(JSON),
           trade_outcome(JSON), was_bull_correct, was_bear_correct, embedding(BLOB)

检索方式:
- Cosine similarity on 市况特征向量 (Python 计算，无需向量数据库)
- 仅在教训超过 10 万条时考虑专用向量库
"""

from __future__ import annotations

import json
import math
import sqlite3
from datetime import datetime

from qmind.graph.state import Lesson, MarketConditionVector, MemoryEntry


class MemoryStore:
    """CVRF 概念记忆库"""

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
        """初始化数据库表"""
        conn = self._get_conn()
        conn.executescript("""
                CREATE TABLE IF NOT EXISTS lessons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    market_condition TEXT NOT NULL,  -- JSON
                    lessons TEXT NOT NULL,            -- JSON array
                    trade_outcome TEXT NOT NULL,       -- JSON
                    was_bull_correct INTEGER,
                    was_bear_correct INTEGER,
                    embedding TEXT,                    -- JSON float array
                    access_count INTEGER DEFAULT 0,
                    last_accessed TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_lessons_symbol
                    ON lessons(symbol);
                CREATE INDEX IF NOT EXISTS idx_lessons_timestamp
                    ON lessons(timestamp);
            """)

    def save(self, entry: MemoryEntry) -> int:
        """保存一条记忆"""
        with self._get_conn() as conn:
            cur = conn.execute(
                """INSERT INTO lessons
                   (timestamp, symbol, market_condition, lessons,
                    trade_outcome, was_bull_correct, was_bear_correct, embedding)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    entry.timestamp.isoformat() if isinstance(entry.timestamp, datetime) else str(entry.timestamp),
                    entry.symbol,
                    entry.market_condition.model_dump_json(),
                    json.dumps([les.model_dump() for les in entry.lessons]),
                    json.dumps(entry.trade_outcome),
                    int(entry.was_bull_correct) if entry.was_bull_correct is not None else None,
                    int(entry.was_bear_correct) if entry.was_bear_correct is not None else None,
                    json.dumps(entry.embedding) if entry.embedding else None,
                ),
            )
            return cur.lastrowid or 0

    def get_recent(self, limit: int = 20) -> list[MemoryEntry]:
        """获取最近的记忆"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lessons ORDER BY timestamp DESC LIMIT ?",
                (limit,),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def get_by_symbol(self, symbol: str, limit: int = 20) -> list[MemoryEntry]:
        """获取某标的的历史教训"""
        with self._get_conn() as conn:
            rows = conn.execute(
                "SELECT * FROM lessons WHERE symbol = ? ORDER BY timestamp DESC LIMIT ?",
                (symbol, limit),
            ).fetchall()
            return [self._row_to_entry(r) for r in rows]

    def count(self) -> int:
        """总教训数"""
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as cnt FROM lessons").fetchone()
            return row["cnt"] if row else 0

    # ── 相似度检索 ──

    def _vectorize_condition(self, cond: MarketConditionVector) -> list[float]:
        """将市况特征转为数值向量"""
        trend_map = {"uptrend": 1.0, "downtrend": -1.0, "sideways": 0.0, "reversal": 0.5, "": 0.0}
        vol_map = {"low": 0.0, "medium": 0.5, "high": 1.0, "": 0.5}
        cycle_map = {"accumulation": 0.0, "markup": 0.33, "distribution": 0.66, "markdown": 1.0, "": 0.5}
        vol_trend_map = {"increasing": 1.0, "decreasing": -1.0, "flat": 0.0, "": 0.0}

        return [
            trend_map.get(cond.trend, 0.0),
            vol_map.get(cond.volatility, 0.5),
            cycle_map.get(cond.market_cycle, 0.5),
            max(-1.0, min(1.0, cond.momentum)),
            vol_trend_map.get(cond.volume_trend, 0.0),
        ]

    @staticmethod
    def _cosine_similarity(a: list[float], b: list[float]) -> float:
        """余弦相似度"""
        if not a or not b:
            return 0.0
        dot = sum(x * y for x, y in zip(a, b, strict=True))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(y * y for y in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def search_similar(
        self,
        condition: MarketConditionVector,
        top_k: int = 5,
        min_similarity: float = 0.3,
        symbol: str | None = None,
    ) -> list[tuple[MemoryEntry, float]]:
        """搜索相似市况的历史教训"""
        query_vec = self._vectorize_condition(condition)

        with self._get_conn() as conn:
            if symbol:
                rows = conn.execute(
                    "SELECT * FROM lessons WHERE symbol = ? ORDER BY timestamp DESC",
                    (symbol,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM lessons ORDER BY timestamp DESC"
                ).fetchall()

        scored: list[tuple[MemoryEntry, float]] = []
        for row in rows:
            entry = self._row_to_entry(row)
            stored_embedding = self._get_embedding(row)
            vec = stored_embedding if stored_embedding else self._vectorize_condition(entry.market_condition)
            sim = self._cosine_similarity(query_vec, vec)

            if sim >= min_similarity:
                scored.append((entry, sim))

        scored.sort(key=lambda x: -x[1])
        return scored[:top_k]

    def _get_embedding(self, row: sqlite3.Row) -> list[float] | None:
        """从行中提取 embedding"""
        raw = row["embedding"]
        if raw:
            try:
                return json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                pass
        return None

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> MemoryEntry:
        """将数据库行转为 MemoryEntry"""
        mc = MarketConditionVector.model_validate_json(row["market_condition"])
        lessons_data = json.loads(row["lessons"])
        lessons = [Lesson.model_validate(les) for les in lessons_data]

        return MemoryEntry(
            id=row["id"],
            timestamp=datetime.fromisoformat(row["timestamp"]),
            symbol=row["symbol"],
            market_condition=mc,
            lessons=lessons,
            trade_outcome=json.loads(row["trade_outcome"]),
            was_bull_correct=bool(row["was_bull_correct"]) if row["was_bull_correct"] is not None else None,
            was_bear_correct=bool(row["was_bear_correct"]) if row["was_bear_correct"] is not None else None,
            embedding=json.loads(row["embedding"]) if row["embedding"] else None,
        )
