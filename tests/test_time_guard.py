"""TimeGuard 时间完整性校验器 单元测试"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from qmind.data.time_guard import TimeGuard, TimeGuardError


class TestTimeGuard:
    """TimeGuard 核心功能测试"""

    def setup_method(self):
        self.guard = TimeGuard(
            decision_time=datetime(2026, 6, 29, 12, 0, 0),
            tolerance_sec=0,
        )

    def test_past_data_accepted(self):
        """过去的数据应通过校验"""
        # 2026-06-29 11:00:00 UTC in ms
        ts = int(datetime(2026, 6, 29, 11, 0, 0, tzinfo=UTC).timestamp() * 1000)
        self.guard.check_timestamp(ts, "test")  # should not raise

    def test_future_data_rejected(self):
        """未来的数据应被拒绝"""
        ts = int(datetime(2026, 6, 29, 13, 0, 0, tzinfo=UTC).timestamp() * 1000)
        with pytest.raises(TimeGuardError, match="Look-Ahead"):
            self.guard.check_timestamp(ts, "future_data")

    def test_exact_decision_time(self):
        """等于决策时间的数据应通过"""
        ts = int(datetime(2026, 6, 29, 12, 0, 0, tzinfo=UTC).timestamp() * 1000)
        self.guard.check_timestamp(ts)  # should not raise

    def test_tolerance_window(self):
        """在容差窗口内的未来数据应通过"""
        guard = TimeGuard(
            decision_time=datetime(2026, 6, 29, 12, 0, 0),
            tolerance_sec=300,  # 5 min tolerance
        )
        # 4 分 30 秒后
        ts = int(datetime(2026, 6, 29, 12, 4, 30, tzinfo=UTC).timestamp() * 1000)
        guard.check_timestamp(ts)  # should not raise

    def test_validate_as_of_nested(self):
        """递归检查应发现嵌套的未来时间戳"""
        data = {
            "level1": {
                "level2": {
                    "as_of": datetime(2026, 6, 29, 15, 0, 0),  # future
                }
            }
        }
        with pytest.raises(TimeGuardError):
            self.guard.validate_as_of(data)

    def test_validate_as_of_clean(self):
        """嵌套结构中无未来数据应通过"""
        data = {
            "level1": {
                "as_of": datetime(2026, 6, 29, 10, 0, 0),  # past
            },
            "items": [1, 2, 3],
        }
        self.guard.validate_as_of(data)  # should not raise
