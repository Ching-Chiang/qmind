"""notification.py 通知推送 单元测试 (mock httpx)"""

from __future__ import annotations

from unittest.mock import patch

import pytest
from httpx import Response

from qmind.notification import Notifier


class TestNotifier:
    @pytest.fixture
    def notifier(self):
        return Notifier(feishu_webhook="https://open.feishu.cn/open-apis/bot/v2/hook/test")

    async def test_send_trade_signal(self, notifier):
        with patch.object(notifier, "_send_feishu", return_value=True) as mock:
            result = await notifier.send_trade_signal(
                "BTC/USDT", "LONG", 0.72, 12.5, "技术面突破",
            )
            assert result is True
            mock.assert_called_once()
            msg = mock.call_args[0][0]
            assert "LONG" in msg.body
            assert "BTC/USDT" in msg.body

    async def test_send_alert(self, notifier):
        with patch.object(notifier, "_send_feishu", return_value=True):
            result = await notifier.send_alert("测试告警", level="warning")
            assert result is True

    async def test_no_webhook_returns_false(self):
        n = Notifier(feishu_webhook="")
        result = await n.send_alert("test")
        assert result is False

    async def test_httpx_post_failure(self, notifier):
        """飞书 API 返回非 200 时应返回 False"""
        with patch("httpx.AsyncClient.post",
                   return_value=Response(429, text="rate limited")):
            result = await notifier.send_alert("test")
            assert result is False

    async def test_httpx_exception(self, notifier):
        with patch("httpx.AsyncClient.post",
                   side_effect=Exception("connection error")):
            result = await notifier.send_alert("test")
            assert result is False
