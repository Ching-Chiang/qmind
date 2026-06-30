"""
通知推送 — 飞书 Webhook + 邮件 SMTP。

飞书通知示例:
QMind 信号: BTC/USDT LONG
入场: 86,200 | 置信度: 0.61 | 仓位: 12.5%
理由: 技术面突破 + 低分歧
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


@dataclass
class NotificationMessage:
    title: str
    body: str
    level: str = "info"  # info / warning / error
    symbol: str = ""
    decision: str = ""


class Notifier:
    """通知推送器"""

    def __init__(self, feishu_webhook: str = "", smtp_config: dict[str, Any] = None):
        self.feishu_webhook = feishu_webhook
        self.smtp_config = smtp_config or {}

    async def send_trade_signal(
        self, symbol: str, decision: str,
        confidence: float, position_pct: float, reason: str,
    ) -> bool:
        """推送交易信号"""
        body = (
            f"QMind 信号: {symbol} {decision}\n"
            f"置信度: {confidence:.2f} | 仓位: {position_pct:.1f}%\n"
            f"理由: {reason}"
        )
        msg = NotificationMessage(
            title=f"QMind Signal: {symbol} {decision}",
            body=body,
            level="info", symbol=symbol, decision=decision,
        )
        return await self._send(msg)

    async def send_alert(self, message: str, level: str = "warning") -> bool:
        """推送告警"""
        msg = NotificationMessage(title=f"QMind Alert [{level}]", body=message, level=level)
        return await self._send(msg)

    async def send_report(self, title: str, body: str) -> bool:
        """推送报告"""
        msg = NotificationMessage(title=title, body=body, level="info")
        return await self._send(msg)

    async def _send(self, msg: NotificationMessage) -> bool:
        """发送通知"""
        feishu_ok = await self._send_feishu(msg)
        email_ok = await self._send_email(msg)
        return feishu_ok or email_ok

    async def _send_feishu(self, msg: NotificationMessage) -> bool:
        """飞书 Webhook 推送"""
        if not self.feishu_webhook:
            return False
        try:
            payload = {
                "msg_type": "post",
                "content": {
                    "post": {
                        "zh_cn": {
                            "title": msg.title,
                            "content": [[{"tag": "text", "text": msg.body}]],
                        }
                    }
                },
            }
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(self.feishu_webhook, json=payload)
                if resp.status_code == 200:
                    logger.info(f"飞书通知已发送: {msg.title}")
                    return True
                logger.warning(f"飞书通知失败: {resp.status_code} {resp.text}")
                return False
        except Exception as e:
            logger.error(f"飞书通知异常: {e}")
            return False

    async def _send_email(self, msg: NotificationMessage) -> bool:
        """邮件推送"""
        if not self.smtp_config:
            return False
        try:
            import smtplib
            from email.message import EmailMessage

            email_msg = EmailMessage()
            email_msg.set_content(msg.body)
            email_msg["Subject"] = msg.title
            email_msg["From"] = self.smtp_config.get("from_addr", "")
            email_msg["To"] = self.smtp_config.get("to_addr", "")

            server = smtplib.SMTP(
                self.smtp_config.get("host", "smtp.gmail.com"),
                self.smtp_config.get("port", 587),
            )
            server.starttls()
            server.login(
                self.smtp_config.get("user", ""),
                self.smtp_config.get("password", ""),
            )
            server.send_message(email_msg)
            server.quit()
            logger.info(f"邮件通知已发送: {msg.title}")
            return True
        except Exception as e:
            logger.error(f"邮件通知异常: {e}")
            return False
