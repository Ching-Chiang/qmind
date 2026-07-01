"""
QMind Web Dashboard — FastAPI Server

Usage:
    python -m qmind.server              # 默认 127.0.0.1:8000
    python -m qmind.server 9000         # 自定义端口
    uvicorn qmind.server:app --port 8000
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="QMind Dashboard")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Cached state ──
_last_result: dict[str, Any] | None = None
_last_run: float = 0
_running = False


def _extract_events(state: dict[str, Any]) -> list[dict[str, Any]]:
    """将 pipeline 结果转换为前端事件序列"""
    events: list[dict[str, Any]] = []
    analyses = state.get("analyses", [])
    debate = state.get("debate")
    decision = state.get("decision")
    risk = state.get("risk")
    execution = state.get("execution_result")
    disagreement = state.get("disagreement", 0.0)
    md = state.get("market_data")

    # 阶段 1: 数据采集
    price = ""
    klines_data = {}
    if md and md.klines:
        for tf, klines in md.klines.items():
            if klines:
                price = f"${klines[-1].close:,.2f}"
                # 发送最近 60 根 K 线用于前端绘图
                klines_data[tf] = [
                    {"t": k.timestamp, "o": k.open, "h": k.high, "l": k.low, "c": k.close, "v": k.volume}
                    for k in klines[-60:]
                ]
                events.append({
                    "stage": "collect", "agent": "system",
                    "content": f"数据采集完成 · {md.symbol} 最新价 {price}",
                    "klines": klines_data,
                    "ts": datetime.utcnow().isoformat(),
                })
                break
    if not price:
        events.append({
            "stage": "collect", "agent": "system",
            "content": "数据采集完成 (模拟数据)",
            "ts": datetime.utcnow().isoformat(),
        })

    # 阶段 2: 分析师
    for a in analyses:
        signals = []
        for s in (a.key_signals or [])[:3]:
            if isinstance(s, dict):
                signals.append(f"{s.get('signal') or s.get('name', '')} {s.get('value', '')}")
            elif hasattr(s, "signal"):
                signals.append(f"{s.signal} {s.value}")
            else:
                signals.append(str(s))

        events.append({
            "stage": "analyst",
            "agent": a.analyst,
            "stance": a.stance,
            "confidence": a.confidence,
            "content": a.core_reason or "",
            "signals": signals,
            "risks": a.risk_factors or [],
            "support_price": a.support_price,
            "resistance_price": a.resistance_price,
            "ts": datetime.utcnow().isoformat(),
        })

    # 分歧度
    events.append({
        "stage": "disagreement",
        "agent": "system",
        "content": f"分歧度 δ={disagreement:.3f}",
        "delta": disagreement,
        "ts": datetime.utcnow().isoformat(),
    })

    # 阶段 3: 辩论
    if debate:
        events.append({
            "stage": "debate",
            "agent": "system",
            "content": "辩论完成",
            "converged": debate.get("converged"),
            "downgrade": debate.get("consensus_confidence", 1.0),
            "agreements": debate.get("agreement_points", [])[:3],
            "disagreements": debate.get("disagreement_points", [])[:3],
            "ts": datetime.utcnow().isoformat(),
        })

    # 阶段 4: 决策
    if decision:
        rc = decision.reasoning_chain or {}
        events.append({
            "stage": "decision",
            "agent": "trader",
            "decision": decision.decision,
            "confidence": decision.confidence,
            "position_pct": decision.position_size_pct,
            "rr_ratio": decision.risk_reward_ratio,
            "entry": decision.entry,
            "stop_loss": decision.stop_loss,
            "take_profit": decision.take_profit,
            "data_cot": rc.get("data_cot", ""),
            "concept_cot": rc.get("concept_cot", ""),
            "thesis_cot": rc.get("thesis_cot", ""),
            "time_horizon": decision.time_horizon,
            "max_acceptable_loss_pct": decision.max_acceptable_loss_pct,
            "ts": datetime.utcnow().isoformat(),
        })

    # 阶段 5: 风控
    if risk:
        reviews = []
        for role in ("aggressive_review", "conservative_review", "neutral_review"):
            rv = getattr(risk, role, None)
            if rv:
                reviews.append({
                    "role": rv.role,
                    "decision": rv.decision,
                    "reason": rv.reason or "",
                    "concerns": rv.concerns or [],
                })
        cvar = getattr(risk, "cvar_check", None) or {}
        cvar_dict = None
        if cvar:
            cvar_dict = {
                "passed": cvar.passed if hasattr(cvar, "passed") else cvar.get("passed", False),
                "current_exposure": cvar.current_exposure if hasattr(cvar, "current_exposure") else cvar.get("current_exposure", 0),
                "threshold": cvar.cvar_threshold if hasattr(cvar, "cvar_threshold") else cvar.get("cvar_threshold", 0),
                "margin": cvar.margin if hasattr(cvar, "margin") else cvar.get("margin", 0),
            }

        events.append({
            "stage": "risk",
            "agent": "system",
            "approved": risk.approved,
            "vetoed_by": risk.vetoed_by or [],
            "reviews": reviews,
            "cvar_check": cvar_dict,
            "ts": datetime.utcnow().isoformat(),
        })

    # 执行
    if execution:
        events.append({
            "stage": "execution",
            "agent": "system",
            "status": execution.get("status", "unknown"),
            "detail": execution.get("reason", "模拟执行"),
            "ts": datetime.utcnow().isoformat(),
        })

    # 成本
    cost = 0.0
    tokens = 0
    try:
        import qmind.config  # noqa: F401 — 确保 .env 已加载
        from qmind.llm.client import LLMClient
        ct = LLMClient().cost_tracker
        cost = ct.total_cost()
        tokens = ct.total_tokens()
    except Exception:
        pass
    events.append({
        "stage": "cost",
        "agent": "system",
        "cost": round(cost, 4),
        "tokens": tokens,
        "ts": datetime.utcnow().isoformat(),
    })

    return events


async def _run_analysis(symbol: str, source: str) -> list[dict[str, Any]]:
    """运行一次完整分析"""
    # 强行确保 API Key 已加载
    _env_path = Path(__file__).resolve().parent.parent / ".env"
    if _env_path.exists():
        for line in _env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip()
                if k not in os.environ:
                    os.environ[k] = v

    from qmind.graph.pipeline import QMindPipeline
    from qmind.llm.client import LLMClient
    from qmind.execution.dry_run import DryRunExchange

    llm = LLMClient()
    exchange = DryRunExchange()
    pipeline = QMindPipeline(llm, exchange=exchange, data_source=source)

    state = await pipeline.run(symbol, "1h")
    events = _extract_events(state)

    # Record history
    decision = state.get("decision")
    risk = state.get("risk")
    _history.insert(0, {
        "symbol": symbol,
        "timestamp": datetime.utcnow().isoformat(),
        "decision": decision.decision if decision else None,
        "confidence": decision.confidence if decision else 0.0,
        "approved": risk.approved if risk else None,
    })

    # Audit log
    try:
        from qmind.audit_log import AuditLogger
        al = AuditLogger()
        if decision:
            al.log_decision(
                symbol=symbol,
                decision=decision.decision,
                confidence=decision.confidence,
                position_size_pct=decision.position_size_pct,
                entry_price=decision.entry.get("price", 0) if decision.entry else 0,
                stop_loss=decision.stop_loss.get("price", 0) if decision.stop_loss else 0,
                risk_reward_ratio=decision.risk_reward_ratio,
                approval=risk.approved if risk else False,
                vetoed_by=risk.vetoed_by if risk else [],
                token_usage={"cost": llm.cost_tracker.total_cost(), "tokens": llm.cost_tracker.total_tokens()},
                analyses=state.get("analyses", []),
                debate=state.get("debate"),
                risk=risk,
            )
    except Exception:
        pass

    return events


# ── API Routes ──

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the dashboard HTML"""
    return HTMLResponse(_DASHBOARD_HTML)


@app.get("/api/analyze")
async def analyze(
    symbol: str = Query("BTC/USDT"),
    source: str = Query("binance"),
):
    """运行分析，返回完整 JSON（非流式）"""
    global _last_result, _last_run, _running

    if _running:
        return JSONResponse({"status": "running"})

    _running = True
    try:
        events = await _run_analysis(symbol, source)
        _last_result = {"events": events, "symbol": symbol, "timestamp": datetime.utcnow().isoformat()}
        _last_run = time.time()
        return JSONResponse(_last_result)
    finally:
        _running = False


@app.get("/api/analyze/stream")
async def analyze_stream(
    symbol: str = Query("BTC/USDT"),
    source: str = Query("binance"),
):
    """SSE 流式推送：分析完成后逐个事件推送"""
    async def event_generator():
        global _running
        if _running:
            yield f"data: {json.dumps({'type': 'error', 'message': '已有分析任务在运行'})}\n\n"
            return

        _running = True
        try:
            yield f"data: {json.dumps({'type': 'meta', 'symbol': symbol, 'source': source})}\n\n"
            events = await _run_analysis(symbol, source)
            for ev in events:
                yield f"data: {json.dumps({'type': 'event', 'data': ev}, default=str)}\n\n"
                await asyncio.sleep(0.12)
            yield f"data: {json.dumps({'type': 'done'})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        finally:
            _running = False

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/api/status")
async def status():
    """返回当前状态（供前端轮询）"""
    return JSONResponse({
        "running": _running,
        "last_run": _last_run,
        "has_result": _last_result is not None,
        "last_result": _last_result,
    })


# ── History / Lessons / Audit / Debug API ──

_history: list[dict[str, Any]] = []


@app.get("/api/debug")
async def debug():
    """环境诊断"""
    import qmind.llm.client as c
    return JSONResponse({
        "DEEPSEEK_KEY_SET": bool(os.environ.get("DEEPSEEK_API_KEY")),
        "DEEPSEEK_KEY_PREFIX": (os.environ.get("DEEPSEEK_API_KEY", "")[:12] + "...") if os.environ.get("DEEPSEEK_API_KEY") else "NONE",
        "ANTHROPIC_KEY_SET": bool(os.environ.get("ANTHROPIC_API_KEY")),
        "OPENAI_KEY_SET": bool(os.environ.get("OPENAI_API_KEY")),
        "ALL_API_ENV": [k for k in sorted(os.environ.keys()) if "API" in k or "KEY" in k],
        "PROVIDER_MODELS": {p.value: c.PROVIDER_MODELS[p] for p in c.LLMProvider},
        "MODEL_PRICING_KEYS": list(c.MODEL_PRICING.keys()),
    })


@app.get("/api/history")
async def get_history(limit: int = Query(10, ge=1, le=50)):
    """最近分析历史"""
    return JSONResponse(_history[:limit])


@app.get("/api/lessons")
async def get_lessons(symbol: str = Query(""), limit: int = Query(5, ge=1, le=50)):
    """CVRF 教训"""
    from qmind.learning.memory import MemoryStore
    store = MemoryStore()
    if symbol:
        entries = store.get_by_symbol(symbol, limit=limit)
    else:
        entries = store.get_recent(limit=limit)
    result = []
    for e in entries:
        result.append({
            "id": e.id if hasattr(e, "id") else 0,
            "timestamp": e.timestamp.isoformat() if hasattr(e.timestamp, "isoformat") else str(e.timestamp),
            "symbol": e.symbol,
            "lessons": [
                {"lesson": l.lesson, "confidence": l.confidence, "source": l.source}
                for l in (e.lessons or [])
            ],
        })
    return JSONResponse(result)


@app.get("/api/audit")
async def get_audit(limit: int = Query(20, ge=1, le=100)):
    """审计日志"""
    from qmind.audit_log import AuditLogger
    logger = AuditLogger()
    records = logger.get_recent(limit=limit)
    # Convert sqlite3.Row to plain dicts
    return JSONResponse([dict(r) for r in records])


# ── Embedded HTML Dashboard ──

_DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>QMind Dashboard</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{font-family:'Segoe UI',-apple-system,sans-serif;background:#0d1117;color:#c9d1d9;height:100vh;display:flex;flex-direction:column}
  /* Header */
  .header{padding:14px 24px;background:#161b22;border-bottom:1px solid #30363d;display:flex;align-items:center;gap:16px;flex-shrink:0;flex-wrap:wrap}
  .header h1{font-size:18px;font-weight:600}
  .header h1 span{color:#58a6ff}
  .badge{font-size:11px;padding:2px 8px;border-radius:10px;background:#1f6feb33;color:#58a6ff;border:1px solid #1f6feb66}
  .status-dot{width:8px;height:8px;border-radius:50%;display:inline-block;margin-right:6px}
  .status-dot.idle{background:#3fb950}
  .status-dot.running{background:#d29922;animation:pulse 1s infinite}
  @keyframes pulse{0%,100%{opacity:1}50%{opacity:.4}}
  .ctrl{margin-left:auto;display:flex;gap:8px;align-items:center;flex-wrap:wrap}
  .ctrl input,.ctrl select{background:#0d1117;border:1px solid #30363d;color:#c9d1d9;padding:4px 8px;border-radius:6px;font-size:13px}
  .ctrl input:focus,.ctrl select:focus{outline:none;border-color:#58a6ff}
  .ctrl button{background:#238636;color:#fff;border:none;padding:4px 16px;border-radius:6px;cursor:pointer;font-size:13px;transition:background .15s}
  .ctrl button:hover{background:#2ea043}
  .ctrl button:disabled{opacity:.5;cursor:not-allowed}
  .ctrl button.danger{background:#da3633}
  .ctrl button.danger:hover{background:#f85149}
  .ctrl label{font-size:12px;color:#8b949e}
  /* Main */
  .main{flex:1;display:flex;overflow:hidden}
  .chat-panel{flex:1;overflow-y:auto;padding:12px 24px 32px;display:flex;flex-direction:column;gap:2px}
  .chat-panel::-webkit-scrollbar{width:6px}
  .chat-panel::-webkit-scrollbar-track{background:transparent}
  .chat-panel::-webkit-scrollbar-thumb{background:#30363d;border-radius:3px}
  /* Sidebar */
  .sidebar{width:280px;border-left:1px solid #30363d;padding:16px;display:flex;flex-direction:column;gap:12px;overflow-y:auto;flex-shrink:0}
  .sidebar h3{font-size:12px;text-transform:uppercase;color:#8b949e;letter-spacing:.05em}
  .sidebar .stat-row{display:flex;justify-content:space-between;font-size:12px;padding:2px 0}
  .sidebar .stat-row .label{color:#8b949e}
  .sidebar .stat-row .val{color:#c9d1d9;font-weight:600}
  /* Messages */
  .msg{margin:2px 0;max-width:100%}
  .msg-row{display:flex;gap:10px;align-items:flex-start;opacity:0;animation:fadeIn .3s ease forwards}
  @keyframes fadeIn{to{opacity:1}}
  .msg-avatar{width:28px;height:28px;border-radius:6px;display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0;margin-top:2px}
  .msg-body{flex:1;min-width:0}
  .msg-meta{font-size:11px;color:#8b949e;margin-bottom:2px}
  .msg-meta .name{font-weight:600}
  .msg-content{font-size:13px;line-height:1.5;white-space:pre-wrap;word-break:break-word}
  .typing-cursor{animation:blink .8s infinite}
  @keyframes blink{0%,100%{opacity:1}50%{opacity:0}}
  /* Avatars */
  .avatar-system{background:#21262d;color:#8b949e}
  .avatar-technical{background:#1f6feb33;color:#58a6ff}
  .avatar-fundamental{background:#23863633;color:#3fb950}
  .avatar-sentiment{background:#9e6a0333;color:#d29922}
  .avatar-news{background:#da363333;color:#f85149}
  .avatar-trader{background:#8957e533;color:#a371f7}
  /* Names */
  .name-system{color:#8b949e}
  .name-technical{color:#58a6ff}
  .name-fundamental{color:#3fb950}
  .name-sentiment{color:#d29922}
  .name-news{color:#f85149}
  .name-trader{color:#a371f7}
  /* Stance */
  .stance{font-size:10px;padding:1px 6px;border-radius:8px;margin-left:6px;font-weight:600}
  .stance.bullish{background:#23863633;color:#3fb950}
  .stance.bearish{background:#da363333;color:#f85149}
  .stance.neutral{background:#9e6a0333;color:#d29922}
  /* Chips */
  .chips{display:flex;flex-wrap:wrap;gap:4px;margin-top:4px}
  .chip{font-size:11px;padding:1px 8px;border-radius:8px;background:#21262d;color:#8b949e;border:1px solid #30363d}
  .chip.risk{border-color:#da363366;color:#f85149}
  /* Phase divider */
  .phase-divider{display:flex;align-items:center;gap:8px;margin:12px 0 4px;color:#8b949e;font-size:11px;text-transform:uppercase;letter-spacing:.05em}
  .phase-divider::after{content:'';flex:1;height:1px;background:#21262d}
  /* Decision card */
  .decision-card{background:#161b22;border:1px solid #30363d;border-radius:8px;padding:12px 16px;margin:4px 0}
  .decision-card .label{font-size:11px;color:#8b949e}
  .decision-card .value{font-size:14px;font-weight:600}
  .decision-card .entry-row{font-size:12px;color:#8b949e;margin-top:6px;display:flex;gap:16px;flex-wrap:wrap}
  /* Risk reviews */
  .review-row{display:flex;gap:8px;margin:2px 0;font-size:13px;align-items:center}
  .review-badge{font-size:10px;padding:1px 6px;border-radius:6px;font-weight:600;flex-shrink:0}
  .badge-approve{background:#23863633;color:#3fb950}
  .badge-reject{background:#da363333;color:#f85149}
  .badge-modify{background:#9e6a0333;color:#d29922}
  .review-concern{font-size:12px;color:#f85149;margin-left:48px;padding:1px 0}
  /* Heartbeat bar */
  .heartbeat-bar{position:fixed;bottom:0;left:0;right:0;height:3px;background:#21262d;z-index:100}
  .heartbeat-bar .fill{height:100%;background:#58a6ff;transition:width 1s linear;width:0%}
  /* Status bar */
  .status-bar{flex-shrink:0;padding:6px 24px;background:#161b22;border-top:1px solid #30363d;display:flex;gap:16px;font-size:12px;color:#8b949e;align-items:center}
  /* Loading dots */
  .loading-dots{display:inline-flex;gap:3px;margin-left:8px}
  .loading-dots span{width:5px;height:5px;border-radius:50%;background:#58a6ff;animation:dotBounce 1.2s infinite}
  .loading-dots span:nth-child(2){animation-delay:.2s}
  .loading-dots span:nth-child(3){animation-delay:.4s}
  @keyframes dotBounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-4px)}}
  /* CoT expand/collapse */
  .cot-toggle{color:#58a6ff;cursor:pointer;font-size:12px;text-decoration:none;margin-left:4px}
  .cot-toggle:hover{text-decoration:underline}
  /* Price levels */
  .price-level{font-size:11px;color:#8b949e;margin-top:3px;padding:2px 0;border-top:1px solid #21262d}
  .price-level b{color:#c9d1d9}
  /* Debate details */
  .debate-details summary{font-size:12px;padding:2px 0}
  .debate-details summary:hover{color:#c9d1d9}
  /* CVaR bar */
  .cvar-bar{margin-top:6px;font-size:11px;padding:6px 8px;border-radius:6px}
  .cvar-pass{background:#23863611;border:1px solid #23863633}
  .cvar-fail{background:#da363311;border:1px solid #da363333}
  .cvar-bar b{color:#c9d1d9}
  .cvar-fill-wrap{height:3px;background:#21262d;border-radius:2px;margin-top:4px;overflow:hidden}
  .cvar-fill{height:100%;border-radius:2px;transition:width .5s}
  /* Filter bar */
  .filter-bar{display:flex;gap:4px;padding:6px 24px;background:#0d1117;border-bottom:1px solid #21262d;flex-wrap:wrap;flex-shrink:0}
  .filter-chip{font-size:11px;padding:2px 10px;border-radius:10px;background:#21262d;color:#8b949e;cursor:pointer;border:1px solid #30363d;user-select:none;transition:all .15s}
  .filter-chip:hover{background:#30363d}
  .filter-chip.active{background:#1f6feb33;color:#58a6ff;border-color:#58a6ff}
  /* TradingView Chart */
  .tv-chart-wrap{background:#0d1117;border-bottom:1px solid #30363d;overflow:hidden;flex-shrink:0;max-height:480px;transition:max-height .3s}
  .tv-chart-wrap.collapsed{max-height:0;border-bottom:none}
  .tv-chart-toggle{padding:4px 24px;background:#0d1117;border-bottom:1px solid #21262d;cursor:pointer;font-size:12px;color:#8b949e;display:flex;align-items:center;gap:6px;flex-shrink:0;user-select:none}
  .tv-chart-toggle:hover{color:#c9d1d9}
  .tv-chart-toggle::before{content:'▾'}
  .tv-chart-toggle.collapsed::before{content:'▸'}
  /* KBD hint */
  .kbd-hint{font-size:11px;color:#484f58;margin-left:auto}
  .kbd-hint kbd{display:inline-block;padding:1px 4px;font-size:10px;font-family:inherit;background:#21262d;border:1px solid #30363d;border-radius:3px;margin:0 1px}
  /* Sidebar panels */
  .side-panel{margin-top:4px}
  .side-panel summary{cursor:pointer;color:#8b949e;font-size:12px;text-transform:uppercase;letter-spacing:.05em}
  .side-panel summary:hover{color:#c9d1d9}
  .side-panel .panel-body{font-size:12px;color:#8b949e;max-height:300px;overflow-y:auto}
  .lesson-card{background:#161b22;border:1px solid #30363d;border-radius:6px;padding:8px;margin:4px 0;font-size:12px}
  .lesson-card .lesson-text{color:#c9d1d9;margin-bottom:4px}
  .lesson-card .lesson-meta{color:#8b949e;font-size:10px}
  .lesson-confidence{height:3px;border-radius:2px;background:#58a6ff;margin-top:4px}
  /* Chart */
  .chart-wrap{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:8px;margin:4px 0}
  .chart-wrap svg{display:block;width:100%;height:auto}
  .audit-table{width:100%;font-size:11px;border-collapse:collapse}
  .audit-table th{text-align:left;color:#8b949e;padding:2px 4px;font-weight:600;border-bottom:1px solid #30363d}
  .audit-table td{padding:3px 4px;border-top:1px solid #21262d;cursor:pointer}
  .audit-table tr:hover td{background:#161b22}
  .audit-detail{display:none;padding:8px;background:#0d1117;font-size:11px;white-space:pre-wrap;max-height:300px;overflow-y:auto;border-radius:4px;margin:2px 0}
  .audit-detail.open{display:block}
  .history-item{display:flex;justify-content:space-between;align-items:center;padding:4px 6px;border-radius:4px;cursor:pointer;font-size:12px}
  .history-item:hover{background:#161b22}
  .history-item .hi-symbol{color:#c9d1d9;font-weight:600}
  .history-item .hi-decision{font-size:10px;padding:1px 5px;border-radius:4px;font-weight:600}

  /* Responsive */
  @media(max-width:768px){
    .sidebar{display:none}
    .header{padding:10px 12px}
    .ctrl{gap:4px}
    .ctrl input,.ctrl select{width:70px;font-size:12px}
    .chat-panel{padding:8px 12px 32px}
  }
</style>
<!-- KLineChart CDN -->
<script src="https://cdn.jsdelivr.net/npm/klinecharts/dist/klinecharts.min.js"></script>
</head>
<body>

<div class="header">
  <h1><span>Q</span>Mind</h1>
  <span class="badge">v0.1.0</span>
  <span id="statusBadge"><span class="status-dot idle"></span>空闲</span>
  <div class="ctrl">
    <label for="symbolInput">标的</label>
    <input id="symbolInput" value="BTC/USDT" size="10" spellcheck="false">
    <label for="sourceSelect">数据源</label>
    <select id="sourceSelect">
      <option value="binance" selected>Binance</option>
      <option value="mock">Mock</option>
      <option value="yfinance">Yahoo</option>
    </select>
    <label for="heartbeatInput">心跳(s)</label>
    <input id="heartbeatInput" value="120" size="3">
    <label for="speedSelect">速度</label>
    <select id="speedSelect">
      <option value="0">瞬间</option>
      <option value="5">快</option>
      <option value="12" selected>中</option>
      <option value="25">慢</option>
    </select>
    <button onclick="alert('runNow='+typeof runNow+'|toggleFilter='+typeof toggleFilter)" title="诊断" style="font-size:11px;padding:2px 8px;background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;cursor:pointer;">🔍</button>
    <button onclick="copyAsMarkdown()" title="Copy as Markdown" style="font-size:11px;padding:2px 8px;background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;cursor:pointer;">MD</button>
    <button onclick="copyAsJSON()" title="Copy as JSON" style="font-size:11px;padding:2px 8px;background:#21262d;color:#8b949e;border:1px solid #30363d;border-radius:4px;cursor:pointer;">JSON</button>
    <button id="runBtn" onclick="runNow()">▶ 运行</button>
    <button id="autoBtn" onclick="toggleAuto()">⏸ 自动</button>
  </div>
</div>

<div class="filter-bar" id="filterBar">
  <span class="filter-chip active" data-filter="all" onclick="toggleFilter(this,'all')">全部</span>
  <span class="filter-chip" data-filter="collect" onclick="toggleFilter(this,'collect')">采集</span>
  <span class="filter-chip" data-filter="analyst" onclick="toggleFilter(this,'analyst')">分析</span>
  <span class="filter-chip" data-filter="debate" onclick="toggleFilter(this,'debate')">辩论</span>
  <span class="filter-chip" data-filter="decision" onclick="toggleFilter(this,'decision')">决策</span>
  <span class="filter-chip" data-filter="risk" onclick="toggleFilter(this,'risk')">风控</span>
</div>

<div class="tv-chart-toggle" id="tvToggle" onclick="toggleTvChart()">
  📊 实时行情图表 <span id="tvSymbol" style="color:#58a6ff;">BTC/USDT</span>
</div>
<div class="tv-chart-wrap" id="tvChartWrap">
  <div id="tvChart" style="height:420px;"></div>
</div>

<div class="main">
  <div class="chat-panel" id="chatPanel"></div>
  <div class="sidebar" id="sidebar">
    <h3>上次结果</h3>
    <div id="resultSummary" style="font-size:12px;color:#8b949e;">暂无</div>
    <h3 style="margin-top:8px">心跳</h3>
    <div id="heartbeatInfo" style="font-size:12px;color:#8b949e;">等待下次运行...</div>

    <details class="side-panel" id="lessonsPanel">
      <summary>CVRF Lessons <span id="lessonCount" style="color:#8b949e;"></span></summary>
      <div class="panel-body" id="lessonsBody"><div style="color:#8b949e;padding:4px 0;">加载中...</div></div>
    </details>

    <details class="side-panel" id="historyPanel">
      <summary>历史记录</summary>
      <div class="panel-body" id="historyBody"><div style="color:#8b949e;padding:4px 0;">暂无</div></div>
    </details>

    <details class="side-panel" id="auditPanel">
      <summary>审计日志</summary>
      <div class="panel-body" id="auditBody"><div style="color:#8b949e;padding:4px 0;">暂无</div></div>
    </details>

    <h3 style="margin-top:8px">服务器</h3>
    <div class="stat-row"><span class="label">端口</span><span class="val">8000</span></div>
    <div class="stat-row"><span class="label">状态</span><span class="val" id="serverStatus">在线</span></div>
  </div>
</div>

<div class="heartbeat-bar"><div class="fill" id="heartFill"></div></div>
<div class="status-bar">
  <span id="statusText">就绪 — 点击运行或开启自动模式</span>
  <span id="costDisplay" style="margin-left:auto;"></span>
</div>

<script>window.onerror=function(m,u,l,c){alert('JS Error at line '+l+': '+m)};console.log('guard loaded')</script>
<script>
// ── State ──
const chatPanel = document.getElementById('chatPanel');
let autoMode = false;
let heartbeatTimer = null;
let heartFillTimer = null;
let heartbeatInterval = 120;
let isRunning = false;
let streamAbort = null; // AbortController for SSE fetch

const AGENTS = {
  system:      { avatar: '⚙️', css: 'system' },
  technical:   { avatar: '📈', css: 'technical' },
  fundamental: { avatar: '📊', css: 'fundamental' },
  sentiment:   { avatar: '💬', css: 'sentiment' },
  news:        { avatar: '📰', css: 'news' },
  trader:      { avatar: '🤖', css: 'trader' },
};

const AGENT_NAMES = {
  technical: '技术面分析师',
  fundamental: '基本面分析师',
  sentiment: '情绪分析师',
  news: '宏观分析师',
  trader: '交易员',
  system: '系统',
};

const STANCE_EMOJI = { bullish: '🟢', neutral: '🟡', bearish: '🔴' };
const PHASE_LABELS = {
  collect: '📊 数据采集',
  analyst: '📈 多维分析',
  disagreement: '⚡ 分歧检测',
  debate: '🎯 多空辩论',
  decision: '🧠 交易决策',
  risk: '🛡️ 风控审核',
  execution: '▶ 执行',
};

// ── Typewriter ──
async function typeText(el, text, speed) {
  if (speed === 0) { el.textContent = text; return; }
  el.textContent = '';
  const pauseChars = new Set(['。','！','？','；','.', '!', '?', ';', '\n']);
  for (let i = 0; i < text.length; i++) {
    el.textContent += text[i];
    const delay = pauseChars.has(text[i]) ? speed * 5 : speed;
    if (i % 4 === 0) await new Promise(r => setTimeout(r, delay));
    chatPanel.scrollTop = chatPanel.scrollHeight;
  }
  const cursor = document.createElement('span');
  cursor.className = 'typing-cursor'; cursor.textContent = '▌';
  el.after(cursor);
  await new Promise(r => setTimeout(r, 300));
  cursor.remove();
}

function getTypeSpeed() {
  const sel = document.getElementById('speedSelect');
  return sel ? parseInt(sel.value) : 12;
}

function addPhaseDivider(label) {
  if (chatPanel.querySelector(`[data-phase="${label}"]`)) return;
  const div = document.createElement('div');
  div.className = 'phase-divider';
  div.dataset.phase = label;
  div.textContent = label;
  chatPanel.appendChild(div);
  chatPanel.scrollTop = chatPanel.scrollHeight;
}

async function addMessage(event) {
  const stage = event.stage || '';
  const agent = event.agent || 'system';
  const cfg = AGENTS[agent] || AGENTS.system;

  // Phase dividers — one per stage
  if (stage === 'collect') addPhaseDivider(PHASE_LABELS.collect);
  if (stage === 'analyst') addPhaseDivider(PHASE_LABELS.analyst);
  if (stage === 'disagreement') addPhaseDivider(PHASE_LABELS.disagreement);
  if (stage === 'debate') addPhaseDivider(PHASE_LABELS.debate);
  if (stage === 'decision') addPhaseDivider(PHASE_LABELS.decision);
  if (stage === 'risk') addPhaseDivider(PHASE_LABELS.risk);
  if (stage === 'execution') addPhaseDivider(PHASE_LABELS.execution);

  const row = document.createElement('div');
  row.className = 'msg';
  if (stage) row.dataset.stage = stage;

  const avatar = document.createElement('div');
  avatar.className = 'msg-avatar avatar-' + cfg.css;
  avatar.textContent = cfg.avatar;

  const body = document.createElement('div');
  body.className = 'msg-body';

  // Meta line
  const meta = document.createElement('div');
  meta.className = 'msg-meta';
  const nameSpan = document.createElement('span');
  nameSpan.className = 'name name-' + cfg.css;
  nameSpan.textContent = AGENT_NAMES[agent] || agent;
  meta.appendChild(nameSpan);

  if (event.stance) {
    const badge = document.createElement('span');
    badge.className = 'stance ' + event.stance;
    badge.textContent = (STANCE_EMOJI[event.stance] || '') + ' ' + event.stance + ' (' + Math.round((event.confidence || 0) * 100) + '%)';
    meta.appendChild(badge);
  }
  body.appendChild(meta);

  // Content with typewriter
  const content = document.createElement('div');
  content.className = 'msg-content';
  if (event.content) {
    const typingSpan = document.createElement('span');
    content.appendChild(typingSpan);
    body.appendChild(content);
    row.appendChild(avatar);
    row.appendChild(body);
    chatPanel.appendChild(row);
    await typeText(typingSpan, event.content, getTypeSpeed());
  }

  // Price levels
  if (event.support_price !== undefined && event.support_price !== null && event.support_price !== 0) {
    const plDiv = document.createElement('div');
    plDiv.className = 'price-level';
    plDiv.innerHTML = '📍 支撑: <b>$' + Number(event.support_price).toLocaleString() + '</b> | 阻力: <b>$' + (event.resistance_price ? Number(event.resistance_price).toLocaleString() : 'N/A') + '</b>';
    body.appendChild(plDiv);
  }

  // Signals
  if (event.signals && event.signals.length) {
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const s of event.signals) {
      if (!s) continue;
      const chip = document.createElement('span');
      chip.className = 'chip';
      chip.textContent = s.length > 60 ? s.slice(0, 60) + '…' : s;
      chips.appendChild(chip);
    }
    body.appendChild(chips);
  }

  // Risk factors
  if (event.risks && event.risks.length) {
    const chips = document.createElement('div');
    chips.className = 'chips';
    for (const r of event.risks) {
      const chip = document.createElement('span');
      chip.className = 'chip risk';
      chip.textContent = r.length > 60 ? r.slice(0, 60) + '…' : r;
      chips.appendChild(chip);
    }
    body.appendChild(chips);
  }

  // K-line chart (on collect event)
  if (event.klines) {
    const chartId = 'chart_' + Math.random().toString(36).slice(2, 7);
    const chartWrap = document.createElement('div');
    chartWrap.className = 'chart-wrap';
    chartWrap.id = chartId;
    body.appendChild(chartWrap);
    // Defer render to allow DOM insertion
    setTimeout(() => renderChart(event.klines, chartId), 50);
  }

  // Debate — collapsible details
  if (event.agreements && event.agreements.length) {
    const details = document.createElement('details');
    details.className = 'debate-details';
    details.style.cssText = 'margin-top:4px;font-size:12px;';
    const summary = document.createElement('summary');
    summary.style.cssText = 'cursor:pointer;color:#8b949e;';
    summary.textContent = '🎯 辩论 (共识 ' + event.agreements.length + ' · 分歧 ' + (event.disagreements ? event.disagreements.length : 0) + ')';
    details.appendChild(summary);
    const inner = document.createElement('div');
    inner.style.cssText = 'padding:4px 0 0 8px;';
    let html = '<div><span style="color:#3fb950;">✓ 共识:</span></div>';
    event.agreements.forEach(p => { html += '<div style="margin-left:12px;color:#8b949e;">\xb7 ' + escHtml(p) + '</div>'; });
    if (event.disagreements && event.disagreements.length) {
      html += '<div style="margin-top:4px;"><span style="color:#f85149;">✗ 分歧:</span></div>';
      event.disagreements.forEach(p => { html += '<div style="margin-left:12px;color:#8b949e;">\xb7 ' + escHtml(p) + '</div>'; });
    }
    inner.innerHTML = html;
    details.appendChild(inner);
    body.appendChild(details);
  }

  // Conviction
  if (event.downgrade !== undefined) {
    const div = document.createElement('div');
    div.style.cssText = 'margin-top:4px;font-size:12px;';
    div.innerHTML = '降级因子: <b>' + event.downgrade.toFixed(2) + '</b> &middot; 收敛: ' + (event.converged === null ? 'N/A' : event.converged ? '✓' : '✗');
    body.appendChild(div);
  }

  // Decision card
  if (event.decision) {
    const card = document.createElement('div');
    card.className = 'decision-card';
    const dec = event.decision;
    const decColor = dec === 'LONG' ? '#3fb950' : dec === 'SHORT' ? '#f85149' : '#d29922';
    const entry = event.entry || {};
    const sl = event.stop_loss || {};
    // Enhanced entry info with type, quantity, time_horizon, max_loss
    let entryHtml = '<div class="entry-row">';
    if (entry.price) entryHtml += '<span>入场: ' + (entry.type || '') + ' $' + Number(entry.price).toLocaleString() + (entry.quantity ? ' x' + entry.quantity : '') + '</span>';
    if (sl.price) entryHtml += '<span>止损: $' + Number(sl.price).toLocaleString() + '</span>';
    entryHtml += '</div>';
    if (event.time_horizon || event.max_acceptable_loss_pct) {
      entryHtml += '<div class="entry-row">';
      if (event.time_horizon) entryHtml += '<span>时间框架: ' + escHtml(event.time_horizon) + '</span>';
      if (event.max_acceptable_loss_pct) entryHtml += '<span>最大亏损: ' + Number(event.max_acceptable_loss_pct).toFixed(1) + '%</span>';
      entryHtml += '</div>';
    }
    card.innerHTML = '<div style="display:flex;gap:16px;flex-wrap:wrap;">'
      + '<div><span class="label">决策</span><br><span class="value" style="color:' + decColor + ';">' + dec + '</span></div>'
      + '<div><span class="label">置信度</span><br><span class="value">' + (event.confidence || 0).toFixed(2) + '</span></div>'
      + '<div><span class="label">仓位</span><br><span class="value">' + (event.position_pct || 0).toFixed(1) + '%</span></div>'
      + '<div><span class="label">盈亏比</span><br><span class="value">' + (event.rr_ratio || 0).toFixed(2) + '</span></div>'
      + '</div>' + entryHtml;
    body.appendChild(card);

    // CoT with expand/collapse
    const cotFields = [
      { key: 'data_cot', label: 'Data-CoT', color: '#58a6ff' },
      { key: 'concept_cot', label: 'Concept-CoT', color: '#d29922' },
      { key: 'thesis_cot', label: 'Thesis-CoT', color: '#a371f7' },
    ];
    for (const f of cotFields) {
      const val = event[f.key];
      if (!val) continue;
      const cotDiv = document.createElement('div');
      cotDiv.style.cssText = 'margin-top:6px;font-size:12px;color:#8b949e;';
      const header = document.createElement('span');
      header.innerHTML = '<b style="color:' + f.color + ';">' + f.label + ':</b> ';
      cotDiv.appendChild(header);
      const preview = document.createElement('span');
      preview.textContent = val.length > 200 ? val.slice(0, 200) + '…' : val;
      cotDiv.appendChild(preview);
      if (val.length > 200) {
        const toggle = document.createElement('a');
        toggle.className = 'cot-toggle';
        toggle.textContent = '[展开]';
        toggle.href = '#';
        toggle.onclick = function(e) {
          e.preventDefault();
          if (preview.textContent.length > 200) {
            preview.textContent = val;
            toggle.textContent = '[收起]';
          } else {
            preview.textContent = val.slice(0, 200) + '…';
            toggle.textContent = '[展开]';
          }
        };
        cotDiv.appendChild(document.createTextNode(' '));
        cotDiv.appendChild(toggle);
      }
      body.appendChild(cotDiv);
    }
  }

  // Risk reviews
  if (event.reviews && event.reviews.length) {
    for (const rv of event.reviews) {
      const rdiv = document.createElement('div');
      rdiv.className = 'review-row';
      const badge = document.createElement('span');
      badge.className = 'review-badge badge-' + rv.decision;
      badge.textContent = rv.decision.toUpperCase();
      rdiv.appendChild(badge);
      rdiv.appendChild(document.createTextNode('[' + rv.role + '] ' + (rv.reason || '').slice(0, 120)));
      body.appendChild(rdiv);
      if (rv.concerns && rv.concerns.length) {
        for (const c of rv.concerns.slice(0, 3)) {
          const cdiv = document.createElement('div');
          cdiv.className = 'review-concern';
          cdiv.textContent = '⚠ ' + (c.length > 80 ? c.slice(0, 80) + '…' : c);
          body.appendChild(cdiv);
        }
      }
    }
    const verdict = document.createElement('div');
    verdict.style.cssText = 'margin-top:4px;font-weight:600;font-size:13px;';
    verdict.innerHTML = event.approved
      ? '<span style="color:#3fb950;">✅ 风控通过</span>'
      : '<span style="color:#f85149;">❌ 风控否决 (' + escHtml((event.vetoed_by || []).join(', ')) + ')</span>';
    body.appendChild(verdict);

    // CVaR details
    if (event.cvar_check) {
      const cv = event.cvar_check;
      const cvDiv = document.createElement('div');
      cvDiv.className = 'cvar-bar ' + (cv.passed ? 'cvar-pass' : 'cvar-fail');
      const pct = cv.threshold > 0 ? Math.min(100, (cv.current_exposure / cv.threshold) * 100) : 0;
      cvDiv.innerHTML = '<span>CVaR 敞口: <b>$' + Number(cv.current_exposure).toLocaleString() + '</b> / ' + Number(cv.threshold).toLocaleString() + ' · 余量: $' + Number(cv.margin).toLocaleString() + ' · ' + (cv.passed ? '✅ 通过' : '❌ 超限') + '</span>'
        + '<div class="cvar-fill-wrap"><div class="cvar-fill" style="width:' + pct + '%;background:' + (cv.passed ? '#3fb950' : '#f85149') + ';"></div></div>';
      body.appendChild(cvDiv);
    }
  }

  // Execution
  if (event.status) {
    const div = document.createElement('div');
    div.style.cssText = 'margin-top:4px;font-size:13px;';
    const statusColors = { dry_run: '#8b949e', live: '#3fb950', rejected: '#f85149', error: '#f85149' };
    div.innerHTML = '执行: <span style="color:' + (statusColors[event.status] || '#8b949e') + ';">' + event.status + ' &middot; ' + escHtml(event.detail || '') + '</span>';
    body.appendChild(div);
  }

  // Cost
  if (event.cost !== undefined) {
    document.getElementById('costDisplay').textContent = '$' + event.cost.toFixed(4) + ' \xb7 ' + event.tokens + ' tokens';
  }

  // If no content was rendered above, still append the row
  if (!event.content && !event.decision && !event.reviews && !event.status && event.cost === undefined) {
    body.appendChild(content);
    row.appendChild(avatar);
    row.appendChild(body);
    chatPanel.appendChild(row);
  }

  chatPanel.scrollTop = chatPanel.scrollHeight;
  await new Promise(r => setTimeout(r, 150));
}

function escHtml(s) {
  if (typeof s !== 'string') return String(s || '');
  return s.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ── SVG Candlestick Chart ──
function renderChart(klines, containerId) {
  const container = document.getElementById(containerId);
  if (!container || !klines) return;
  const tf = Object.keys(klines)[0];
  const bars = klines[tf];
  if (!bars || bars.length < 2) { container.innerHTML = '<div style="color:#8b949e;font-size:12px;padding:4px;">数据不足</div>'; return; }

  const W = 600, H = 240, PAD = 40;
  const count = Math.min(bars.length, 60);
  const data = bars.slice(-count);
  const maxP = Math.max(...data.map(b => b.h));
  const minP = Math.min(...data.map(b => b.l));
  const range = maxP - minP || 1;
  const scaleY = (p) => PAD + (1 - (p - minP) / range) * (H - 2 * PAD);
  const bw = (W - PAD * 2) / count * 0.7;
  const gap = (W - PAD * 2) / count;

  let svg = '<svg viewBox="0 0 ' + W + ' ' + H + '" xmlns="http://www.w3.org/2000/svg">';
  // Gridlines
  for (let i = 0; i <= 4; i++) {
    const y = PAD + (H - 2 * PAD) * i / 4;
    const p = (maxP - range * i / 4).toFixed(0);
    svg += '<line x1="' + PAD + '" y1="' + y + '" x2="' + (W - 5) + '" y2="' + y + '" stroke="#21262d" stroke-width="1"/>';
    svg += '<text x="' + (PAD - 4) + '" y="' + (y + 3) + '" fill="#484f58" font-size="10" text-anchor="end">' + p + '</text>';
  }
  // Candles
  for (let i = 0; i < count; i++) {
    const b = data[i];
    const x = PAD + i * gap + (gap - bw) / 2;
    const isUp = b.c >= b.o;
    const color = isUp ? '#3fb950' : '#f85149';
    const top = Math.min(b.o, b.c);
    const bot = Math.max(b.o, b.c);
    // Wick
    svg += '<line x1="' + (x + bw / 2) + '" y1="' + scaleY(b.h) + '" x2="' + (x + bw / 2) + '" y2="' + scaleY(b.l) + '" stroke="' + color + '" stroke-width="1"/>';
    // Body
    svg += '<rect x="' + x + '" y="' + scaleY(top) + '" width="' + bw + '" height="' + Math.max(1, scaleY(bot) - scaleY(top)) + '" fill="' + color + '"/>';
  }
  svg += '</svg>';
  container.innerHTML = svg;
}

// ── Main: SSE streaming (with JSON fallback) ──
async function runNow() {
  if (isRunning) return;
  isRunning = true;

  const symbol = document.getElementById('symbolInput').value.trim() || 'BTC/USDT';
  const source = document.getElementById('sourceSelect').value;

  document.getElementById('runBtn').disabled = true;
  document.getElementById('statusBadge').innerHTML = '<span class="status-dot running"></span>运行中...';
  document.getElementById('statusText').textContent = symbol + ' 分析中...';

  // Clear previous
  chatPanel.innerHTML = '';
  streamAbort = new AbortController();
  let lastCost = null;
  let fallbackMode = false;

  try {
    // Try SSE streaming first
    const resp = await fetch('/api/analyze/stream?symbol=' + encodeURIComponent(symbol) + '&source=' + encodeURIComponent(source) + '&_=' + Date.now(), {
      signal: streamAbort.signal,
    });
    if (!resp.ok) throw new Error('HTTP ' + resp.status);
    if (!resp.body) throw new Error('No ReadableStream');

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let _accumulatedEvents = [];

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const msg = JSON.parse(line.slice(6));

        if (msg.type === 'meta') {
          document.getElementById('statusText').textContent = msg.symbol + ' 分析中...';
        } else if (msg.type === 'event') {
          if (autoMode && document.querySelector('#autoBtn').textContent === '▶ 自动') break;
          await addMessage(msg.data);
          _accumulatedEvents.push(msg.data);
          if (msg.data.stage === 'cost') lastCost = msg.data;
        } else if (msg.type === 'done') {
          _lastEvents = _accumulatedEvents;
          if (lastCost) {
            document.getElementById('resultSummary').innerHTML =
              '<div style="margin-bottom:4px;">' + symbol + '</div>'
              + '<div style="color:#8b949e;">' + lastCost.tokens + ' tokens</div>'
              + '<div style="font-size:16px;font-weight:600;">$' + lastCost.cost.toFixed(4) + '</div>';
          }
          _lastSymbol = symbol;
          document.getElementById('statusBadge').innerHTML = '<span class="status-dot idle"></span>空闲';
          document.getElementById('statusText').textContent = '上次: ' + symbol;
        } else if (msg.type === 'error') {
          document.getElementById('statusText').textContent = '错误: ' + msg.message;
          document.getElementById('statusBadge').innerHTML = '<span class="status-dot idle"></span>错误';
        }
      }
    }
  } catch (err) {
    if (err.name === 'AbortError') return;
    // Fallback to JSON endpoint
    console.warn('SSE failed, falling back to JSON:', err.message);
    fallbackMode = true;
    try {
      const resp = await fetch('/api/analyze?symbol=' + encodeURIComponent(symbol) + '&source=' + encodeURIComponent(source) + '&_=' + Date.now());
      const data = await resp.json();
      if (data.status === 'running') {
        document.getElementById('statusText').textContent = '已有任务在运行';
        return;
      }
      chatPanel.innerHTML = '';
      const events = data.events || [];
      for (const evt of events) {
        if (autoMode && document.querySelector('#autoBtn').textContent === '▶ 自动') break;
        await addMessage(evt);
      }
      _lastEvents = events;
      _lastSymbol = data.symbol || symbol;
      const last = events[events.length - 1];
      if (last && last.cost !== undefined) {
        lastCost = last;
        document.getElementById('resultSummary').innerHTML =
          '<div style="margin-bottom:4px;">' + (data.symbol || symbol) + '</div>'
          + '<div style="color:#8b949e;">' + last.tokens + ' tokens</div>'
          + '<div style="font-size:16px;font-weight:600;">$' + last.cost.toFixed(4) + '</div>';
      }
      document.getElementById('statusBadge').innerHTML = '<span class="status-dot idle"></span>空闲';
      document.getElementById('statusText').textContent = '上次: ' + (data.symbol || symbol);
    } catch (fallbackErr) {
      document.getElementById('statusText').textContent = '错误: ' + fallbackErr.message;
      document.getElementById('statusBadge').innerHTML = '<span class="status-dot idle"></span>错误';
    }
  } finally {
    isRunning = false;
    document.getElementById('runBtn').disabled = false;
    streamAbort = null;
  }
}

// ── Auto / Heartbeat ──
function toggleAuto() {
  autoMode = !autoMode;
  const btn = document.getElementById('autoBtn');
  if (autoMode) {
    btn.textContent = '▶ 自动';
    btn.style.background = '#da3633';
    heartbeatInterval = parseInt(document.getElementById('heartbeatInput').value) || 120;
    // Run first analysis immediately
    runNow().then(() => { scheduleNext(); });
  } else {
    btn.textContent = '⏸ 自动';
    btn.style.background = '#238636';
    clearTimeout(heartbeatTimer);
    clearInterval(heartFillTimer);
    document.getElementById('heartFill').style.width = '0%';
    document.getElementById('heartbeatInfo').textContent = '已暂停';
  }
}

function scheduleNext() {
  if (!autoMode) return;
  clearTimeout(heartbeatTimer);
  clearInterval(heartFillTimer);

  const totalMs = heartbeatInterval * 1000;
  const start = Date.now();
  document.getElementById('heartFill').style.width = '0%';
  document.getElementById('heartbeatInfo').textContent = '下次心跳: ' + heartbeatInterval + 's';

  heartFillTimer = setInterval(() => {
    const elapsed = Date.now() - start;
    const pct = Math.min(100, (elapsed / totalMs) * 100);
    document.getElementById('heartFill').style.width = pct + '%';
    document.getElementById('heartbeatInfo').textContent = '下次心跳: ' + Math.max(0, Math.round((totalMs - elapsed) / 1000)) + 's';
    if (pct >= 100) clearInterval(heartFillTimer);
  }, 200);

  heartbeatTimer = setTimeout(async () => {
    if (!autoMode) return;
    await runNow();
    scheduleNext();
  }, totalMs);
}

// ── Global state for copy/export ──
let _lastEvents = null;
let _lastSymbol = '';

// ── Filter bar ──
let _activeStageFilters = new Set();
function toggleFilter(el, stage) {
  if (stage === 'all') {
    document.querySelectorAll('.filter-chip').forEach(c => c.classList.remove('active'));
    el.classList.add('active');
    _activeStageFilters = new Set();
    document.querySelectorAll('.msg').forEach(m => m.style.display = '');
    return;
  }
  document.querySelector('.filter-chip[data-filter="all"]').classList.remove('active');
  el.classList.toggle('active');
  if (_activeStageFilters.has(stage)) _activeStageFilters.delete(stage);
  else _activeStageFilters.add(stage);
  applyFilters();
}
function applyFilters() {
  if (_activeStageFilters.size === 0) {
    document.querySelectorAll('.msg').forEach(m => m.style.display = '');
    return;
  }
  document.querySelectorAll('.msg').forEach(m => {
    const stages = m.dataset.stage || '';
    m.style.display = _activeStageFilters.has(stages) ? '' : 'none';
  });
}

// ── Keyboard shortcuts ──
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
  if (e.key === 'r' || e.key === 'R') { e.preventDefault(); runNow(); }
  if (e.key === 'a' || e.key === 'A') { e.preventDefault(); toggleAuto(); }
  if (e.key === 'Escape' && isRunning) {
    if (streamAbort) { streamAbort.abort(); streamAbort = null; }
    isRunning = false;
    document.getElementById('runBtn').disabled = false;
    document.getElementById('statusBadge').innerHTML = '<span class="status-dot idle"></span>已停止';
    document.getElementById('statusText').textContent = '已停止';
  }
});

// ── Copy / Export ──
function copyAsMarkdown() {
  if (!_lastEvents || !_lastEvents.length) { showToast('暂无分析结果'); return; }
  let md = '# QMind 分析报告 — ' + _lastSymbol + '\n\n';
  for (const ev of _lastEvents) {
    if (ev.content) md += '**' + (AGENT_NAMES[ev.agent] || ev.agent) + '**: ' + ev.content + '\n\n';
    if (ev.decision) md += '**决策**: ' + ev.decision + ' | 置信度: ' + ev.confidence + ' | 仓位: ' + ev.position_pct + '% | 盈亏比: ' + ev.rr_ratio + '\n\n';
  }
  navigator.clipboard.writeText(md).then(() => showToast('已复制 Markdown')).catch(() => {});
}
function copyAsJSON() {
  if (!_lastEvents) { showToast('暂无分析结果'); return; }
  navigator.clipboard.writeText(JSON.stringify({symbol: _lastSymbol, events: _lastEvents}, null, 2))
    .then(() => showToast('已复制 JSON')).catch(() => {});
}
function showToast(msg) {
  const el = document.getElementById('statusText');
  const orig = el.textContent;
  el.textContent = '✓ ' + msg;
  setTimeout(() => { el.textContent = orig; }, 2000);
}

// ── Sidebar: Lessons ──
async function loadLessons(symbol) {
  const body = document.getElementById('lessonsBody');
  try {
    const resp = await fetch('/api/lessons?symbol=' + encodeURIComponent(symbol) + '&limit=5&_=' + Date.now());
    const data = await resp.json();
    document.getElementById('lessonCount').textContent = data.length ? '(' + data.length + ')' : '';
    if (!data.length) { body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">暂无教训</div>'; return; }
    let html = '';
    for (const l of data) {
      const lessons = l.lessons || [];
      html += '<div class="lesson-card">';
      for (const les of lessons.slice(0, 2)) {
        html += '<div class="lesson-text">💡 ' + escHtml((les.lesson || '').slice(0, 120)) + '</div>';
        if (les.confidence) html += '<div class="lesson-confidence" style="width:' + (les.confidence * 100) + '%;"></div>';
      }
      html += '<div class="lesson-meta">' + escHtml(l.symbol || '') + ' · ' + escHtml((l.timestamp || '').slice(0, 10)) + '</div>';
      html += '</div>';
    }
    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">加载失败</div>';
  }
}

// ── Sidebar: History ──
async function loadHistory() {
  const body = document.getElementById('historyBody');
  try {
    const resp = await fetch('/api/history?limit=10&_=' + Date.now());
    const data = await resp.json();
    if (!data.length) { body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">暂无</div>'; return; }
    let html = '';
    for (const h of data) {
      const dc = {LONG:'#3fb950',SHORT:'#f85149',HOLD:'#d29922'}[h.decision] || '#8b949e';
      html += '<div class="history-item" onclick="document.getElementById(\'symbolInput\').value=\'' + escHtml(h.symbol) + '\';runNow();">'
        + '<span class="hi-symbol">' + escHtml(h.symbol) + '</span>'
        + '<span><span class="hi-decision" style="color:' + dc + ';">' + (h.decision || '—') + '</span>'
        + ' <span style="color:#8b949e;font-size:10px;">' + Number(h.confidence || 0).toFixed(2) + '</span></span></div>';
    }
    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">加载失败</div>';
  }
}

// ── Sidebar: Audit ──
async function loadAudit() {
  const body = document.getElementById('auditBody');
  try {
    const resp = await fetch('/api/audit?limit=20&_=' + Date.now());
    const data = await resp.json();
    if (!data.length) { body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">暂无</div>'; return; }
    let html = '<table class="audit-table"><tr><th>时间</th><th>标的</th><th>决策</th><th>置信</th><th>结果</th></tr>';
    for (const r of data) {
      const dc = {LONG:'#3fb950',SHORT:'#f85149',HOLD:'#d29922'}[r.decision] || '#8b949e';
      const app = r.approval ? '<span style="color:#3fb950;">✓</span>' : r.approval === 0 ? '<span style="color:#f85149;">✗</span>' : '—';
      const ts = (r.timestamp || '').slice(0, 16).replace('T', ' ');
      html += '<tr onclick="toggleAuditDetail(this)">'
        + '<td>' + escHtml(ts) + '</td>'
        + '<td>' + escHtml(r.symbol || '') + '</td>'
        + '<td style="color:' + dc + ';">' + (r.decision || '—') + '</td>'
        + '<td>' + (r.confidence || 0).toFixed(2) + '</td>'
        + '<td>' + app + '</td></tr>'
        + '<tr style="display:none;"><td colspan="5"><div class="audit-detail open">' + escHtml(JSON.stringify(r, null, 2)) + '</div></td></tr>';
    }
    html += '</table>';
    body.innerHTML = html;
  } catch (e) {
    body.innerHTML = '<div style="color:#8b949e;padding:4px 0;">加载失败</div>';
  }
}
function toggleAuditDetail(row) {
  const detailRow = row.nextElementSibling;
  if (detailRow && detailRow.style.display === 'none') {
    detailRow.style.display = '';
  } else if (detailRow) {
    detailRow.style.display = 'none';
  }
}

// ── Init ──
window.addEventListener('DOMContentLoaded', () => {
  addPhaseDivider('💡 就绪 — 点击运行或开启自动模式');

  // Load sidebar data
  loadHistory();
  loadAudit();

  // Open lessons panel when lessons arrive
  document.getElementById('lessonsPanel').addEventListener('toggle', () => {
    if (document.getElementById('lessonsPanel').open) {
      loadLessons(document.getElementById('symbolInput').value.trim() || 'BTC/USDT');
    }
  });
  document.getElementById('historyPanel').addEventListener('toggle', () => {
    if (document.getElementById('historyPanel').open) loadHistory();
  });
  document.getElementById('auditPanel').addEventListener('toggle', () => {
    if (document.getElementById('auditPanel').open) loadAudit();
  });
});

// ── Chart: KLineChart + Binance API ──
var _kc = null, _kcInt = null;

function toggleTvChart() {
  var w = document.getElementById('tvChartWrap'), t = document.getElementById('tvToggle');
  w.classList.toggle('collapsed'); t.classList.toggle('collapsed');
  if (!w.classList.contains('collapsed') && !_kc) initKC();
  else if (w.classList.contains('collapsed') && _kcInt) { clearInterval(_kcInt); _kcInt = null; }
}

function initKC() {
  if (typeof klinecharts === 'undefined') { setTimeout(initKC, 500); return; }
  var c = document.getElementById('tvChart'); c.innerHTML = '';
  _kc = klinecharts.init('tvChart');
  _kc.setStyles({
    grid: { horizontalLine: { color: '#21262d' }, verticalLine: { color: '#21262d' } },
    xAxis: { axisLine: { color: '#30363d' } },
    yAxis: { axisLine: { color: '#30363d' } },
    candle: {
      bar: { upColor: '#3fb950', downColor: '#f85149' },
      priceMark: { show: true },
      tooltip: { show: true, labels: ['开','收','高','低','量'] },
    },
    separator: { color: '#30363d' },
  });
  _kc.createIndicator('VOL', true, { bar: { upColor: '#3fb95066', downColor: '#f8514966' } });
  // Timeframe bar
  var bar = document.createElement('div');
  bar.style.cssText = 'display:flex;gap:2px;padding:6px 12px;background:#0d1117;border-top:1px solid #21262d;flex-wrap:wrap;';
  window._ctf = '1h';
  [['1分','1m'],['5分','5m'],['15分','15m'],['1h','1h'],['4h','4h'],['日线','1d'],['周线','1w'],['月线','1M']].forEach(function(t){
    var b = document.createElement('span');
    b.textContent = t[0]; b.dataset.v = t[1];
    b.style.cssText = 'font-size:11px;padding:2px 8px;border-radius:4px;cursor:pointer;color:#8b949e;background:#161b22;border:1px solid #30363d;user-select:none;';
    if(t[1]==='1h') b.style.cssText += ';color:#58a6ff;border-color:#58a6ff;background:#1f6feb22';
    b.onclick = function(){
      window._ctf = t[1];
      bar.querySelectorAll('span').forEach(function(x){ x.style.color='#8b949e'; x.style.borderColor='#30363d'; x.style.background='#161b22'; });
      this.style.color='#58a6ff'; this.style.borderColor='#58a6ff'; this.style.background='#1f6feb22';
      loadKC();
    };
    bar.appendChild(b);
  });
  c.parentNode.insertBefore(bar, c.nextSibling);
  loadKC();
  _kcInt = setInterval(loadKC, 1000);
}

async function loadKC() {
  if (!_kc) return;
  var s = document.getElementById('symbolInput').value.trim() || 'BTC/USDT';
  document.getElementById('tvSymbol').textContent = s;
  var tf = window._ctf || '1h';
  var n = (tf==='1w'||tf==='1M') ? 100 : 200;
  try {
    var r = await fetch('https://api.binance.com/api/v3/klines?symbol=' + s.replace('/','').toUpperCase() + '&interval=' + tf + '&limit=' + n);
    if (!r.ok) throw Error(''+r.status);
    var d = await r.json();
    _kc.applyNewData(d.map(function(x){
      return {timestamp: parseInt(x[0]), open: parseFloat(x[1]), high: parseFloat(x[2]), low: parseFloat(x[3]), close: parseFloat(x[4]), volume: parseFloat(x[5])};
    }));
    _kc.resize();
  } catch(e) { console.warn('KC fetch:', e.message); }
}
document.getElementById('symbolInput').addEventListener('change', function(){ loadKC(); });
</script>
</body>
</html>"""


if __name__ == "__main__":
    import uvicorn
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print(f"  QMind Dashboard → http://127.0.0.1:{port}")
    uvicorn.run("qmind.server:app", host="127.0.0.1", port=port, reload=False)
