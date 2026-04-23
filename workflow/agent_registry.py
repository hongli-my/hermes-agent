"""
workflow/agent_registry.py — 活体 Agent 注册表

在 workflow 执行期间，每个 agent 节点会把正在运行的 AIAgent 注册到这里。
cli 的心跳线程每秒遍历注册表，通过读取 agent._last_activity_ts / _last_activity_desc
判断 agent 是否真的在干活（真实信号，非本地假动画）。

使用：
    from workflow.agent_registry import register_agent, unregister_agent, snapshot

    token = register_agent(label="analyze[3/67]", agent=agent)
    try:
        agent.run_conversation(...)
    finally:
        unregister_agent(token)
"""

from __future__ import annotations

import threading
import time
import uuid
from typing import Any, Dict, List, Optional

# token -> 记录
_REGISTRY: Dict[str, Dict[str, Any]] = {}
_LOCK = threading.Lock()


def register_agent(label: str, agent: Any) -> str:
    """把 agent 登记为"正在运行"。返回 token，用于稍后反注册。"""
    token = uuid.uuid4().hex
    with _LOCK:
        _REGISTRY[token] = {
            "label": label,
            "agent": agent,
            "started_at": time.time(),
        }
    return token


def unregister_agent(token: Optional[str]) -> None:
    """反注册；token 可为 None（注册失败时的兜底）。"""
    if not token:
        return
    with _LOCK:
        _REGISTRY.pop(token, None)


def snapshot() -> List[Dict[str, Any]]:
    """返回当前所有活体 agent 的浅拷贝快照（供心跳线程读取）。

    每个条目：
        {
          "label": "analyze[3/67]",
          "started_at": <epoch>,
          "last_activity_ts": <epoch>,   # 来自 agent._last_activity_ts
          "last_activity_desc": "receiving stream response",
          "idle_seconds": 12.3,          # now - last_activity_ts
          "running_seconds": 45.1,       # now - started_at
        }
    读取失败的字段会被置为 None/0，不抛异常。
    """
    now = time.time()
    out: List[Dict[str, Any]] = []
    with _LOCK:
        items = list(_REGISTRY.values())

    for rec in items:
        agent = rec.get("agent")
        started_at = rec.get("started_at") or now

        last_ts = getattr(agent, "_last_activity_ts", None)
        last_desc = getattr(agent, "_last_activity_desc", None)

        if isinstance(last_ts, (int, float)) and last_ts > 0:
            idle = max(0.0, now - last_ts)
        else:
            idle = None

        out.append({
            "label": rec.get("label") or "?",
            "started_at": started_at,
            "last_activity_ts": last_ts,
            "last_activity_desc": last_desc or "(no activity yet)",
            "idle_seconds": idle,
            "running_seconds": max(0.0, now - started_at),
        })

    # 按运行最久的排前面，方便心跳行展示
    out.sort(key=lambda r: r["running_seconds"], reverse=True)
    return out


def count() -> int:
    """当前正在运行的 agent 数量。"""
    with _LOCK:
        return len(_REGISTRY)
