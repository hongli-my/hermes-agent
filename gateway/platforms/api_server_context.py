"""
api_server_context.py — Context Info HTTP API handler.

Standalone mixin that adds /api/context endpoint to APIServerAdapter.

Priority for token data:
  1. adapter._last_context_info — captured from agent after each chat
     completion (same source as CLI status bar: context_compressor)
  2. gateway_runner._running_agents / _agent_cache — live agents from
     other platforms (telegram, qqbot, etc.)
  3. SessionDB — fallback for historical/cumulative data only
"""

import time
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_AGENT_PENDING_SENTINEL = object()  # sentinel for pending agent slots


def _format_duration(started_at: Optional[float], ended_at: Optional[float]) -> Optional[str]:
    """Format a time interval as a human-readable string."""
    if not started_at:
        return None
    end = ended_at or time.time()
    try:
        elapsed = int(end - float(started_at))
    except (TypeError, ValueError):
        return None
    if elapsed < 60:
        return f"{elapsed}s"
    minutes, seconds = divmod(elapsed, 60)
    if minutes < 60:
        return f"{minutes}m {seconds}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes}m"


def _find_live_agent(adapter: Any) -> Optional[Any]:
    """Find any live or cached agent from the gateway runner.

    Used as a secondary source after _last_context_info.
    """
    runner = getattr(adapter, "gateway_runner", None)
    if not runner:
        logger.debug("_find_agent: no gateway_runner on adapter")
        return None
    # Running agents first (mid-turn)
    running = getattr(runner, "_running_agents", {})
    for _key, agent in list(running.items()):
        if agent is not _AGENT_PENDING_SENTINEL:
            return agent

    # Cached agents (between turns)
    cache_lock = getattr(runner, "_agent_cache_lock", None)
    cache = getattr(runner, "_agent_cache", None)
    if cache_lock and cache is not None:
        with cache_lock:
            for _key, cached_tuple in list(cache.items()):
                if cached_tuple:
                    return cached_tuple[0]

    return None


def register_context_routes(adapter: Any) -> None:
    """Register /api/context route on the adapter's aiohttp app."""
    from gateway.platforms.api_server_context import _handle_get_context
    # Bind handler to adapter instance as a method
    import types
    adapter._handle_get_context = types.MethodType(_handle_get_context, adapter)
    adapter._app.router.add_get("/api/context", adapter._handle_get_context)


async def _handle_get_context(self: Any, request: Any) -> Any:
    """GET /api/context — context usage info for the status bar.

    Returns used_tokens / max_tokens from the same source as the
    CLI status bar (context_compressor), plus cumulative totals.
    """
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    try:
        # Get latest session from DB for duration / message_count
        # Optional: query specific session by ?session_id=xxx
        requested_sid = request.query.get("session_id", "").strip()

        sessions = db.list_sessions_rich(limit=1, order_by_last_active=True)
        session = sessions[0] if sessions else None

        # If a specific session was requested, try to find it
        if requested_sid and (not session or session["id"] != requested_sid):
            try:
                # list_sessions_rich doesn't have a filter-by-id, so scan recent sessions
                all_sessions = db.list_sessions_rich(limit=50, order_by_last_active=True)
                for s in all_sessions:
                    if s["id"] == requested_sid:
                        session = s
                        break
            except Exception:
                pass

        if not session:
            return web.json_response({
                "ok": True,
                "context": {
                    "model": None,
                    "used_tokens": 0,
                    "max_tokens": 0,
                    "percent": 0,
                    "duration": None,
                    "session_id": None,
                    "active": False,
                    "message_count": 0,
                    "cumulative_input": 0,
                    "cumulative_output": 0,
                },
            })

        # Duration
        started_at = session.get("started_at")
        ended_at = session.get("ended_at")
        duration = _format_duration(
            float(started_at) if started_at else None,
            float(ended_at) if ended_at else None,
        )

        # Active check
        active = False
        if ended_at:
            try:
                active = (time.time() - float(ended_at)) < 60
            except (TypeError, ValueError):
                pass
        elif started_at:
            active = True

        msg_count = session.get("message_count") or 0
        session_id = session["id"]

        # ── Priority 1: _last_context_info from recent chat completion ────
        ctx = getattr(self, "_last_context_info", None)
        if ctx and ctx.get("used_tokens", 0) > 0:
            used_tokens = ctx["used_tokens"]
            max_tokens = ctx["max_tokens"]
            percent = round(used_tokens / max_tokens * 100) if max_tokens > 0 else 0
            return web.json_response({
                "ok": True,
                "context": {
                    "model": ctx.get("model") or session.get("model") or "",
                    "used_tokens": used_tokens,
                    "max_tokens": max_tokens,
                    "percent": percent,
                    "duration": duration,
                    "session_id": ctx.get("session_id", session_id),
                    "active": active,
                    "message_count": msg_count,
                    "cumulative_input": ctx.get("cumulative_input", 0),
                    "cumulative_output": ctx.get("cumulative_output", 0),
                },
            })

        # ── Priority 2: live agent from gateway runner ────────────────────
        agent = _find_live_agent(self)
        if agent:
            comp = getattr(agent, "context_compressor", None)
            if comp:
                used_tokens = getattr(comp, "last_prompt_tokens", 0) or 0
                max_tokens = getattr(comp, "context_length", 0) or 0
                percent = round(used_tokens / max_tokens * 100) if max_tokens > 0 else 0
                return web.json_response({
                    "ok": True,
                    "context": {
                        "model": getattr(agent, "model", "") or session.get("model") or "",
                        "used_tokens": used_tokens,
                        "max_tokens": max_tokens,
                        "percent": percent,
                        "duration": duration,
                        "session_id": getattr(agent, "session_id", session_id),
                        "active": True,
                        "message_count": msg_count,
                        "cumulative_input": getattr(agent, "session_prompt_tokens", 0) or 0,
                        "cumulative_output": getattr(agent, "session_completion_tokens", 0) or 0,
                    },
                })

        # ── Priority 3: DB fallback (no context window info, only cumulative) ──
        cumulative_input = session.get("input_tokens") or 0
        cumulative_output = session.get("output_tokens") or 0
        model = session.get("model") or ""

        return web.json_response({
            "ok": True,
            "context": {
                "model": model,
                "used_tokens": 0,
                "max_tokens": 0,
                "percent": 0,
                "duration": duration,
                "session_id": session_id,
                "active": active,
                "message_count": msg_count,
                "cumulative_input": cumulative_input,
                "cumulative_output": cumulative_output,
            },
        })

    except Exception as e:
        logger.error("Error in /api/context: %s", e, exc_info=True)
        return web.json_response({"ok": False, "error": str(e)}, status=500)
