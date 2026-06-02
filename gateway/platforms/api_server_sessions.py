"""
api_server_sessions.py — Session / Message / Search HTTP API handlers.

Designed as a **standalone mixin** that adds session-management endpoints
to APIServerAdapter without modifying the core api_server.py logic.
This keeps the diff minimal for easy upstream merges.

Usage in api_server.py connect():
    from gateway.platforms.api_server_sessions import register_session_routes
    register_session_routes(self)

Endpoints added:
    GET    /api/sessions                       — list sessions
    GET    /api/sessions/{session_id}           — session detail + lineage
    GET    /api/sessions/{session_id}/messages  — session messages
    PATCH  /api/sessions/{session_id}/rename    — rename session
    DELETE /api/sessions/{session_id}           — delete session
    GET    /api/sessions/search                 — full-text search
"""

from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ── Helpers ────────────────────────────────────────────────────────────

def _format_ts(ts: Optional[float]) -> Optional[str]:
    """Format a unix timestamp to local time string, or None."""
    if ts is None:
        return None
    try:
        ts = float(ts)
    except (TypeError, ValueError):
        return None
    if ts <= 0:
        return None
    from datetime import datetime
    return datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M:%S")


def _format_duration(start: Optional[float], end: Optional[float]) -> Optional[str]:
    """Format a duration in seconds to human-readable string."""
    if start is None:
        return None
    secs = int((end or time.time()) - start)
    if secs < 60:
        return f"{secs}s"
    mins, secs = divmod(secs, 60)
    if mins < 60:
        return f"{mins}m {secs}s"
    hrs, mins = divmod(mins, 60)
    return f"{hrs}h {mins}m {secs}s"


def _format_tokens(n: Optional[int]) -> str:
    """Format token count with K/M suffix."""
    if n is None:
        return "0"
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}K"
    return str(n)


def _extract_tool_names(tool_calls: Any) -> List[str]:
    """Extract unique tool names from tool_calls JSON."""
    if not tool_calls:
        return []
    if isinstance(tool_calls, str):
        try:
            import json
            tool_calls = json.loads(tool_calls)
        except Exception:
            return []
    if not isinstance(tool_calls, list):
        return []
    names = []
    seen = set()
    for tc in tool_calls:
        name = tc.get("function", {}).get("name") or tc.get("name")
        if name and name not in seen:
            seen.add(name)
            names.append(name)
    return names


def _content_preview(content: Optional[str], max_len: int = 200) -> str:
    """Truncate content for preview."""
    if not content:
        return ""
    text = content.replace("\n", " ").replace("\r", "")
    if len(text) > max_len:
        return text[:max_len] + "..."
    return text


# ── Route registration ────────────────────────────────────────────────

def register_session_routes(adapter: Any) -> None:
    """Register session/message/search routes on the adapter's aiohttp app.

    Call this from APIServerAdapter.connect() after self._app is created.
    Handler methods are bound to the adapter instance at runtime so that
    no modifications to the APIServerAdapter class definition are needed.
    """
    # Bind handler methods to the adapter instance
    adapter._handle_list_sessions = _handle_list_sessions.__get__(adapter, type(adapter))
    adapter._handle_get_session = _handle_get_session.__get__(adapter, type(adapter))
    adapter._handle_get_session_messages = _handle_get_session_messages.__get__(adapter, type(adapter))
    adapter._handle_rename_session = _handle_rename_session.__get__(adapter, type(adapter))
    adapter._handle_delete_session = _handle_delete_session.__get__(adapter, type(adapter))
    adapter._handle_search_sessions = _handle_search_sessions.__get__(adapter, type(adapter))
    adapter._handle_delete_message = _handle_delete_message.__get__(adapter, type(adapter))
    adapter._handle_fork_session = _handle_fork_session.__get__(adapter, type(adapter))

    app = adapter._app
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    app.router.add_get("/api/sessions/search", adapter._handle_search_sessions)
    app.router.add_get("/api/sessions/{session_id}", adapter._handle_get_session)
    app.router.add_get("/api/sessions/{session_id}/messages", adapter._handle_get_session_messages)
    app.router.add_patch("/api/sessions/{session_id}/rename", adapter._handle_rename_session)
    app.router.add_post("/api/sessions/{session_id}/fork", adapter._handle_fork_session)
    app.router.add_delete("/api/sessions/{session_id}", adapter._handle_delete_session)
    app.router.add_delete("/api/sessions/{session_id}/messages/{message_id}", adapter._handle_delete_message)
    logger.debug("Session API routes registered")


# ── Handler implementations ──────────────────────────────────────────
# These are mixed into APIServerAdapter at runtime via register_session_routes.
# They use self._ensure_session_db() and self._check_auth() from the adapter.

async def _handle_list_sessions(self, request: Any) -> Any:
    """GET /api/sessions — list top-level sessions."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    limit = int(request.query.get("limit", "50"))
    offset = int(request.query.get("offset", "0"))
    source = request.query.get("source")

    try:
        sessions = db.list_sessions_rich(
            source=source,
            limit=limit,
            offset=offset,
            include_children=False,
            order_by_last_active=True,
        )
    except Exception as e:
        logger.debug("list_sessions error: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    # Enrich with formatted fields
    for s in sessions:
        s["started_at_fmt"] = _format_ts(s.get("started_at"))
        s["ended_at_fmt"] = _format_ts(s.get("ended_at"))
        s["duration"] = _format_duration(s.get("started_at"), s.get("ended_at"))
        s["input_tokens_fmt"] = _format_tokens(s.get("input_tokens"))
        s["output_tokens_fmt"] = _format_tokens(s.get("output_tokens"))
        if not s.get("title"):
            s["title"] = s.get("preview") or f"Session {s['id'][:19]}"

    return web.json_response({"ok": True, "data": sessions})


async def _handle_get_session(self, request: Any) -> Any:
    """GET /api/sessions/{session_id} — session detail with lineage."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    # Resolve short prefix to full ID
    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    try:
        session = db.get_session(session_id)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    if not session:
        return web.json_response({"ok": False, "error": "session not found"}, status=404)

    # Build lineage (parent chain, max 10 levels to prevent infinite loop)
    lineage = []
    current = session
    _depth = 0
    while current and _depth < 10:
        lineage.insert(0, {
            "id": current["id"],
            "title": current.get("title"),
            "started_at": current.get("started_at"),
            "depth": len(lineage),
        })
        parent_id = current.get("parent_session_id")
        if not parent_id:
            break
        current = db.get_session(parent_id)
        _depth += 1

    # Enrich
    session["started_at_fmt"] = _format_ts(session.get("started_at"))
    session["ended_at_fmt"] = _format_ts(session.get("ended_at"))
    session["duration"] = _format_duration(session.get("started_at"), session.get("ended_at"))
    session["input_tokens_fmt"] = _format_tokens(session.get("input_tokens"))
    session["output_tokens_fmt"] = _format_tokens(session.get("output_tokens"))
    if not session.get("title"):
        session["title"] = f"Session {session_id[:19]}"

    return web.json_response({"ok": True, "data": session, "lineage": lineage})


async def _handle_get_session_messages(self, request: Any) -> Any:
    """GET /api/sessions/{session_id}/messages — messages for one session only."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    try:
        messages = db.get_messages(session_id)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    if not messages:
        messages = []

    # Enrich each message
    for m in messages:
        m["tool_names"] = _extract_tool_names(m.get("tool_calls"))
        m["timestamp_fmt"] = _format_ts(m.get("timestamp"))
        m["content_preview"] = _content_preview(m.get("content"))

    return web.json_response({"ok": True, "data": messages, "session_id": session_id})


async def _handle_rename_session(self, request: Any) -> Any:
    """PATCH /api/sessions/{session_id}/rename — rename a session."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    try:
        import json as _json
        body = _json.loads(await request.text())
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)

    new_title = body.get("title")
    if not new_title:
        return web.json_response({"ok": False, "error": "title required"}, status=400)

    if len(new_title) > 100:
        new_title = new_title[:100]

    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    try:
        ok = db.set_session_title(session_id, new_title)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    if not ok:
        return web.json_response({"ok": False, "error": "session not found"}, status=404)

    return web.json_response({"ok": True, "session_id": session_id, "title": new_title})


async def _handle_delete_session(self, request: Any) -> Any:
    """DELETE /api/sessions/{session_id} — delete session.

    Query parameter ``cascade``: when set to ``true`` or ``1``, all child
    (and grandchild, etc.) sessions are also deleted recursively.  This is
    the default mode for the WebUI so that deleting a parent also removes
    its forked branches.  Without ``cascade``, child sessions are orphaned.
    """
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    cascade = request.query.get("cascade", "").lower() in ("true", "1")
    import sys
    print(f"[DEBUG-DELETE] session_id={session_id} cascade={cascade} query={dict(request.query)}", file=sys.stderr, flush=True)

    try:
        ok = db.delete_session(session_id, cascade=cascade)
    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    if not ok:
        return web.json_response({"ok": False, "error": "session not found"}, status=404)

    result: Dict[str, Any] = {"ok": True, "session_id": session_id}
    if cascade:
        result["cascade"] = True
    return web.json_response(result)


async def _handle_fork_session(self, request: Any) -> Any:
    """POST /api/sessions/{session_id}/fork — branch a session (copy history into new session).

    Reuses the same logic as the /branch CLI command: create_session + append_message
    + set_session_title, which is more robust than db.fork_session().
    """
    import uuid
    from aiohttp import web
    from datetime import datetime

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    # Verify the parent session exists
    parent = db.get_session(session_id)
    if not parent:
        return web.json_response({"ok": False, "error": "session not found"}, status=404)

    # Generate new session ID (same format as /branch command)
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    short_uuid = uuid.uuid4().hex[:6]
    new_id = f"{timestamp_str}_{short_uuid}"

    # Determine branch title (reuse get_next_title_in_lineage like /branch does)
    base_title = parent.get("title") or "branch"
    branch_title = db.get_next_title_in_lineage(base_title)

    try:
        # Create the new session with parent link
        db.create_session(
            session_id=new_id,
            source=parent.get("source", "api"),
            model=parent.get("model"),
            model_config=parent.get("model_config"),
            system_prompt=parent.get("system_prompt"),
            user_id=parent.get("user_id"),
            parent_session_id=session_id,
        )

        # Copy messages one by one (same as /branch command)
        messages = db.get_messages(session_id)
        for msg in messages:
            try:
                db.append_message(
                    session_id=new_id,
                    role=msg.get("role", "user"),
                    content=msg.get("content"),
                    tool_name=msg.get("tool_name") or msg.get("name"),
                    tool_calls=msg.get("tool_calls"),
                    tool_call_id=msg.get("tool_call_id"),
                    finish_reason=msg.get("finish_reason"),
                    reasoning=msg.get("reasoning"),
                    reasoning_content=msg.get("reasoning_content"),
                    reasoning_details=msg.get("reasoning_details"),
                    codex_reasoning_items=msg.get("codex_reasoning_items"),
                    codex_message_items=msg.get("codex_message_items"),
                )
            except Exception:
                pass  # Best-effort copy

        # Set title after messages are copied
        db.set_session_title(new_id, branch_title)

        # Mark parent as branched so list_sessions_rich surfaces the child
        db.end_session(session_id, "branched")

    except Exception as e:
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    return web.json_response({"ok": True, "session_id": new_id, "parent_session_id": session_id})


async def _handle_search_sessions(self, request: Any) -> Any:
    """GET /api/sessions/search?q=keyword — full-text search across messages."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    query = request.query.get("q", "").strip()
    if not query:
        return web.json_response({"ok": False, "error": "q parameter required"}, status=400)

    limit = int(request.query.get("limit", "20"))
    offset = int(request.query.get("offset", "0"))

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    try:
        results = db.search_messages(
            query=query,
            limit=limit,
            offset=offset,
        )
    except Exception as e:
        logger.debug("search_messages error: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    return web.json_response({"ok": True, "data": results, "query": query})


# ── Message round delete ────────────────────────────────────────────

_MSG_ID_RE = r"\d{1,10}"  # SQLite auto-increment row ID


async def _handle_delete_message(self, request: Any) -> Any:
    """DELETE /api/sessions/{session_id}/messages/{message_id}

    Delete an entire conversation round containing the specified message.
    A "round" is all messages from the nearest preceding user message
    (inclusive) up to but not including the next user message.

    Returns the list of deleted message IDs for UI feedback.
    """
    from aiohttp import web
    import re

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    session_id = request.match_info.get("session_id", "")
    message_id_str = request.match_info.get("message_id", "")

    if not session_id:
        return web.json_response({"ok": False, "error": "session_id required"}, status=400)
    if not re.match(_MSG_ID_RE, message_id_str):
        return web.json_response({"ok": False, "error": "invalid message_id"}, status=400)

    message_id = int(message_id_str)

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    resolved = db.resolve_session_id(session_id)
    if resolved:
        session_id = resolved

    try:
        deleted_ids = db.delete_message_round(session_id, message_id)
    except Exception as e:
        logger.exception("delete_message_round error")
        return web.json_response({"ok": False, "error": str(e)}, status=500)

    if not deleted_ids:
        return web.json_response({"ok": False, "error": "message not found"}, status=404)

    return web.json_response({
        "ok": True,
        "deleted_ids": deleted_ids,
        "count": len(deleted_ids),
    })
