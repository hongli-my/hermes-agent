"""
api_server_memory.py — Memory HTTP API handler.

Standalone mixin that adds /api/memory/* endpoints to APIServerAdapter.

Usage in api_server.py connect():
    from gateway.platforms.api_server_memory import register_memory_routes
    register_memory_routes(self)

Memory files:
    ~/.hermes/memories/MEMORY.md  — agent's personal notes
    ~/.hermes/memories/USER.md    — user profile

Entry format: lines separated by § (section sign), each entry is a
declarative fact like "User prefers Chinese."
"""

from __future__ import annotations

import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

_MEMORY_LIMITS = {
    "memory": 2200,
    "user": 1375,
}

_ENTRY_SEP = "§"

# ── Invisible unicode detection ────────────────────────────────────────

_INVISIBLE_RE = re.compile(
    r"[\u200b-\u200f\u2028-\u202f\u2060-\u206f\ufeff\u00ad]"
)

# ── Helpers ────────────────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Get HERMES_HOME path."""
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return Path(os.environ.get("HERMES_HOME", f"{home}/.hermes"))


def _memory_path(target: str) -> Path:
    """Get file path for a memory target."""
    home = _hermes_home()
    if target == "user":
        return home / "memories" / "USER.md"
    return home / "memories" / "MEMORY.md"


def _parse_entries(content: str) -> List[str]:
    """Parse memory content into entries (separated by §)."""
    if not content or not content.strip():
        return []
    entries = content.split(_ENTRY_SEP)
    return [e.strip() for e in entries if e.strip()]


def _serialize_entries(entries: List[str]) -> str:
    """Serialize entries back to memory file format."""
    if not entries:
        return ""
    return f"\n{_ENTRY_SEP} ".join(entries) + "\n"


def _read_memory(target: str) -> Tuple[List[str], int, int]:
    """Read memory file and return (entries, char_count, char_limit)."""
    path = _memory_path(target)
    limit = _MEMORY_LIMITS.get(target, 2200)
    try:
        content = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return [], 0, limit
    except Exception:
        return [], 0, limit
    entries = _parse_entries(content)
    char_count = sum(len(e) for e in entries)
    return entries, char_count, limit


def _write_memory(target: str, entries: List[str]) -> Tuple[bool, Optional[str]]:
    """Write entries to memory file atomically."""
    path = _memory_path(target)
    content = _serialize_entries(entries)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_text(content, encoding="utf-8")
        tmp_path.rename(path)
        return True, None
    except Exception as e:
        return False, str(e)


def _check_invisible(text: str) -> Optional[str]:
    """Check for invisible unicode characters. Returns warning or None."""
    matches = _INVISIBLE_RE.findall(text)
    if matches:
        codes = ", ".join(f"U+{ord(c):04X}" for c in matches[:5])
        return f"Content contains invisible unicode: {codes}"
    return None


def _format_usage(char_count: int, char_limit: int) -> str:
    """Format usage string like '45% — 990/2200 chars'."""
    pct = round(char_count / char_limit * 100) if char_limit > 0 else 0
    return f"{pct}% — {char_count}/{char_limit} chars"


def _make_response(target: str, entries: List[str], message: str = "") -> Dict[str, Any]:
    """Build standard memory API response."""
    char_count = sum(len(e) for e in entries)
    char_limit = _MEMORY_LIMITS.get(target, 2200)
    resp = {
        "ok": True,
        "target": target,
        "entries": entries,
        "usage": _format_usage(char_count, char_limit),
        "entry_count": len(entries),
        "char_count": char_count,
        "char_limit": char_limit,
    }
    if message:
        resp["message"] = message
    return resp


# ── Route registration ────────────────────────────────────────────────

def register_memory_routes(adapter: Any) -> None:
    """Register memory routes on the adapter's aiohttp app."""
    adapter._handle_list_memory = _handle_list_memory.__get__(adapter, type(adapter))
    adapter._handle_add_memory = _handle_add_memory.__get__(adapter, type(adapter))
    adapter._handle_replace_memory = _handle_replace_memory.__get__(adapter, type(adapter))
    adapter._handle_remove_memory = _handle_remove_memory.__get__(adapter, type(adapter))

    app = adapter._app
    app.router.add_get("/api/memory", adapter._handle_list_memory)
    app.router.add_post("/api/memory/add", adapter._handle_add_memory)
    app.router.add_post("/api/memory/replace", adapter._handle_replace_memory)
    app.router.add_post("/api/memory/remove", adapter._handle_remove_memory)
    logger.debug("Memory API routes registered")


# ── Handler implementations ──────────────────────────────────────────

async def _handle_list_memory(self, request: Any) -> Any:
    """GET /api/memory — list all memory entries."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    mem_entries, mem_chars, mem_limit = _read_memory("memory")
    user_entries, user_chars, user_limit = _read_memory("user")

    return web.json_response({
        "ok": True,
        "memory": {
            "target": "memory",
            "entries": mem_entries,
            "usage": _format_usage(mem_chars, mem_limit),
            "entry_count": len(mem_entries),
            "char_count": mem_chars,
            "char_limit": mem_limit,
        },
        "user": {
            "target": "user",
            "entries": user_entries,
            "usage": _format_usage(user_chars, user_limit),
            "entry_count": len(user_entries),
            "char_count": user_chars,
            "char_limit": user_limit,
        },
    })


async def _handle_add_memory(self, request: Any) -> Any:
    """POST /api/memory/add — add a memory entry."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    try:
        import json as _json
        body = _json.loads(await request.text())
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)

    target = body.get("target", "memory")
    if target not in ("memory", "user"):
        return web.json_response({"ok": False, "error": "target must be 'memory' or 'user'"}, status=400)

    content = body.get("content", "").strip()
    if not content:
        return web.json_response({"ok": False, "error": "content required"}, status=400)

    # Check invisible unicode
    warning = _check_invisible(content)

    entries, char_count, char_limit = _read_memory(target)

    # Check for duplicate
    for e in entries:
        if e == content:
            return web.json_response({"ok": False, "error": "duplicate entry"}, status=409)

    # Check size limit
    if char_count + len(content) > char_limit:
        return web.json_response({"ok": False, "error": "exceeds char limit"}, status=400)

    entries.append(content)
    ok, err = _write_memory(target, entries)
    if not ok:
        return web.json_response({"ok": False, "error": f"write failed: {err}"}, status=500)

    resp = _make_response(target, entries, "Entry added.")
    if warning:
        resp["warning"] = warning
    return web.json_response(resp)


async def _handle_replace_memory(self, request: Any) -> Any:
    """POST /api/memory/replace — replace a memory entry."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    try:
        import json as _json
        body = _json.loads(await request.text())
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)

    target = body.get("target", "memory")
    if target not in ("memory", "user"):
        return web.json_response({"ok": False, "error": "target must be 'memory' or 'user'"}, status=400)

    old_text = body.get("old_text", "").strip()
    new_content = body.get("content", "").strip()
    if not old_text or not new_content:
        return web.json_response({"ok": False, "error": "old_text and content required"}, status=400)

    warning = _check_invisible(new_content)

    entries, char_count, char_limit = _read_memory(target)

    # Find exact match
    match_idx = None
    for i, e in enumerate(entries):
        if old_text in e:
            match_idx = i
            break

    if match_idx is None:
        return web.json_response({"ok": False, "error": "old_text not found"}, status=404)

    # Check for ambiguous match
    for i, e in enumerate(entries):
        if old_text in e and i != match_idx:
            return web.json_response({"ok": False, "error": "old_text matches multiple entries"}, status=400)

    # Check size limit after replacement
    new_char_count = char_count - len(entries[match_idx]) + len(new_content)
    if new_char_count > char_limit:
        return web.json_response({"ok": False, "error": "exceeds char limit"}, status=400)

    entries[match_idx] = new_content
    ok, err = _write_memory(target, entries)
    if not ok:
        return web.json_response({"ok": False, "error": f"write failed: {err}"}, status=500)

    resp = _make_response(target, entries, "Entry replaced.")
    if warning:
        resp["warning"] = warning
    return web.json_response(resp)


async def _handle_remove_memory(self, request: Any) -> Any:
    """POST /api/memory/remove — remove a memory entry."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    try:
        import json as _json
        body = _json.loads(await request.text())
    except Exception:
        return web.json_response({"ok": False, "error": "invalid JSON body"}, status=400)

    target = body.get("target", "memory")
    if target not in ("memory", "user"):
        return web.json_response({"ok": False, "error": "target must be 'memory' or 'user'"}, status=400)

    old_text = body.get("old_text", "").strip()
    if not old_text:
        return web.json_response({"ok": False, "error": "old_text required"}, status=400)

    entries, _, _ = _read_memory(target)

    # Find exact match
    match_idx = None
    for i, e in enumerate(entries):
        if old_text in e:
            match_idx = i
            break

    if match_idx is None:
        return web.json_response({"ok": False, "error": "old_text not found"}, status=404)

    # Check for ambiguous match
    for i, e in enumerate(entries):
        if old_text in e and i != match_idx:
            return web.json_response({"ok": False, "error": "old_text matches multiple entries"}, status=400)

    entries.pop(match_idx)
    ok, err = _write_memory(target, entries)
    if not ok:
        return web.json_response({"ok": False, "error": f"write failed: {err}"}, status=500)

    return web.json_response(_make_response(target, entries, "Entry removed."))
