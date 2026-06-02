"""
api_server_skills.py — Skills HTTP API handler.

Standalone mixin that adds /api/skills/* endpoints to APIServerAdapter.

Usage in api_server.py connect():
    from gateway.platforms.api_server_skills import register_skills_routes
    register_skills_routes(self)

Skills directories:
    ~/.hermes/skills/          — built-in skills
    ~/.hermes/custom-skills/   — custom user skills
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────

_SKILL_SOURCES = {
    "builtin": "skills",
    "custom": "custom-skills",
}

# ── Helpers ────────────────────────────────────────────────────────────


def _hermes_home() -> Path:
    """Get HERMES_HOME path."""
    home = os.environ.get("HOME", os.path.expanduser("~"))
    return Path(os.environ.get("HERMES_HOME", f"{home}/.hermes"))


def _skills_dir(source: str) -> Path:
    """Get skills directory for a source."""
    subdir = _SKILL_SOURCES.get(source)
    if not subdir:
        return Path()
    return _hermes_home() / subdir


def _parse_yaml_frontmatter(content: str) -> dict:
    """Parse simple YAML frontmatter from SKILL.md content."""
    meta = {}
    if not content.startswith("---"):
        return meta
    parts = content.split("---", 2)
    if len(parts) < 3:
        return meta
    fm = parts[1].strip()
    for line in fm.split("\n"):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        # Simple key: value parsing
        m = re.match(r'^(\w[\w_-]*):\s*(.*)', line)
        if m:
            key, val = m.group(1), m.group(2).strip()
            # Handle quoted strings
            if val.startswith('"') and val.endswith('"'):
                val = val[1:-1]
            elif val.startswith("'") and val.endswith("'"):
                val = val[1:-1]
            # Handle lists [a, b, c]
            elif val.startswith("["):
                try:
                    val = json.loads(val)
                except Exception:
                    pass
            meta[key] = val
    return meta


def _scan_skill_dir(skill_path: Path) -> Optional[Dict[str, Any]]:
    """Scan a single skill directory and return metadata."""
    if not skill_path.is_dir():
        return None

    dir_name = skill_path.name
    meta = {}
    description = ""

    # Try SKILL.md first, then DESCRIPTION.md
    for md_name in ("SKILL.md", "DESCRIPTION.md"):
        md_path = skill_path / md_name
        if not md_path.exists():
            continue
        try:
            content = md_path.read_text(encoding="utf-8")
            meta = _parse_yaml_frontmatter(content)
            # If frontmatter has description, use it
            if meta.get("description") and meta["description"] not in ("", "---"):
                description = meta["description"]
                break
            # Extract from body text
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
            for line in body.split("\n"):
                line = line.strip()
                if not line or line.startswith("```"):
                    continue
                # Heading line → use heading text
                heading = re.match(r'^#+\s*(.+)', line)
                if heading:
                    description = heading.group(1).strip()
                    break
                # First non-empty line
                description = line
                break
            break  # Found a file, stop looking
        except Exception:
            pass

    # Scan .py files for tools list
    tools = []
    try:
        for f in sorted(skill_path.iterdir()):
            if f.is_file() and f.suffix == ".py" and f.name != "__init__.py":
                tools.append(f.stem)
    except Exception:
        pass

    return {
        "name": meta.get("name", dir_name),
        "dir_name": dir_name,
        "path": str(skill_path),
        "description": meta.get("description", description),
        "category": meta.get("category", ""),
        "tools": tools,
    }


def _is_category_dir(cat_dir: Path) -> bool:
    """Check if a directory is a category containing sub-skill directories."""
    if not cat_dir.is_dir():
        return False
    for d in cat_dir.iterdir():
        if not d.is_dir():
            continue
        if d.name.startswith(".") or d.name.startswith("_"):
            continue
        # Has SKILL.md or DESCRIPTION.md or *.py — it's a sub-skill
        if (d / "SKILL.md").exists() or (d / "DESCRIPTION.md").exists():
            return True
        # Has any .py files (tool skill without SKILL.md)
        try:
            if any(f.suffix == ".py" for f in d.iterdir() if f.is_file()):
                return True
        except Exception:
            pass
    return False


def _scan_skills(source: str, category: Optional[str] = None) -> List[Dict[str, Any]]:
    """Scan skills directory and return list of skill metadata."""
    base_dir = _skills_dir(source)
    if not base_dir.is_dir():
        return []

    results = []
    category_dirs = set()

    # First pass: identify category directories
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue
        if _is_category_dir(entry):
            category_dirs.add(entry.name)

    # Second pass: scan skills
    for entry in sorted(base_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name.startswith(".") or entry.name.startswith("_"):
            continue

        if entry.name in category_dirs:
            # Category directory — scan sub-skills
            for skill_dir in sorted(entry.iterdir()):
                if not skill_dir.is_dir():
                    continue
                if skill_dir.name.startswith(".") or skill_dir.name.startswith("_"):
                    continue
                skill = _scan_skill_dir(skill_dir)
                if skill:
                    skill["source"] = source
                    skill["category"] = skill.get("category") or entry.name
                    if category is None or skill.get("category") == category:
                        results.append(skill)
        else:
            # Direct skill directory
            skill = _scan_skill_dir(entry)
            if skill:
                skill["source"] = source
                if category is None or skill.get("category") == category:
                    results.append(skill)

    return results


def _read_skill_detail(source: str, dir_name: str) -> Optional[Dict[str, Any]]:
    """Read full skill detail including docs and scripts."""
    base_dir = _skills_dir(source)
    if not base_dir.is_dir():
        return None

    # Find skill directory (direct or in category subdirectory)
    skill_path = base_dir / dir_name
    if not skill_path.is_dir():
        # Search in category subdirectories
        for cat_dir in base_dir.iterdir():
            if not cat_dir.is_dir():
                continue
            candidate = cat_dir / dir_name
            if candidate.is_dir():
                skill_path = candidate
                break

    if not skill_path.is_dir():
        return None

    meta = _scan_skill_dir(skill_path)
    if not meta:
        return None

    meta["source"] = source

    # Read docs (*.md files)
    docs = {}
    for f in sorted(skill_path.iterdir()):
        if f.is_file() and f.suffix == ".md" and f.name != "SKILL.md":
            try:
                docs[f.name] = f.read_text(encoding="utf-8")
            except Exception:
                pass
    # Also include SKILL.md body (after frontmatter)
    skill_md = skill_path / "SKILL.md"
    if skill_md.exists():
        try:
            content = skill_md.read_text(encoding="utf-8")
            body = content
            if content.startswith("---"):
                parts = content.split("---", 2)
                if len(parts) >= 3:
                    body = parts[2].strip()
            if body:
                docs["SKILL.md"] = body
        except Exception:
            pass

    # Read scripts (*.py files)
    py_files = {}
    for f in sorted(skill_path.iterdir()):
        if f.is_file() and f.suffix == ".py":
            try:
                py_files[f.name] = f.read_text(encoding="utf-8")
            except Exception:
                pass

    meta["docs"] = docs
    meta["py_files"] = py_files
    return meta


# ── Route registration ────────────────────────────────────────────────

def register_skills_routes(adapter: Any) -> None:
    """Register skills routes on the adapter's aiohttp app."""
    adapter._handle_list_skills = _handle_list_skills.__get__(adapter, type(adapter))
    adapter._handle_builtin_skills = _handle_builtin_skills.__get__(adapter, type(adapter))
    adapter._handle_custom_skills = _handle_custom_skills.__get__(adapter, type(adapter))
    adapter._handle_skill_detail = _handle_skill_detail.__get__(adapter, type(adapter))
    adapter._handle_skills_usage = _handle_skills_usage.__get__(adapter, type(adapter))

    app = adapter._app
    app.router.add_get("/api/skills", adapter._handle_list_skills)
    app.router.add_get("/api/skills/builtin", adapter._handle_builtin_skills)
    app.router.add_get("/api/skills/custom", adapter._handle_custom_skills)
    app.router.add_get("/api/skills/usage", adapter._handle_skills_usage)
    # detail must be last to avoid {source} capturing "builtin"/"custom"/"usage"
    app.router.add_get("/api/skills/{source}/{dir_name:.+}", adapter._handle_skill_detail)
    logger.debug("Skills API routes registered")


# ── Handler implementations ──────────────────────────────────────────

async def _handle_list_skills(self, request: Any) -> Any:
    """GET /api/skills — list all skills (builtin + custom)."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    builtin = _scan_skills("builtin")
    custom = _scan_skills("custom")

    return web.json_response({
        "ok": True,
        "builtin": builtin,
        "custom": custom,
    })


async def _handle_builtin_skills(self, request: Any) -> Any:
    """GET /api/skills/builtin — list builtin skills."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    category = request.query.get("category")
    skills = _scan_skills("builtin", category=category)

    return web.json_response({
        "ok": True,
        "skills": skills,
        "source": "builtin",
    })


async def _handle_custom_skills(self, request: Any) -> Any:
    """GET /api/skills/custom — list custom skills."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    category = request.query.get("category")
    skills = _scan_skills("custom", category=category)

    return web.json_response({
        "ok": True,
        "skills": skills,
        "source": "custom",
    })


async def _handle_skill_detail(self, request: Any) -> Any:
    """GET /api/skills/{source}/{dir_name} — skill detail with docs and scripts."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    source = request.match_info.get("source", "")
    dir_name = request.match_info.get("dir_name", "")

    if source not in ("builtin", "custom"):
        return web.json_response({"ok": False, "error": "source must be 'builtin' or 'custom'"}, status=400)

    # Sanitize dir_name — prevent path traversal
    if "/" in dir_name or ".." in dir_name:
        return web.json_response({"ok": False, "error": "invalid dir_name"}, status=400)

    detail = _read_skill_detail(source, dir_name)
    if not detail:
        return web.json_response({"ok": False, "error": "skill not found"}, status=404)

    return web.json_response({"ok": True, **detail})


async def _handle_skills_usage(self, request: Any) -> Any:
    """GET /api/skills/usage — skill usage statistics from messages table."""
    from aiohttp import web

    auth_err = self._check_auth(request)
    if auth_err:
        return auth_err

    db = self._ensure_session_db()
    if db is None:
        return web.json_response({"ok": False, "error": "SessionDB unavailable"}, status=503)

    try:
        # Query skill usage from messages with role='tool'
        now_ts = time.time()
        week_ago = now_ts - 7 * 86400

        # Get all tool messages from last 7 days
        recent_messages = db.search_messages("skill", limit=1000)
        # We need raw messages, not search results — use a direct query approach
        # For now, use the search_messages result as approximation

        usage_7d = {}
        usage_all = {}

        # Count skill names in tool message content
        messages = db.get_messages("__all__") if hasattr(db, 'get_all_tool_messages') else []
        # Fallback: aggregate from sessions
        sessions = db.list_sessions_rich(limit=100, order_by_last_active=True)
        for s in sessions:
            sid = s.get("id")
            if not sid:
                continue
            try:
                msgs = db.get_messages(sid)
                for m in msgs:
                    if m.get("role") != "tool":
                        continue
                    tool_calls = m.get("tool_calls")
                    if isinstance(tool_calls, str):
                        try:
                            tool_calls = json.loads(tool_calls)
                        except Exception:
                            tool_calls = None
                    if not isinstance(tool_calls, list):
                        continue
                    for tc in tool_calls:
                        name = tc.get("function", {}).get("name", "")
                        if not name:
                            continue
                        usage_all[name] = (usage_all.get(name) or 0) + 1
                        ts = m.get("timestamp", 0)
                        try:
                            ts = float(ts)
                        except (TypeError, ValueError):
                            ts = 0
                        if ts >= week_ago:
                            usage_7d[name] = (usage_7d.get(name) or 0) + 1
            except Exception:
                continue

        return web.json_response({
            "ok": True,
            "usage_7d": usage_7d,
            "usage_all": usage_all,
        })

    except Exception as e:
        logger.debug("skills_usage error: %s", e)
        return web.json_response({"ok": False, "error": str(e)}, status=500)
