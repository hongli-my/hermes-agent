#!/usr/bin/env python3.11
"""
Hermes Agent — Textual TUI
===========================
A rich terminal UI for Hermes Agent using Textual.

Design: OpenCode-style layout
  - Main area: document-style transcript (user prompts + assistant markdown,
    interleaved with reasoning blocks and tool-call rows)
  - Right sidebar: Context (tokens / % / cost) · Usage · MCP · LSP · footer
  - Rounded input box docked at the bottom
  - Status bar: left = mode · model, right = tokens · commands hint

Architecture:
  - Direct Python: constructs AIAgent, calls run_conversation() in a thread
  - Does NOT use tui_gateway (which writes JSON-RPC to stdout)
  - Streams the agent's work through AIAgent's callback hooks:
      stream_delta_callback   → assistant text segments (None = flush boundary)
      reasoning_callback      → model reasoning / thinking
      tool_start_callback     → a tool invocation began
      tool_complete_callback  → a tool invocation finished
    All callbacks fire on the agent worker thread and hop back to the UI
    thread via Textual's call_from_thread.

Usage: hermes --textual-tui
       or: python3.11 hermes_tui.py
"""

import os
import sys
import time
import json
import re
import difflib
import threading
import uuid
from pathlib import Path

# Ensure hermes modules are importable
_hermes_src = str(Path(__file__).resolve().parent.parent)
if _hermes_src not in sys.path:
    sys.path.insert(0, _hermes_src)


def _build_agent():
    """Construct an AIAgent without tui_gateway (avoids stdout JSON-RPC pollution)."""
    from run_agent import AIAgent
    from hermes_cli.env_loader import load_hermes_dotenv
    from hermes_constants import get_hermes_home
    from hermes_cli.runtime_provider import resolve_runtime_provider
    from hermes_cli.config import load_config

    hermes_home = get_hermes_home()
    load_hermes_dotenv(hermes_home=hermes_home, project_env=Path(_hermes_src) / ".env")

    cfg = load_config()
    agent_cfg = cfg.get("agent") or {}

    model = os.environ.get("HERMES_MODEL", "") or cfg.get("model", "")
    provider_name = os.environ.get("HERMES_PROVIDER", "") or cfg.get("provider", "")

    runtime = resolve_runtime_provider(
        requested=provider_name or None,
        target_model=model or None,
    )

    max_turns = 90
    try:
        mt = (os.environ.get("HERMES_MAX_ITERATIONS", "")
              or cfg.get("max_turns")
              or agent_cfg.get("max_iterations"))
        if mt:
            max_turns = int(mt)
    except (ValueError, TypeError):
        pass

    # Resolve enabled_toolsets: None = all tools (default), list = only those toolsets.
    # CRITICAL: [] means "no tools" (empty allowlist), None means "all tools".
    # Do NOT write `enabled_toolsets or []` — that turns None into [] which
    # disables ALL tools.  See _compute_tool_definitions() in model_tools.py.
    enabled_toolsets = None
    try:
        et = os.environ.get("HERMES_ENABLED_TOOLSETS", "")
        if et:
            enabled_toolsets = [t.strip() for t in et.split(",") if t.strip()]
    except Exception:
        pass
    # Also respect the TUI-specific env var (matching tui_gateway).
    try:
        et2 = os.environ.get("HERMES_TUI_TOOLSETS", "")
        if et2:
            enabled_toolsets = [t.strip() for t in et2.split(",") if t.strip()]
    except Exception:
        pass
    # If still None, try loading from config (matching _load_enabled_toolsets).
    if enabled_toolsets is None:
        try:
            from hermes_cli.tools_config import _get_platform_tools
            enabled_toolsets = list(_get_platform_tools(cfg, "cli", include_default_mcp_servers=True)) or None
        except Exception:
            pass

    # Resolve disabled_toolsets from config.
    disabled_toolsets = None
    try:
        dt = agent_cfg.get("disabled_toolsets") or cfg.get("disabled_toolsets")
        if dt:
            if isinstance(dt, str):
                disabled_toolsets = [t.strip() for t in dt.split(",") if t.strip()]
            elif isinstance(dt, list):
                disabled_toolsets = [str(t).strip() for t in dt if str(t).strip()]
    except Exception:
        pass

    # If --resume was passed, reuse the existing session ID; otherwise generate a new one.
    _resume_id = os.environ.get("HERMES_RESUME_SESSION", "").strip()
    _is_resume = False
    if _resume_id:
        session_key = _resume_id
        _is_resume = True
        # Clean up so a subsequent /clear doesn't accidentally resume again.
        os.environ.pop("HERMES_RESUME_SESSION", None)
    else:
        session_key = f"textual-tui-{uuid.uuid4().hex[:8]}"

    reasoning_config = None
    try:
        from hermes_constants import parse_reasoning_effort
        _effort = str((agent_cfg.get("reasoning_effort", "") or "").strip())
        reasoning_config = parse_reasoning_effort(_effort)
    except Exception:
        pass

    service_tier = None
    try:
        _st = str((agent_cfg.get("service_tier", "") or "").strip().lower())
        if _st in {"fast", "priority", "on"}:
            service_tier = "priority"
    except Exception:
        pass

    # ── SessionDB: enables session persistence and session_search tool ──
    session_db = None
    try:
        from hermes_state import SessionDB
        session_db = SessionDB()
    except Exception as e:
        print(f"[TUI] SessionDB not available — sessions will NOT be indexed: {e}", file=sys.stderr, flush=True)

    # ── Prefill messages: few-shot priming from config ──
    prefill_messages = None
    try:
        pf_path = agent_cfg.get("prefill_messages_file") or cfg.get("prefill_messages_file", "")
        if pf_path:
            path = Path(pf_path).expanduser()
            if not path.is_absolute():
                path = Path(hermes_home) / path
            if path.exists():
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    prefill_messages = data
    except Exception:
        pass

    # ── Request overrides: fast mode / priority processing ──
    request_overrides = None
    try:
        if service_tier:
            from hermes_cli.models import resolve_fast_mode_overrides
            request_overrides = resolve_fast_mode_overrides(
                runtime.get("model") or model
            )
    except Exception:
        pass

    # ── Ephemeral system prompt: env var takes precedence over config ──
    system_prompt = (
        os.environ.get("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        or (agent_cfg.get("system_prompt") or "").strip()
        or None
    )

    agent = AIAgent(
        model=runtime.get("model") or model,
        max_iterations=max_turns,
        provider=runtime.get("provider") or "",
        base_url=runtime.get("base_url") or "",
        api_key=runtime.get("api_key") or "",
        api_mode=runtime.get("api_mode") or "",
        acp_command=runtime.get("command") or "",
        acp_args=runtime.get("args") or [],
        credential_pool=runtime.get("credential_pool"),
        quiet_mode=True,
        verbose_logging=False,
        reasoning_config=reasoning_config,
        service_tier=service_tier,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        platform="tui",
        session_id=session_key,
        session_db=session_db,
        ephemeral_system_prompt=system_prompt,
        prefill_messages=prefill_messages,
        request_overrides=request_overrides,
        checkpoints_enabled=False,
        pass_session_id=False,
        skip_context_files=False,
        skip_memory=False,
    )
    # Stash a flag so the UI knows whether to load old messages from SessionDB.
    agent._tui_is_resume = _is_resume
    return agent


from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import VerticalScroll, Container, Horizontal, Vertical, Center
from textual.widgets import Static, TextArea, OptionList, Input
from textual.screen import ModalScreen
from textual.message import Message
from textual import events
from textual.reactive import reactive
from rich.markdown import Markdown
from rich.text import Text
from rich.console import Group


# ─── Theme (OpenCode-style warm dark palette) ───

C_BG = "#1b1b19"          # main background
C_PANEL = "#1b1b19"       # sidebar / bars background
C_BORDER = "#3a3a36"      # subtle borders / rules
C_TEXT = "#cfcabb"        # normal foreground
C_DIM = "#6b675c"         # dim / secondary text
C_MAGENTA = "#d75f87"     # section headers / labels
C_BLUE = "#6f9bd8"        # links / values / running tools
C_GREEN = "#87bf7a"       # connected / ready / success
C_RED = "#e06c75"         # errors / failures
C_AMBER = "#d8a657"       # reasoning / thinking accent

# Short, terminal-safe glyphs per tool (narrow Unicode — no emoji width traps).
_TOOL_ICONS = {
    "terminal": "▸",
    "process": "▸",
    "execute_code": "›_",
    "read_file": "▤",
    "write_file": "✎",
    "patch": "✎",
    "search_files": "⌕",
    "web_search": "⌘",
    "web_extract": "⌘",
    "browser_navigate": "◍",
    "browser_click": "◍",
    "browser_type": "◍",
    "delegate_task": "◆",
    "clarify": "?",
    "todo": "☑",
    "image_generate": "▦",
    "text_to_speech": "♪",
    "vision_analyze": "◉",
    "skill_view": "▣",
    "skill_manage": "▣",
    "skills_list": "▣",
    "cronjob": "◷",
}

_SPINNER = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]

# Regex to strip pseudo-XML tool call blocks that some models (e.g. glm-5.1-fp8)
# emit inline in content instead of using the API tool_calls field:
#   ▸\n{"name":"web_search",...}\n◂  and  <tool_result>...</tool_result>
_RE_TOOL_CALL = re.compile(
    r"▸[\s\S]*?◂",            # ▸...◂ pseudo tool call
)
_RE_TOOL_RESULT = re.compile(
    r"<tool_result>[\s\S]*?</tool_result>",  # <tool_result>...</tool_result>
)
_RE_MULTI_NEWLINE = re.compile(r"\n{3,}")    # 3+ newlines → 2


def _strip_inline_tool_xml(text: str) -> str:
    """Remove inline pseudo-XML tool call/result blocks, think tags, and collapse extra newlines.

    Delegates to agent_runtime_helpers.strip_think_blocks for the heavy lifting
    (think/reasoning tags, tool-call XML blocks), then applies our own ▸...◂
    pseudo-tool-call pattern and newline collapsing.
    """
    if not text:
        return ""
    # Use the agent's built-in scrubber for think blocks and tool-call XML.
    try:
        from agent.agent_runtime_helpers import strip_think_blocks as _strip_think
        text = _strip_think(None, text)
    except Exception:
        # Fallback: minimal think-tag stripping if import fails.
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<thinking>.*?</thinking>', '', text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r'<reasoning>.*?</reasoning>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Our own patterns for ▸...◂ pseudo tool calls and <tool_result> blocks.
    text = _RE_TOOL_CALL.sub("", text)
    text = _RE_TOOL_RESULT.sub("", text)
    text = _RE_MULTI_NEWLINE.sub("\n\n", text)
    return text.strip()


def _fmt_duration(seconds: float | None) -> str:
    if seconds is None:
        return ""
    if seconds < 10:
        return f"{seconds:.1f}s"
    if seconds < 60:
        return f"{round(seconds)}s"
    mins, secs = divmod(int(round(seconds)), 60)
    return f"{mins}m {secs}s" if secs else f"{mins}m"


def _tool_preview(name: str, args: dict) -> str:
    """Best-effort one-line preview of a tool call's primary argument."""
    try:
        from agent.display import build_tool_preview
        result = build_tool_preview(name, args, max_len=72)
        if result and result != "None":
            return result
    except Exception:
        pass
    if isinstance(args, dict):
        for v in args.values():
            if isinstance(v, str) and v.strip():
                s = v.strip().replace("\n", " ")
                return s[:72] + ("…" if len(s) > 72 else "")
    return ""


def _tool_succeeded(result) -> bool:
    """Inspect a tool result for an explicit failure signal."""
    try:
        data = json.loads(result) if isinstance(result, str) else result
        if isinstance(data, dict):
            if data.get("success") is False:
                return False
            if data.get("error"):
                return False
    except Exception:
        pass
    return True


# ─── Helpers ───

def _copy_to_clipboard(text: str) -> None:
    """Best-effort clipboard copy (macOS / Linux / WSL)."""
    import subprocess
    for cmd in (
        ["pbcopy"],
        ["xclip", "-selection", "clipboard"],
        ["xsel", "--clipboard", "--input"],
        ["wl-copy"],
        ["clip.exe"],
    ):
        try:
            subprocess.run(cmd, input=text.encode(), check=True, timeout=3)
            return
        except Exception:
            pass


def _extract_code_blocks(markdown_text: str) -> list[tuple[str, str]]:
    """Extract all fenced code blocks as (lang, code) tuples."""
    import re
    pattern = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
    return [(m.group(1) or "", m.group(2)) for m in pattern.finditer(markdown_text)]


def _split_markdown_by_codeblocks(markdown_text: str) -> list[tuple[str, tuple | str]]:
    """Split markdown into alternating ('markdown', text) and ('code', (lang, code)) segments."""
    import re
    pattern = re.compile(r"```(\w*)\n(.*?)\n```", re.DOTALL)
    parts = []
    last_end = 0
    for m in pattern.finditer(markdown_text):
        start, end = m.span()
        if start > last_end:
            parts.append(("markdown", markdown_text[last_end:start]))
        parts.append(("code", (m.group(1) or "", m.group(2))))
        last_end = end
    if last_end < len(markdown_text):
        parts.append(("markdown", markdown_text[last_end:]))
    return parts


# ─── Message widgets (document style) ───

class UserMessage(Static):
    """A user prompt — magenta '❯' gutter, left-aligned."""
    def __init__(self, text: str, **kwargs):
        content = Text()
        content.append("❯ ", style=f"bold {C_MAGENTA}")
        content.append(text, style=C_TEXT)
        super().__init__(content, **kwargs)
        self.add_class("msg-user")


class DiffBlock(Static):
    """Inline diff display for file modifications — shows +/- lines with color coding."""

    _DIFF_GREEN = "#a9dc76"   # added lines
    _DIFF_RED = "#ff6187"     # removed lines
    _DIFF_DIM = "#6b675c"     # hunk header / line numbers

    def __init__(self, file_path: str, diff_text: str, **kwargs):
        self._file_path = file_path
        self._diff_text = diff_text
        super().__init__(self._render_diff(), classes="diff-block", **kwargs)

    def _render_diff(self) -> Text:
        """Render a unified diff as colored Rich Text."""
        result = Text()
        result.append(f"📄 {self._file_path}", style=C_BLUE)
        result.append("\n")

        for line in self._diff_text.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                continue
            if line.startswith("@@"):
                result.append(line, style=self._DIFF_DIM)
                result.append("\n")
            elif line.startswith("+"):
                result.append(line, style=self._DIFF_GREEN)
                result.append("\n")
            elif line.startswith("-"):
                result.append(line, style=self._DIFF_RED)
                result.append("\n")
            elif line.startswith(" "):
                result.append(line, style=C_DIM)
                result.append("\n")

        return result


class CopyButton(Static):
    """A small clickable header above a code block — click to copy the code."""

    def __init__(self, lang: str, code: str, **kwargs):
        self._code = code
        content = Text("Copy", style=f"underline {C_BLUE}", justify="right")
        super().__init__(content, **kwargs)
        self.add_class("copy-btn")

    def on_click(self) -> None:
        _copy_to_clipboard(self._code)
        self.update(Text("Copied!", style=f"bold {C_GREEN}", justify="right"))
        def _reset():
            time.sleep(1.5)
            try:
                self.app.call_from_thread(self.update, Text("Copy", style=f"underline {C_BLUE}", justify="right"))
            except Exception:
                pass
        threading.Thread(target=_reset, daemon=True).start()


class ToolCallBlock(Static):
    """A single tool invocation — running → done/failed, updated in place."""

    def __init__(self, name: str, preview: str = "", **kwargs):
        self._tool_name = str(name) if name else "?"
        self._preview = preview or ""
        self._idx = 0
        self._done = False
        super().__init__(self._running_text(), **kwargs)
        self.add_class("tool-call")
        self.add_class("running")

    def _icon(self) -> str:
        return _TOOL_ICONS.get(self._tool_name, "⚙")

    def _running_text(self) -> Text:
        sp = _SPINNER[self._idx % len(_SPINNER)]
        t = Text()
        t.append(f"{sp} ", style=C_BLUE)
        t.append(f"{self._icon()} ", style=C_BLUE)
        t.append(str(self._tool_name), style=f"bold {C_BLUE}")
        if self._preview:
            t.append("  ", style=C_DIM)
            t.append(str(self._preview), style=C_DIM)
        return t

    def tick(self):
        if not self._done:
            self._idx += 1
            self.update(self._running_text())

    def complete(self, ok: bool, duration: str = "", summary: str = ""):
        self._done = True
        self.remove_class("running")
        t = Text()
        if ok:
            t.append("✓ ", style=C_GREEN)
            t.append(f"{self._icon()} ", style=C_DIM)
            t.append(str(self._tool_name), style=C_TEXT)
        else:
            t.append("✗ ", style=C_RED)
            t.append(f"{self._icon()} ", style=C_RED)
            t.append(str(self._tool_name), style=C_RED)
            self.add_class("failed")
        if self._preview:
            t.append("  ", style=C_DIM)
            t.append(str(self._preview), style=C_DIM)
        meta = [m for m in (duration, summary) if m]
        if meta:
            t.append("  · " + " · ".join(meta), style=C_DIM)
        self.update(t)


class ToolSummaryBlock(Static):
    """A collapsible summary row that replaces individual ToolCallBlocks after completion.

    Collapsed: shows "▸ N tools · X.Xs" (click to expand)
    Expanded:  shows each tool's name + duration (click to collapse)
    """

    def __init__(self, **kwargs):
        self._completed: list[tuple[str, str, str, str, bool]] = []  # (icon, name, duration, preview, ok)
        self._expanded = False
        self._total_started: float | None = None
        super().__init__(Text("", style=C_DIM), **kwargs)
        self.add_class("tool-summary")
        self.add_class("collapsed")

    def add_completed(self, icon: str, name: str, duration: str, preview: str = "", ok: bool = True):
        """Record a completed tool."""
        self._completed.append((icon, name, duration, preview, ok))
        self._refresh()

    def set_total_duration(self, seconds: float):
        """Override total duration (set at turn end for accuracy)."""
        self._total_duration_override = _fmt_duration(seconds)
        self._refresh()

    def _total_duration_str(self) -> str:
        if hasattr(self, "_total_duration_override"):
            return self._total_duration_override
        return ""

    def _refresh(self):
        n = len(self._completed)
        if n == 0:
            self.update(Text(""))
            return

        if self._expanded:
            # Expanded: show each tool line
            lines = []
            for icon, name, duration, preview, ok in self._completed:
                t = Text()
                if ok:
                    t.append("  ✓ ", style=C_GREEN)
                else:
                    t.append("  ✗ ", style=C_RED)
                t.append(f"{icon} ", style=C_DIM if ok else C_RED)
                t.append(name, style=C_TEXT if ok else C_RED)
                if preview:
                    t.append(f"  {preview}", style=C_DIM)
                if duration:
                    t.append(f"  · {duration}", style=C_DIM)
                lines.append(t)
            self.update(Group(*lines))
        else:
            # Collapsed: single summary line
            n_ok = sum(1 for _, _, _, _, ok in self._completed if ok)
            n_fail = n - n_ok
            t = Text()
            t.append("▸ ", style=C_DIM)
            tool_names = [name for _, name, _, _, _ in self._completed]
            # Show up to 3 tool names, then +N
            if n <= 3:
                label = ", ".join(tool_names)
            else:
                label = ", ".join(tool_names[:3]) + f" +{n - 3}"
            t.append(f"{n} tool{'s' if n != 1 else ''}", style=C_TEXT)
            t.append(f"  {label}", style=C_DIM)
            if n_fail > 0:
                t.append(f"  · {n_fail} failed", style=C_RED)
            dur = self._total_duration_str()
            if dur:
                t.append(f"  · {dur}", style=C_DIM)
            self.update(t)

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self.remove_class("collapsed")
            self.add_class("expanded")
        else:
            self.remove_class("expanded")
            self.add_class("collapsed")
        self._refresh()


class ReasoningBlock(Static):
    """Streamed model reasoning / thinking — collapsible, dim, italic.

    Active/streaming: shows live text with amber border.
    Finalized (collapsed): shows "▸ thinking · N lines" — click to expand.
    Finalized (expanded): shows full text — click to collapse.
    """

    def __init__(self, **kwargs):
        self._buffer = ""
        self._expanded = False
        super().__init__(Text("⠋ thinking…", style=f"italic {C_DIM}"), **kwargs)
        self.add_class("reasoning")
        self.add_class("active")

    def append(self, text: str):
        if not text:
            return
        self._buffer += text
        # While streaming, show live text.
        self._refresh_body()

    def _refresh_body(self):
        body = self._buffer.strip()
        if not body:
            return
        head = Text("thinking", style=f"bold {C_AMBER}")
        self.update(Group(head, Text(body, style=f"italic {C_DIM}")))

    @property
    def has_text(self) -> bool:
        return bool(self._buffer.strip())

    def finalize(self):
        self.remove_class("active")
        if not self._buffer.strip():
            self.remove()
            return
        # Default to collapsed after finalization.
        self._expanded = False
        self.add_class("collapsed")
        self.remove_class("expanded")
        self._render_collapsed()

    def _line_count(self) -> int:
        return self._buffer.strip().count("\n") + 1

    def _render_collapsed(self):
        n = self._line_count()
        t = Text()
        t.append("▸ ", style=C_DIM)
        t.append("thinking", style=f"bold {C_AMBER}")
        t.append(f"  · {n} lines", style=C_DIM)
        self.update(t)

    def _render_expanded(self):
        body = self._buffer.strip()
        head = Text("▾ thinking", style=f"bold {C_AMBER}")
        self.update(Group(head, Text(body, style=f"italic {C_DIM}")))

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self.remove_class("collapsed")
            self.add_class("expanded")
            self._render_expanded()
        else:
            self.remove_class("expanded")
            self.add_class("collapsed")
            self._render_collapsed()


class ReasoningSummaryBlock(Static):
    """A single collapsible row that collects all thinking/reasoning across the turn.

    Collapsed: "▸ thinking · N segments · M lines" (click to expand)
    Expanded:  each segment shown, separated by dim rules (click to collapse)
    """

    def __init__(self, **kwargs):
        self._segments: list[str] = []  # collected reasoning texts
        self._expanded = False
        super().__init__(Text(""), **kwargs)
        self.add_class("reasoning-summary")
        self.add_class("collapsed")

    def add_segment(self, text: str):
        """Add a finalized reasoning segment."""
        text = text.strip()
        if not text:
            return
        self._segments.append(text)
        self._refresh()

    def _total_lines(self) -> int:
        return sum(s.count("\n") + 1 for s in self._segments)

    def _refresh(self):
        n = len(self._segments)
        if n == 0:
            self.update(Text(""))
            return

        if self._expanded:
            parts: list = []
            for i, seg in enumerate(self._segments):
                if i > 0:
                    parts.append(Text("  ──", style=C_DIM))
                parts.append(Text(seg, style=f"italic {C_DIM}"))
            head = Text("▾ thinking", style=f"bold {C_AMBER}")
            self.update(Group(head, *parts))
        else:
            t = Text()
            t.append("▸ ", style=C_DIM)
            t.append("thinking", style=f"bold {C_AMBER}")
            t.append(f"  · {n} segment{'s' if n != 1 else ''}", style=C_DIM)
            t.append(f"  · {self._total_lines()} lines", style=C_DIM)
            self.update(t)

    def toggle(self):
        self._expanded = not self._expanded
        if self._expanded:
            self.remove_class("collapsed")
            self.add_class("expanded")
        else:
            self.remove_class("expanded")
            self.add_class("collapsed")
        self._refresh()


# ─── Provider picker modal ───

class ProviderPickerScreen(ModalScreen[str]):
    """Modal dialog to pick a custom provider from config."""

    BINDINGS = [
        Binding("escape", "close", "Cancel"),
    ]

    def __init__(self, current_provider: str, **kwargs):
        super().__init__(**kwargs)
        self._current_provider = current_provider
        self._providers: list[tuple[str, str]] = []  # (slug, "slug · model · base_url")
        self._build_provider_list()

    def _build_provider_list(self):
        """Collect custom providers from config + built-in providers."""
        try:
            import yaml
            cfg_path = os.path.join(os.environ.get("HERMES_HOME", "") or os.path.expanduser("~/.hermes"), "config.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        # Custom providers
        custom = cfg.get("custom_providers", [])
        if isinstance(custom, list):
            for p in custom:
                if isinstance(p, dict):
                    name = p.get("name", "")
                    model = p.get("model", "")
                    base_url = p.get("base_url", "")
                    label = f"{name} · {model} · {base_url}" if model else f"{name} · {base_url}"
                    self._providers.append((name, label))
        elif isinstance(custom, dict):
            for name, val in custom.items():
                model = val.get("model", "") if isinstance(val, dict) else ""
                base_url = val.get("base_url", "") if isinstance(val, dict) else ""
                label = f"{name} · {model} · {base_url}" if model else f"{name} · {base_url}"
                self._providers.append((name, label))

        # Built-in providers with configured keys
        providers = cfg.get("providers", {})
        if isinstance(providers, dict):
            for name in providers:
                if not any(s == name for s, _ in self._providers):
                    self._providers.append((name, name))

        if not self._providers:
            self._providers = [("(none)", "No providers configured")]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="provider-picker"):
                yield Static("  Switch Provider  (Esc to cancel)", id="pp-header")
                yield OptionList(*[label for _, label in self._providers], id="pp-list")

    def on_mount(self) -> None:
        ol = self.query_one("#pp-list", OptionList)
        for i, (slug, _) in enumerate(self._providers):
            if slug.lower() == (self._current_provider or "").lower():
                ol.highlighted = i
                break
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        idx = event.option_index
        if 0 <= idx < len(self._providers):
            self.dismiss(self._providers[idx][0])
        else:
            self.dismiss("")

    def action_close(self):
        self.dismiss("")


# ─── Approval modal ───

class ApprovalScreen(ModalScreen[str]):
    """Modal dialog for dangerous command approval.

    Blocks the agent thread via threading.Event until the user picks a choice.
    Returns one of: 'once', 'session', 'always', 'deny'.
    """

    BINDINGS = [
        Binding("escape", "deny", "Deny"),
    ]

    def __init__(self, command: str, description: str,
                 allow_permanent: bool = True, **kwargs):
        super().__init__(**kwargs)
        self._command = command
        self._description = description
        self._allow_permanent = allow_permanent

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="prompt-modal"):
                yield Static("  ⚠  Command Approval Required", id="prompt-modal-title")
                yield Static("", id="prompt-modal-body")
                yield Static("", id="prompt-modal-hint")

    def on_mount(self) -> None:
        body = self.query_one("#prompt-modal-body", Static)
        cmd_display = self._command
        if len(cmd_display) > 300:
            cmd_display = cmd_display[:300] + "…"
        body.update(Text.from_markup(
            f"[{C_RED}]{cmd_display}[/{C_RED}]\n"
            f"[{C_DIM}]{self._description}[/{C_DIM}]"
        ))
        hint = self.query_one("#prompt-modal-hint", Static)
        keys = "[o] Once   [s] Session"
        if self._allow_permanent:
            keys += "   [a] Always"
        keys += "   [d/Esc] Deny"
        hint.update(Text(keys, style=C_DIM))

    def _key_choice(self, key: str) -> None:
        mapping = {
            "o": "once",
            "s": "session",
            "a": "always",
            "d": "deny",
        }
        choice = mapping.get(key.lower())
        if choice == "always" and not self._allow_permanent:
            choice = None
        if choice:
            self.dismiss(choice)

    async def _on_key(self, event: events.Key) -> None:
        if event.key in ("o", "s", "a", "d"):
            event.stop()
            event.prevent_default()
            self._key_choice(event.key)
            return
        # Enter defaults to deny
        if event.key == "enter":
            event.stop()
            event.prevent_default()
            self.dismiss("deny")
            return
        await super()._on_key(event)

    def action_deny(self):
        self.dismiss("deny")


# ─── Clarify modal ───

class ClarifyScreen(ModalScreen[str]):
    """Modal dialog for the clarify tool.

    Shows a question and optional choices.  Returns the user's answer.
    """

    BINDINGS = [
        Binding("escape", "skip", "Skip"),
    ]

    def __init__(self, question: str, choices: list | None = None, **kwargs):
        super().__init__(**kwargs)
        self._question = question
        self._choices = choices or []

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="prompt-modal"):
                yield Static("  ❓  Clarification", id="prompt-modal-title")
                yield Static("", id="prompt-modal-body")
                yield Input("", id="prompt-modal-input", placeholder="Type your answer…")
                yield Static("", id="prompt-modal-hint")

    def on_mount(self) -> None:
        body = self.query_one("#prompt-modal-body", Static)
        q_display = self._question
        if len(q_display) > 500:
            q_display = q_display[:500] + "…"
        body.update(Markdown(q_display))

        hint = self.query_one("#prompt-modal-hint", Static)
        if self._choices:
            choice_labels = "  ".join(
                f"[{i+1}] {c}" for i, c in enumerate(self._choices)
            )
            hint.update(Text(
                f"Number or type answer · Enter to submit · Esc to skip\n{choice_labels}",
                style=C_DIM,
            ))
        else:
            hint.update(Text("Enter to submit · Esc to skip", style=C_DIM))

        self.query_one("#prompt-modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        text = event.value.strip()
        if not text:
            self.dismiss("")
            return
        # If user typed a number matching a choice, return that choice's text
        if self._choices:
            try:
                idx = int(text) - 1
                if 0 <= idx < len(self._choices):
                    self.dismiss(str(self._choices[idx]))
                    return
            except (ValueError, TypeError):
                pass
        self.dismiss(text)

    def action_skip(self):
        self.dismiss("")


# ─── Sudo password modal ───

class SudoScreen(ModalScreen[str]):
    """Modal dialog for sudo password input."""

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="prompt-modal"):
                yield Static("  🔑  Sudo Password Required", id="prompt-modal-title")
                yield Static("A command requires sudo privileges.", id="prompt-modal-body")
                yield Input("", id="prompt-modal-input", placeholder="Password…", password=True)
                yield Static("Enter to submit · Esc to cancel (command will fail)", id="prompt-modal-hint")

    def on_mount(self) -> None:
        self.query_one("#prompt-modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss(event.value)

    def action_cancel(self):
        self.dismiss("")


# ─── Secret capture modal ───

class SecretScreen(ModalScreen[dict]):
    """Modal dialog for capturing secret values (API keys etc).

    Returns a dict with 'value' key (empty string = skipped).
    """

    BINDINGS = [
        Binding("escape", "skip", "Skip"),
    ]

    def __init__(self, env_var: str, prompt_text: str,
                 metadata: dict | None = None, **kwargs):
        super().__init__(**kwargs)
        self._env_var = env_var
        self._prompt_text = prompt_text
        self._metadata = metadata

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="prompt-modal"):
                yield Static("  🔐  Secret Required", id="prompt-modal-title")
                yield Static("", id="prompt-modal-body")
                yield Input("", id="prompt-modal-input", placeholder="Enter secret value…", password=True)
                yield Static("", id="prompt-modal-hint")

    def on_mount(self) -> None:
        body = self.query_one("#prompt-modal-body", Static)
        meta_info = ""
        if self._metadata:
            skill = self._metadata.get("skill_name", "")
            if skill:
                meta_info = f"\n[{C_DIM}Skill: {skill}[/{C_DIM}]"
            help_text = self._metadata.get("help", "")
            if help_text:
                meta_info += f"\n[{C_DIM}]{help_text}[/{C_DIM}]"
        body.update(Text.from_markup(
            f"[{C_BLUE}]{self._env_var}[/{C_BLUE}]\n"
            f"{self._prompt_text}{meta_info}"
        ))
        hint = self.query_one("#prompt-modal-hint", Static)
        hint.update(Text("Enter to submit · Esc to skip", style=C_DIM))
        self.query_one("#prompt-modal-input", Input).focus()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        event.stop()
        self.dismiss({"value": event.value.strip()})

    def action_skip(self):
        self.dismiss({"value": ""})


# ─── Model picker modal ───

class ModelPickerScreen(ModalScreen[str]):
    """Modal dialog to pick a model or custom provider."""

    BINDINGS = [
        Binding("escape", "close", "Cancel"),
    ]

    def __init__(self, current_model: str, current_provider: str = "", **kwargs):
        super().__init__(**kwargs)
        self._current_model = current_model
        self._current_provider = current_provider
        self._entries: list[tuple[str, str]] = []  # (value_to_apply, display_label)
        self._build_entries()

    def _build_entries(self):
        """Build combined list: custom providers first, then catalog models."""
        # Custom providers from config
        try:
            import yaml
            cfg_path = os.path.join(os.environ.get("HERMES_HOME", "") or os.path.expanduser("~/.hermes"), "config.yaml")
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception:
            cfg = {}

        custom = cfg.get("custom_providers", [])
        if isinstance(custom, list):
            for p in custom:
                if isinstance(p, dict):
                    name = p.get("name", "")
                    model = p.get("model", "")
                    base_url = p.get("base_url", "")
                    if name:
                        label = f"◆ {name} · {model} · {base_url}" if model else f"◆ {name} · {base_url}"
                        # Encode as "provider:slug" so _apply_model_switch knows
                        self._entries.append((f"provider:{name}", label))
        elif isinstance(custom, dict):
            for name, val in custom.items():
                model = val.get("model", "") if isinstance(val, dict) else ""
                base_url = val.get("base_url", "") if isinstance(val, dict) else ""
                label = f"◆ {name} · {model} · {base_url}" if model else f"◆ {name} · {base_url}"
                self._entries.append((f"provider:{name}", label))

        # Separator
        if self._entries:
            self._entries.append(("", "─── Catalog Models ───"))

        # Standard catalog models
        try:
            from hermes_cli.models import _PROVIDER_MODELS
            seen = set()
            for provider, models in _PROVIDER_MODELS.items():
                for m in models:
                    key = m.lower()
                    if key not in seen:
                        seen.add(key)
                        self._entries.append((m, f"  {m}"))
        except Exception:
            pass

        if not self._entries:
            self._entries = [("", "No models found")]

    def compose(self) -> ComposeResult:
        with Center():
            with Vertical(id="model-picker"):
                yield Static("  Switch Model / Provider  (Esc to cancel)", id="mp-header")
                yield OptionList(*[label for _, label in self._entries], id="mp-list")

    def on_mount(self) -> None:
        ol = self.query_one("#mp-list", OptionList)
        # Highlight current provider or model
        for i, (val, _) in enumerate(self._entries):
            if val.startswith("provider:"):
                slug = val[len("provider:"):]
                if slug.lower() == (self._current_provider or "").lower():
                    ol.highlighted = i
                    break
            elif val.lower() == (self._current_model or "").lower():
                ol.highlighted = i
                break
        ol.focus()

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        idx = event.option_index
        if 0 <= idx < len(self._entries):
            val = self._entries[idx][0]
            self.dismiss(val if val else "")
        else:
            self.dismiss("")

    def action_close(self):
        self.dismiss("")


# ─── Multiline prompt input ───

class SessionPickerScreen(ModalScreen[str | None]):
    """Modal to browse and resume recent sessions from SessionDB."""

    BINDINGS = [
        Binding("escape", "close", "Close", show=False),
        Binding("q", "close", "Close", show=False),
    ]

    def __init__(self, current_session_id: str = "", **kwargs):
        self._current_session_id = current_session_id
        super().__init__(**kwargs)

    def compose(self) -> ComposeResult:
        yield Static("📂 Recent Sessions", id="sp-title")
        yield OptionList(id="sp-list")
        yield Static("Enter to resume · Esc to close", id="sp-hint")

    def on_mount(self) -> None:
        self._load_sessions()

    def _load_sessions(self):
        try:
            from hermes_state import SessionDB
            db = SessionDB()
            sessions = db.list_sessions(limit=30)
        except Exception:
            sessions = []

        ol = self.query_one("#sp-list", OptionList)
        for s in sessions:
            sid = s.get("session_id", "")
            title = s.get("title", "") or sid[:12]
            msg_count = s.get("message_count", 0)
            updated = s.get("updated_at", "")
            if isinstance(updated, str):
                updated = updated[:16].replace("T", " ")
            current = " ◀" if sid == self._current_session_id else ""
            label = f"{title} ({msg_count} msgs, {updated}){current}"
            ol.add_option(Option(label, id=sid))

        if not sessions:
            ol.add_option(Option("(no recent sessions)"))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected):
        option = event.option
        sid = getattr(option, "id", None)
        self.dismiss(sid)

    def action_close(self):
        self.dismiss(None)


class SlashCompleter(Static):
    """A thin popup below the prompt that shows matching slash commands."""

    _COMMANDS: list[tuple[str, str]] = [
        ("/help", "Show available commands"),
        ("/model", "Switch model"),
        ("/provider", "Switch provider (custom providers)"),
        ("/compress", "Compress conversation context"),
        ("/retry", "Retry last message"),
        ("/undo", "Remove last exchange"),
        ("/reload", "Reload .env variables"),
        ("/reload-mcp", "Reload MCP servers"),
        ("/reload-skills", "Rescan skills directory"),
        ("/yolo", "Toggle auto-approve all commands"),
        ("/clear", "Clear chat and start new session"),
        ("/quit", "Exit"),
    ]

    def __init__(self, **kwargs):
        super().__init__("", **kwargs)
        self.display = False
        self._matches: list[tuple[str, str]] = []
        self._selected: int = 0

    def update_matches(self, text: str) -> None:
        """Recompute matches from current input text."""
        if not text.startswith("/"):
            self.display = False
            return
        prefix = text.lower().split()[0]
        self._matches = [(c, d) for c, d in self._COMMANDS if c.startswith(prefix)]
        if not self._matches:
            self.display = False
            return
        self._selected = 0
        self.display = True
        self._render_matches()

    def select_next(self) -> None:
        if self._matches:
            self._selected = (self._selected + 1) % len(self._matches)
            self._render_matches()

    def select_prev(self) -> None:
        if self._matches:
            self._selected = (self._selected - 1) % len(self._matches)
            self._render_matches()

    def get_selected(self) -> str | None:
        if self._matches and 0 <= self._selected < len(self._matches):
            return self._matches[self._selected][0]
        return None

    def _render_matches(self) -> None:
        lines = []
        for i, (cmd, desc) in enumerate(self._matches):
            if i == self._selected:
                lines.append(f"  ▸ {cmd:<18} {desc}")
            else:
                lines.append(f"    {cmd:<18} {C_DIM}{desc}")
        self.update(Text.from_markup("\n".join(lines)))


class PromptArea(TextArea):
    """Multiline input.

    Submit / newline keys (terminal-dependent):
      - Enter            → submit
      - Ctrl+J           → newline (reliable in plain terminals)
      - Shift/Alt+Enter  → newline (only with enhanced keyboard protocol)
      - "\\" + Enter      → newline (shell-style continuation, works everywhere)
    """

    _NEWLINE_KEYS = {"shift+enter", "ctrl+j", "alt+enter"}

    class Submitted(Message):
        def __init__(self, value: str):
            self.value = value
            super().__init__()

    async def _on_key(self, event: events.Key) -> None:
        # ── Shift+Tab: cycle permission mode (must intercept before TextArea eats it) ──
        if event.key == "shift+tab":
            event.stop()
            event.prevent_default()
            try:
                self.app.action_cycle_permission()
            except Exception:
                pass
            return

        # Drive slash completer from the parent app.
        app = self.app
        completer = None
        try:
            completer = app.query_one("#slash-completer", SlashCompleter)
        except Exception:
            pass

        # When completer is visible: arrow keys navigate, Tab/Right accepts.
        if completer and completer.display:
            if event.key == "down":
                event.stop()
                event.prevent_default()
                completer.select_next()
                return
            if event.key == "up":
                event.stop()
                event.prevent_default()
                completer.select_prev()
                return
            if event.key in ("tab", "right"):
                event.stop()
                event.prevent_default()
                selected = completer.get_selected()
                if selected:
                    # Replace current text with the selected command.
                    self.text = selected + " "
                    self.cursor_location = (0, len(self.text))
                    completer.display = False
                return
            if event.key == "escape":
                event.stop()
                event.prevent_default()
                completer.display = False
                return

        if event.key == "enter":
            event.stop()
            event.prevent_default()
            # Shell-style continuation: a trailing backslash before the cursor
            # turns Enter into a newline instead of a submit.
            row, col = self.cursor_location
            try:
                line = self.document.get_line(row)
            except Exception:
                line = ""
            if col > 0 and col <= len(line) and line[col - 1] == "\\":
                self.action_delete_left()
                self.insert("\n")
                return
            # Hide completer on submit.
            if completer:
                completer.display = False
            self.post_message(self.Submitted(self.text))
            return
        if event.key in self._NEWLINE_KEYS:
            event.stop()
            event.prevent_default()
            self.insert("\n")
            return
        await super()._on_key(event)

        # After any other key, update completer matches.
        if completer:
            completer.update_matches(self.text)


# ─── Sidebar ───

class Sidebar(Vertical):
    """Right-hand info panel: Context · Usage · MCP · LSP · footer."""

    def compose(self) -> ComposeResult:
        yield Static("", id="sb-title")
        yield Static("", id="sb-context")
        yield Static("", id="sb-tasks")
        yield Static("", id="sb-usage")
        yield Static("", id="sb-mcp")
        yield Static("", id="sb-lsp")
        yield Static("", id="sb-footer")

    def _section(self, header: str, rows: list[tuple[str, str]]) -> Group:
        parts: list = [Text(header, style=f"bold {C_MAGENTA}")]
        for value, style in rows:
            parts.append(Text(value, style=style))
        return Group(*parts)

    def set_title(self, text: str):
        self.query_one("#sb-title", Static).update(Text(text, style=f"bold {C_TEXT}"))

    def set_context(self, tokens: int, pct: float | None, cost: float, compressions: int = 0, context_len: int | None = None):
        rows = [(f"{tokens:,} tokens", C_DIM)]
        if context_len:
            rows.append((f"{context_len:,} ctx", C_DIM))
        rows.append((f"{compressions} compressions", C_DIM))
        self.query_one("#sb-context", Static).update(self._section("Context", rows))

    def set_tasks(self, tasks: list[dict]):
        if not tasks:
            self.query_one("#sb-tasks", Static).update("")
            return
        rows = []
        for t in tasks:
            status = t.get("status", "pending")
            if status == "completed":
                icon, style = "✓", C_GREEN
            elif status == "in_progress":
                icon, style = "▶", C_MAGENTA
            elif status == "cancelled":
                icon, style = "✗", C_RED
            else:
                icon, style = "○", C_DIM
            content = str(t.get("content", "") or t.get("id", ""))
            if len(content) > 30:
                content = content[:28] + "…"
            rows.append((f"{icon} {content}", style))
        self.query_one("#sb-tasks", Static).update(self._section("Tasks", rows))

    def set_usage(self, tools: list[str], skills: list[str]):
        rows: list[tuple[str, str]] = []
        for s in skills:
            rows.append((f"◆ {s}", C_MAGENTA))
        for t in tools:
            rows.append((f"· {t}", C_DIM))
        if not rows:
            rows = [("no tools used", C_DIM)]
        self.query_one("#sb-usage", Static).update(self._section("Usage", rows))

    def set_mcp(self, servers: list[tuple[str, bool]]):
        if servers:
            rows = []
            for name, ok in servers:
                dot = "●" if ok else "○"
                label = "Connected" if ok else "Disconnected"
                style = C_GREEN if ok else C_DIM
                rows.append((f"{dot} {name} {label}", style))
        else:
            rows = [("no MCP servers", C_DIM)]
        self.query_one("#sb-mcp", Static).update(self._section("MCP", rows))

    def set_lsp(self, text: str = "LSPs are disabled"):
        self.query_one("#sb-lsp", Static).update(self._section("LSP", [(text, C_DIM)]))

    def set_footer(self, cwd: str, version: str):
        g = Group(
            Text(cwd, style=C_DIM),
            Text(f"● Hermes {version}", style=C_GREEN),
        )
        self.query_one("#sb-footer", Static).update(g)


# ─── Main App ───

class HermesApp(App):
    """Hermes Agent TUI — OpenCode-style layout."""

    CSS = """
    Screen {
        layout: vertical;
        background: #1b1b19;
    }

    /* ── Body: main transcript + sidebar ── */
    #body {
        height: 1fr;
    }

    #chat-area {
        width: 1fr;
        height: 1fr;
        background: #1b1b19;
        padding: 1 2 0 2;
        scrollbar-size: 1 1;
        scrollbar-background: #1b1b19;
        scrollbar-color: #3a3a36;
    }

    /* ── Sidebar ── */
    #sidebar {
        width: 32;
        height: 1fr;
        background: #1b1b19;
        border-left: solid #3a3a36;
        padding: 1 2;
    }

    #sb-title  { margin: 0 0 1 0; }
    #sb-context { margin: 0 0 1 0; }
    #sb-tasks   { margin: 0 0 1 0; }
    #sb-usage   { margin: 0 0 1 0; }
    #sb-mcp     { margin: 0 0 1 0; }
    #sb-lsp     { margin: 0 0 1 0; }
    #sb-footer  { dock: bottom; }

    /* ── Messages ── */
    .msg-user {
        margin: 1 0 1 0;
        color: #cfcabb;
        width: 100%;
    }

    .msg-assistant {
        margin: 0 0 1 0;
        color: #cfcabb;
        width: 100%;
    }

    .msg-assistant.streaming {
        border-left: tall #d75f87;
        padding-left: 1;
    }

    /* ── Diff block ── */
    .diff-block {
        margin: 0 0 1 0;
        padding: 0 1;
        width: 100%;
        background: #1e1e1c;
        border-left: tall #4a90d9;
    }

    /* ── Session picker ── */
    #sp-title {
        text-align: center;
        padding: 1;
        color: #cfcabb;
        text-style: bold;
    }
    #sp-list {
        height: 16;
        background: #1b1b19;
    }
    #sp-hint {
        text-align: center;
        padding: 0 1;
        color: #6b675c;
    }

    /* ── Reasoning / thinking block ── */
    .reasoning {
        margin: 0 0 1 0;
        width: 100%;
        padding-left: 1;
        color: #6b675c;
    }

    .reasoning.active {
        border-left: tall #d8a657;
    }

    .reasoning.collapsed {
        border-left: tall #3a3a36;
    }

    .reasoning.collapsed:hover {
        border-left: tall #d8a657;
        background: #222220;
    }

    .reasoning.expanded {
        border-left: tall #d8a657;
    }

    /* ── System message (slash command feedback) ── */
    .system-msg {
        margin: 1 0;
        width: 100%;
        padding: 0 1;
        color: #888;
    }

    /* ── Slash command completer ── */
    #slash-completer {
        width: 100%;
        height: auto;
        max-height: 10;
        background: #252522;
        color: #cfcabb;
        padding: 0 1;
        border-bottom: tall #3a3a36;
    }

    /* ── Model picker modal ── */
    #model-picker {
        width: 60;
        height: 22;
        background: #1e1e1c;
        border: tall $primary;
        padding: 0;
    }
    #mp-header {
        width: 100%;
        padding: 0 1;
        background: #2a2a27;
        color: $text;
        text-style: bold;
    }
    #mp-list {
        width: 100%;
        height: 1fr;
    }
    #mp-list:focus > .option-list--option-highlighted {
        background: #3a3a36;
    }

    /* ── Provider picker modal ── */
    #provider-picker {
        width: 60;
        height: 14;
        background: #1e1e1c;
        border: tall $primary;
        padding: 0;
    }
    #pp-header {
        width: 100%;
        padding: 0 1;
        background: #2a2a27;
        color: $text;
        text-style: bold;
    }
    #pp-list {
        width: 100%;
        height: 1fr;
    }
    #pp-list:focus > .option-list--option-highlighted {
        background: #3a3a36;
    }

    /* ── Prompt modals (approval / clarify / sudo / secret) ── */
    #prompt-modal {
        width: 64;
        max-height: 20;
        background: #1e1e1c;
        border: tall $primary;
        padding: 1 2;
    }
    #prompt-modal-title {
        width: 100%;
        padding: 0 0 1 0;
        color: $text;
        text-style: bold;
    }
    #prompt-modal-body {
        width: 100%;
        padding: 0 0 1 0;
        color: #cfcabb;
    }
    #prompt-modal-input {
        width: 100%;
        height: 3;
        margin: 0 0 1 0;
    }
    #prompt-modal-hint {
        width: 100%;
        color: #6b675c;
    }

    /* ── Reasoning summary (collapsed/expanded) ── */
    .reasoning-summary {
        margin: 0 0 1 0;
        width: 100%;
        padding: 0 1;
        border-left: tall #3a3a36;
        color: #6b675c;
    }

    .reasoning-summary.collapsed:hover {
        border-left: tall #d8a657;
        background: #222220;
    }

    .reasoning-summary.expanded {
        border-left: tall #d8a657;
    }

    /* ── Tool-call row ── */
    .tool-call {
        margin: 0 0 1 0;
        width: 100%;
        padding-left: 1;
    }

    .tool-call.running {
        border-left: tall #6f9bd8;
    }

    .tool-call.failed {
        border-left: tall #e06c75;
    }

    /* ── Tool summary (collapsed/expanded) ── */
    .tool-summary {
        margin: 0 0 1 0;
        width: 100%;
        padding: 0 1;
        border-left: tall #3a3a36;
        color: #6b675c;
    }

    .tool-summary.collapsed:hover {
        border-left: tall #6f9bd8;
        background: #222220;
    }

    .tool-summary.expanded {
        border-left: tall #6f9bd8;
    }

    .welcome {
        margin: 1 0;
        color: #6b675c;
        width: 100%;
    }

    .error-msg {
        margin: 1 0;
        padding: 0 1;
        border-left: tall #e06c75;
        color: #e06c75;
        width: 100%;
    }

    /* ── Copy button for code blocks ── */
    .copy-btn {
        margin: 0 0 0 0;
        width: 100%;
        padding: 0 1;
        color: #6b675c;
        text-style: bold;
        text-align: right;
        height: 1;
    }

    .copy-btn:hover {
        background: #222220;
        color: #cfcabb;
    }

    /* ── Input box (rounded, multiline, auto-grow) ── */
    #input-container {
        height: auto;
        min-height: 3;
        max-height: 14;
        background: #1b1b19;
        border: round #3a3a36;
        margin: 1 0 0 0;
        padding: 0 1;
    }

    #prompt-input {
        width: 100%;
        height: auto;
        min-height: 1;
        max-height: 12;
        color: #cfcabb;
        background: #1b1b19;
        border: none;
        padding: 0;
        scrollbar-size: 1 1;
        scrollbar-background: #1b1b19;
        scrollbar-color: #3a3a36;
    }

    #prompt-input:focus {
        background: #1b1b19;
    }

    /* ── Status bar ── */
    #notify-bar {
        dock: bottom;
        height: auto;
        max-height: 3;
        padding: 0 2;
        background: #2d2b33;
        color: #cfcabb;
        display: none;
    }
    #notify-bar.active {
        display: block;
    }

    #status-bar {
        height: 1;
        background: #1b1b19;
        color: #6b675c;
        margin: 0 2 1 2;
    }
    """

    TITLE = "Hermes"
    SUB_TITLE = ""

    BINDINGS = [
        Binding("ctrl+c", "quit", "Quit", show=False),
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear_chat", "Clear"),
        Binding("ctrl+n", "new_session", "New session"),
        Binding("ctrl+o", "open_sessions", "Sessions"),
        Binding("ctrl+m", "pick_model", "Model"),
        Binding("ctrl+p", "pick_provider", "Provider"),
        Binding("ctrl+k", "clear_input", "Clear input"),
        Binding("ctrl+r", "retry_last", "Retry"),
        Binding("ctrl+z", "undo_last", "Undo"),
        Binding("ctrl+e", "compress_context", "Compress"),
        Binding("shift+tab", "cycle_permission", "Mode"),
        Binding("escape", "interrupt", "Interrupt", show=False),
    ]

    # ── Working animation (Cylon shuttle) ──
    _BAR_TRACK = 14                       # total cells in the track
    _BAR_SHADES = [                       # comet head → tail (bright → dim yellow)
        "#e8e36a", "#cfc856", "#a89f40", "#7e772f", "#564f22",
    ]
    _BAR_EMPTY = "#33332e"                # dim, unlit track cell
    _BAR_INTERVAL = 0.08                  # seconds per frame

    is_working: reactive[bool] = reactive(False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._agent = None
        self._history: list = []
        # Per-turn state — simple flat model.
        # _stream_buf accumulates all text deltas for this turn.
        # _response_widget is the single Static that shows streamed/final text.
        self._stream_buf: str = ""
        self._response_widget: Static | None = None
        self._active_reasoning: ReasoningBlock | None = None
        self._reasoning_summary: ReasoningSummaryBlock | None = None
        self._tool_blocks: dict[str, tuple[ToolCallBlock, float]] = {}
        self._tool_summary: ToolSummaryBlock | None = None
        self._turn_start_time: float = 0.0
        self._status_kind: str = ""
        self._status_text: str | None = None
        self._turn_tools: list[str] = []
        self._turn_skills: list[str] = []
        self._model_name: str = ""
        self._version: str = ""
        self._context_len: int | None = None
        self._anim_timer = None
        self._anim_phase: int = 0
        self._spinner_idx: int = 0
        self._cc_last_sid: str | None = None  # cache for _agent_compressions
        self._cc_last_db: int = 0
        self._tasks: list[dict] = []  # plan tasks from todo tool
        self._file_snapshots: dict[str, list[str]] = {}  # path → lines before edit (for diff)
        self._permission_mode: str = "default"  # default | plan | auto
        self._PERMISSION_MODES = ["default", "plan", "auto"]

    @property
    def _mode(self) -> str:
        """Display label derived from permission_mode — always in sync."""
        return {"default": "Default", "plan": "Plan", "auto": "Auto"}.get(self._permission_mode, "Default")

    # ── Compose ──

    def compose(self) -> ComposeResult:
        with Horizontal(id="body"):
            yield VerticalScroll(id="chat-area")
            yield Sidebar(id="sidebar")
        with Container(id="input-container"):
            yield SlashCompleter(id="slash-completer")
            prompt = PromptArea(id="prompt-input", soft_wrap=True, show_line_numbers=False)
            yield prompt
        yield Static("", id="notify-bar")
        yield Static("", id="status-bar")

    def on_mount(self) -> None:
        self.query_one("#prompt-input").focus()

        sb = self.query_one(Sidebar)
        sb.set_title("Hermes Agent")
        sb.set_context(0, None, 0.0)
        sb.set_mcp([])
        sb.set_lsp()
        sb.set_footer(self._short_cwd(), self._hermes_version())

        msgs = self.query_one("#chat-area")
        msgs.mount(Static(
            Markdown(
                "**Welcome to Hermes Agent**\n\n"
                "Send a message to start chatting.\n\n"
                "`Enter` Send  ·  `Ctrl+J` Newline  ·  `Ctrl+L` Clear  ·  `Ctrl+N` New\n"
                "`Ctrl+O` Sessions  ·  `Ctrl+M` Model  ·  `Ctrl+P` Provider  ·  `Shift+Tab` Mode\n"
                "`Ctrl+R` Retry  ·  `Ctrl+Z` Undo  ·  `Ctrl+E` Compress  ·  `Ctrl+Q` Quit"
            ),
            classes="welcome",
        ))

        self._render_status()
        self._build_agent_bg()

    # ── Helpers ──

    @staticmethod
    def _short_cwd() -> str:
        try:
            cwd = Path(os.getcwd())
            home = Path.home()
            if cwd == home or home in cwd.parents:
                return "~/" + str(cwd.relative_to(home)) if cwd != home else "~"
            return str(cwd)
        except Exception:
            return os.getcwd()

    @staticmethod
    def _hermes_version() -> str:
        try:
            from hermes_cli import __version__
            return str(__version__)
        except Exception:
            return "dev"

    @staticmethod
    def _fmt_tokens(n: int) -> str:
        if n >= 1000:
            return f"{n / 1000:.1f}K"
        return str(n)

    # ── Working animation ──

    def _work_bar(self) -> Text:
        """A Cylon/Knight-Rider shuttle: a yellow comet bouncing left↔right."""
        track = self._BAR_TRACK
        span = max(track - 1, 1)
        p = self._anim_phase % (2 * span)
        if p <= span:
            head, direction = p, 1
        else:
            head, direction = 2 * span - p, -1
        shades = self._BAR_SHADES
        bar = Text()
        for i in range(track):
            dist = (head - i) * direction
            if 0 <= dist < len(shades):
                bar.append("█", style=shades[dist])
            else:
                bar.append("█", style=self._BAR_EMPTY)
        return bar

    def _tick_anim(self) -> None:
        self._anim_phase += 1
        self._spinner_idx = (self._spinner_idx + 1) % len(_SPINNER)
        self._render_status()
        # Drive the spinner on any running tool rows.
        for block, _started in list(self._tool_blocks.values()):
            try:
                block.tick()
            except Exception:
                pass
        # Also refresh the response widget spinner if streaming.
        if self._response_widget is not None and self.is_working:
            self._refresh_streaming_display()

    # ── Agent init ──

    def _build_agent_bg(self):
        def build():
            try:
                agent = _build_agent()
                self._agent = agent
                self._model_name = getattr(agent, "model", "") or ""
                self.call_from_thread(self._on_agent_ready)
            except Exception as e:
                import traceback
                print(f"[TUI] _build_agent failed: {e}", file=sys.stderr, flush=True)
                traceback.print_exc(file=sys.stderr)
                self.call_from_thread(self._show_error, str(e))

        threading.Thread(target=build, daemon=True).start()

    def _on_agent_ready(self):
        self._version = self._hermes_version()
        self._wire_agent_callbacks()
        self._wire_blocking_callbacks()
        try:
            compressor = getattr(self._agent, "context_compressor", None)
            self._context_len = getattr(compressor, "context_length", None) if compressor else None
        except Exception:
            self._context_len = None

        # ── Resume: load previous session history from SessionDB ──
        self._load_resumed_session()

        # ── Restore token counts / compression count from SessionDB ──
        self._restore_session_state()

        self._refresh_sidebar()
        self._render_status()

    def _restore_session_state(self):
        """Restore session_total_tokens and compression_count from SessionDB
        so the sidebar shows historical values, not zero, on resume."""
        if self._agent is None:
            return
        session_id = getattr(self._agent, "session_id", None)
        if not session_id:
            return

        # Try to get session_db from the agent or create one.
        session_db = (
            getattr(self._agent, "_session_db", None)
            or getattr(self._agent, "session_db", None)
        )
        if not session_db:
            try:
                from hermes_state import SessionDB
                session_db = SessionDB()
            except Exception:
                return

        # ── Restore token counts from session metadata ──
        try:
            meta = session_db.get_session(session_id)
            if meta:
                input_tok = meta.get("input_tokens", 0) or 0
                output_tok = meta.get("output_tokens", 0) or 0
                total = input_tok + output_tok
                if total > self._agent.session_total_tokens:
                    self._agent.session_total_tokens = total
        except Exception:
            pass

        # ── Count compressions from the parent chain ──
        try:
            chain = session_db._session_lineage_root_to_tip(session_id)
            compressions = 0
            for sid in chain[:-1]:  # exclude the current session itself
                ancestor = session_db.get_session(sid)
                if ancestor and ancestor.get("end_reason") == "compression":
                    compressions += 1
            compressor = getattr(self._agent, "context_compressor", None)
            if compressor is not None and compressions > 0:
                compressor.compression_count = compressions
        except Exception:
            pass

    def _load_resumed_session(self):
        """If the agent was started with a resume session ID, load the old
        conversation history from SessionDB and render it in the chat area."""
        if not self._agent:
            return
        session_id = getattr(self._agent, "session_id", None)
        if not session_id:
            return

        # Only resume if _build_agent() detected a --resume flag.
        if not getattr(self._agent, "_tui_is_resume", False):
            return

        session_db = getattr(self._agent, "_session_db", None) or getattr(self._agent, "session_db", None)
        if not session_db:
            try:
                from hermes_state import SessionDB
                session_db = SessionDB()
            except Exception:
                return

        # Resolve compression chain (parent → child with actual messages).
        try:
            resolved_id = session_db.resolve_resume_session_id(session_id)
        except Exception:
            resolved_id = session_id
        if resolved_id and resolved_id != session_id:
            session_id = resolved_id
            self._agent.session_id = session_id

        try:
            restored = session_db.get_messages_as_conversation(session_id, include_ancestors=True)
        except Exception:
            return

        if not restored:
            return

        # Filter out session_meta rows; keep user/assistant/tool messages.
        restored = [m for m in restored if m.get("role") not in ("session_meta",)]
        if not restored:
            return

        # Remove the welcome message before rendering history.
        msgs = self.query_one("#chat-area")
        for w in msgs.query(".welcome"):
            w.remove()

        # Set the internal history so run_conversation() sends it to the LLM.
        self._history = restored

        # Count display messages
        user_msgs = [m for m in restored if m.get("role") == "user"]
        msg_count = len(user_msgs)
        title_part = ""
        try:
            meta = session_db.get_session(session_id)
            if meta and meta.get("title"):
                title_part = f' "{meta["title"]}"'
        except Exception:
            pass

        msgs.mount(Static(
            Text(f"↻ Resumed session {session_id}{title_part} "
                 f"({msg_count} user message{'s' if msg_count != 1 else ''}, "
                 f"{len(restored)} total messages)",
                 style=C_GREEN),
            classes="system-msg",
        ))

        # Render each message in chronological order.
        # - user:      UserMessage widget
        # - assistant (with tool_calls):  ToolCallBlock for each call
        # - assistant (text only):        Markdown widget (full content, not truncated)
        # - tool:      Completion indicator on the matching ToolCallBlock
        for m in restored:
            role = m.get("role", "")
            content = m.get("content", "")
            tool_calls = m.get("tool_calls")

            if role == "user":
                if isinstance(content, str) and content.strip():
                    msgs.mount(UserMessage(content.strip()))

            elif role == "assistant":
                # If assistant message has tool_calls, render tool blocks.
                if tool_calls:
                    for tc in tool_calls:
                        fn = tc.get("function", {})
                        # Handle both OpenAI nested format {"function": {"name": ..., "arguments": ...}}
                        # and the simplified flat format {"name": ..., "arguments": ...} stored by
                        # run_agent._persist_recent_messages.
                        name = (fn.get("name") or tc.get("name")) or "?"
                        arguments = fn.get("arguments") or tc.get("arguments") or "{}"
                        try:
                            import json as _json
                            args = _json.loads(arguments)
                        except Exception:
                            args = {}
                        preview = _tool_preview(name, args)
                        block = ToolCallBlock(name, preview)
                        block.complete(ok=True, summary="replayed")
                        msgs.mount(block)

                # Also render assistant text content (if any).
                if isinstance(content, str) and content.strip():
                    display = content.strip()
                    code_blocks = _extract_code_blocks(display)
                    if code_blocks:
                        parts = _split_markdown_by_codeblocks(display)
                        for kind, content_part in parts:
                            if kind == "markdown":
                                if content_part.strip():
                                    msgs.mount(Static(
                                        Markdown(content_part.strip()),
                                        classes="msg-assistant",
                                    ))
                            elif kind == "code":
                                lang, code = content_part
                                msgs.mount(CopyButton(lang, code))
                                msgs.mount(Static(
                                    Markdown(f"```{lang}\n{code}\n```"),
                                    classes="msg-assistant",
                                ))
                    else:
                        msgs.mount(Static(
                            Markdown(display),
                            classes="msg-assistant",
                        ))

            elif role == "tool":
                # Tool result — don't mount a separate widget, but we could
                # show a minimal result preview in the future. For now the
                # ToolCallBlock above already shows "✓ tool_name · replayed".
                pass

        # Re-open the session (clear ended_at so it's active again).
        try:
            session_db.reopen_session(session_id)
        except Exception:
            pass

        self._scroll_to_bottom()

    def _wire_agent_callbacks(self):
        """Route AIAgent's work callbacks to the UI thread.

        Each callback fires on the agent worker thread, so they hop back to
        the Textual event loop via call_from_thread before touching widgets.

        IMPORTANT: We use stream_delta_callback (NOT stream_callback) to avoid
        double-firing.  _fire_stream_delta() calls both stream_delta_callback
        and _stream_callback; we only set the former.
        """
        a = self._agent
        if a is None:
            return

        def delta(text):
            self.call_from_thread(self._on_delta, text)

        def reasoning(text):
            self.call_from_thread(self._on_reasoning, text)

        def thinking(text):
            self.call_from_thread(self._on_reasoning, text)

        def tool_start(tc_id, name, args):
            self.call_from_thread(self._on_tool_start, tc_id, name, args)

        def tool_complete(tc_id, name, args, result):
            self.call_from_thread(self._on_tool_complete, tc_id, name, args, result)

        def tool_progress(event_type, name=None, preview=None, args=None, **kwargs):
            self.call_from_thread(
                self._on_tool_progress, event_type, name, preview, args, kwargs
            )

        def status(kind, text=None):
            self.call_from_thread(self._on_status, str(kind), text)

        def clarify(question, choices):
            return self._blocking_modal(ClarifyScreen, question, choices)

        a.stream_delta_callback = delta
        a.reasoning_callback = reasoning
        a.thinking_callback = thinking
        a.tool_start_callback = tool_start
        a.tool_complete_callback = tool_complete
        a.tool_progress_callback = tool_progress
        a.status_callback = status
        a.clarify_callback = clarify

    # ── Blocking modal helpers ──

    def _blocking_modal(self, screen_cls, *args, timeout: int = 300):
        """Show a modal screen and block the calling thread until dismissed.

        This is the core mechanism for approval/clarify/sudo/secret callbacks:
        the agent worker thread calls this, we show a modal on the UI thread,
        and block until the user responds.  Returns whatever the modal dismisses.

        Args:
            screen_cls: ModalScreen subclass to show.
            *args: Arguments forwarded to the screen constructor.
            timeout: Maximum seconds to wait (default 5 min).
        """
        result_box = [None]
        event = threading.Event()

        def on_dismiss(value):
            result_box[0] = value
            event.set()

        self.call_from_thread(self._push_blocking_modal, screen_cls, args, on_dismiss)

        # Poll in 1s slices so we can fire activity heartbeats to prevent
        # the gateway inactivity watchdog from killing the agent.
        try:
            from tools.environments.base import touch_activity_if_due
        except Exception:
            touch_activity_if_due = None

        _now = time.monotonic()
        _deadline = _now + timeout
        _activity_state = {"last_touch": _now, "start": _now}

        while True:
            remaining = _deadline - time.monotonic()
            if remaining <= 0:
                break
            if event.wait(timeout=min(1.0, remaining)):
                break
            if touch_activity_if_due is not None:
                touch_activity_if_due(_activity_state, "waiting for user response")

        # Ensure modal is popped if timeout occurred
        if not event.is_set():
            self.call_from_thread(self._pop_blocking_modal_if_still_open)

        return result_box[0]

    def _push_blocking_modal(self, screen_cls, args, on_dismiss):
        """Push a modal screen on the UI thread (called from _blocking_modal)."""
        try:
            screen = screen_cls(*args)
            # Stash the dismiss callback so we can wire it up
            screen._tui_on_dismiss = on_dismiss
            self.push_screen(screen, on_dismiss)
        except Exception as e:
            on_dismiss(None)

    def _pop_blocking_modal_if_still_open(self):
        """Pop the topmost modal if it's one of our blocking ones (timeout cleanup)."""
        try:
            screen = self.screen
            if isinstance(screen, (ApprovalScreen, ClarifyScreen, SudoScreen, SecretScreen)):
                screen.dismiss(None)
        except Exception:
            pass

    def _wire_blocking_callbacks(self):
        """Wire gateway approval system (called once on agent ready).

        Thread-local callbacks (approval, sudo, secret) are NOT set here because
        they live in threading.local() — they must be registered on the actual
        agent worker thread each turn.  See _register_thread_local_callbacks().
        """
        session_key = getattr(self._agent, "session_id", "") or ""

        # ── Enable gateway-style approval env flags ──
        os.environ["HERMES_GATEWAY_SESSION"] = "1"
        os.environ["HERMES_EXEC_ASK"] = "1"
        os.environ["HERMES_INTERACTIVE"] = "1"
        if session_key:
            os.environ["HERMES_SESSION_KEY"] = session_key

        # ── Gateway approval: register_gateway_notify + resolve_gateway_approval ──
        # This handles the is_gateway path inside _check_dangerous_command
        # (tools/approval.py), which blocks on _gateway_queues.
        self._gateway_approval_registered = False
        try:
            from tools.approval import register_gateway_notify, load_permanent_allowlist
            register_gateway_notify(
                session_key,
                lambda data: self._on_gateway_approval_request(data),
            )
            self._gateway_approval_registered = True
            load_permanent_allowlist()
        except Exception:
            pass

    def _register_thread_local_callbacks(self):
        """Register thread-local callbacks on the current (agent worker) thread.

        Called at the start of each _run_turn because _callback_tls is
        per-thread — the UI thread's callbacks are invisible to the worker.
        """
        # ── Thread-local approval callback ──
        try:
            from tools.terminal_tool import set_approval_callback
            set_approval_callback(self._approval_handler)
        except Exception:
            pass

        # ── Sudo password callback ──
        try:
            from tools.terminal_tool import set_sudo_password_callback
            set_sudo_password_callback(self._sudo_handler)
        except Exception:
            pass

        # ── Secret capture callback ──
        try:
            from tools.skills_tool import set_secret_capture_callback
            set_secret_capture_callback(self._secret_handler)
        except Exception:
            pass

    # ── Approval / sudo / secret handlers (run on agent worker thread) ──

    def _approval_handler(self, command, description, *, allow_permanent=True):
        """Thread-local approval callback — called by prompt_dangerous_approval."""
        result = self._blocking_modal(
            ApprovalScreen, command, description, allow_permanent, timeout=300
        )
        return result or "deny"

    def _on_gateway_approval_request(self, data: dict):
        """Gateway notify callback — called when _check_dangerous_command blocks.

        Shows an approval modal and resolves the gateway queue entry when the
        user responds.
        """
        session_key = getattr(self._agent, "session_id", "") or ""
        command = data.get("command", "")
        description = data.get("description", "")
        allow_permanent = not data.get("pattern_keys", [])

        result = self._blocking_modal(
            ApprovalScreen, command, description, allow_permanent, timeout=300
        )
        choice = result or "deny"

        # Resolve the oldest gateway approval entry for this session
        try:
            from tools.approval import resolve_gateway_approval
            resolve_gateway_approval(session_key, choice)
        except Exception:
            pass

    def _sudo_handler(self):
        """Sudo password callback — returns password or empty string."""
        result = self._blocking_modal(SudoScreen, timeout=120)
        return result or ""

    def _secret_handler(self, env_var, prompt_text, metadata=None):
        """Secret capture callback — returns a dict matching the gateway protocol."""
        result = self._blocking_modal(
            SecretScreen, env_var, prompt_text, metadata, timeout=300
        )
        if not result or not result.get("value"):
            return {
                "success": True,
                "stored_as": env_var,
                "validated": False,
                "skipped": True,
                "message": "skipped",
            }
        val = result["value"]
        try:
            from hermes_cli.config import save_env_value_secure
            saved = save_env_value_secure(env_var, val)
        except Exception:
            saved = {"success": False, "stored_as": env_var}
        return {
            **saved,
            "skipped": False,
            "message": "ok",
        }

    # ── Input ──

    def on_prompt_area_submitted(self, event: "PromptArea.Submitted"):
        text = event.value.strip()
        if not text or self._agent is None:
            return

        self.query_one("#prompt-input", PromptArea).text = ""
        msgs = self.query_one("#chat-area")

        for w in msgs.query(".welcome"):
            w.remove()

        # ── Slash command interception ──
        if text.startswith("/"):
            handled = self._process_slash_command(text)
            if handled:
                return
            # Unknown command — fall through and send to agent as regular message.

        msgs.mount(UserMessage(text))
        self._scroll_to_bottom()

        if self.is_working:
            return

        # Reset per-turn state.
        self._stream_buf = ""
        self._response_widget = None
        self._active_reasoning = None
        self._reasoning_summary = None
        self._tool_blocks.clear()
        self._tool_summary = None
        self._turn_start_time = time.time()
        self._status_kind = ""
        self._status_text = None
        self._turn_tools = []
        self._turn_skills = []
        self._spinner_idx = 0

        self.is_working = True
        self._run_turn(text)

    # ── Slash commands ──

    def _process_slash_command(self, text: str) -> bool:
        """Handle slash commands locally. Returns True if consumed, False to pass through."""
        parts = text.split(None, 1)
        cmd = parts[0].lower().lstrip("/")
        args = parts[1].strip() if len(parts) > 1 else ""

        # Resolve aliases via central registry.
        try:
            from hermes_cli.commands import resolve_command
            resolved = resolve_command(cmd)
            canonical = resolved.name if resolved else cmd
        except Exception:
            canonical = cmd

        if canonical == "help":
            return self._cmd_help()
        elif canonical == "model":
            # Check if the user typed /provider explicitly
            if cmd.lower() == "provider":
                return self._cmd_provider(args)
            return self._cmd_model(args)
        elif canonical == "compress":
            return self._cmd_compress(args)
        elif canonical == "retry":
            return self._cmd_retry()
        elif canonical == "undo":
            return self._cmd_undo()
        elif canonical == "reload":
            return self._cmd_reload()
        elif canonical == "reload-mcp":
            return self._cmd_reload_mcp()
        elif canonical == "reload-skills":
            return self._cmd_reload_skills()
        elif canonical == "clear":
            self.action_clear_chat()
            return True
        elif canonical in ("quit", "exit"):
            self.exit()
            return True
        elif canonical == "yolo":
            return self._cmd_yolo(args)
        else:
            # Unknown command — not consumed, send to agent as message.
            return False

    def _show_system_msg(self, msg: str):
        """Show a dim system message in chat area."""
        self.query_one("#chat-area").mount(
            Static(Text(msg, style=C_DIM), classes="system-msg")
        )
        self._scroll_to_bottom()

    def _cmd_help(self) -> bool:
        lines = [
            "▸ Slash commands:",
            "  /help              Show this help",
            "  /model [name]      Switch model (no args = picker)",
            "  /provider [name]   Switch provider (no args = picker)",
            "  /compress [focus]  Compress conversation context",
            "  /retry             Retry last message",
            "  /undo              Remove last exchange from history",
            "  /reload            Reload .env variables",
            "  /reload-mcp        Reload MCP servers from config",
            "  /reload-skills     Rescan ~/.hermes/skills/",
            "  /yolo [off]        Toggle auto-approve all commands",
            "  /clear             Clear chat and start new session",
            "  /quit              Exit",
            "",
            "▸ Keyboard shortcuts:",
            "  Ctrl+M             Switch model",
            "  Ctrl+P             Switch provider",
            "  Shift+Tab          Cycle mode (Default→Plan→Auto)",
            "  Ctrl+N             New session",
            "  Ctrl+O             Browse recent sessions",
            "  Ctrl+R             Retry last message",
            "  Ctrl+Z             Undo last exchange",
            "  Ctrl+E             Compress context",
            "  Ctrl+K             Clear input",
            "  Ctrl+L             Clear chat",
            "  Ctrl+Q             Quit",
        ]
        self.query_one("#chat-area").mount(
            Static(Group(*[Text(l, style=C_DIM) for l in lines]), classes="system-msg")
        )
        self._scroll_to_bottom()
        return True

    def _cmd_model(self, args: str) -> bool:
        if self.is_working:
            self._show_system_msg("Cannot switch model while agent is working.")
            return True
        if not args:
            # No argument: open model/provider picker dialog.
            def on_pick(result: str | None):
                if result:
                    self._apply_model_switch(result)
            self.push_screen(
                ModelPickerScreen(self._model_name, getattr(self._agent, "provider", "") or ""),
                on_pick,
            )
            return True
        # Direct model name or provider:slug provided.
        self._apply_model_switch(args.strip())
        return True

    def _cmd_provider(self, args: str) -> bool:
        """Switch provider — opens picker if no arg, otherwise switches directly."""
        if self.is_working:
            self._show_system_msg("Cannot switch provider while agent is working.")
            return True
        if not args:
            def on_pick(result: str | None):
                if result:
                    self._apply_model_switch(f"provider:{result}")
            self.push_screen(
                ProviderPickerScreen(getattr(self._agent, "provider", "") or ""),
                on_pick,
            )
            return True
        self._apply_model_switch(f"provider:{args.strip()}")
        return True

    def _apply_model_switch(self, new_value: str):
        """Apply a model or provider switch.

        new_value is either:
          - a bare model name (e.g. "gpt-4o") → resolve through provider catalog
          - "provider:slug" (e.g. "provider:koala") → switch to custom provider
        """
        if new_value.startswith("provider:"):
            slug = new_value[len("provider:"):]
            self._switch_to_provider(slug)
            return

        # Bare model name — resolve through provider catalog.
        try:
            from hermes_cli.runtime_provider import resolve_runtime_provider
            runtime = resolve_runtime_provider(target_model=new_value)
            resolved = runtime.get("model") or new_value
            self._agent.model = resolved
            self._agent._cached_system_prompt = None
            self._model_name = resolved
            self._render_status()
            self._refresh_sidebar()
            self._show_system_msg(f"Model switched to: {resolved}")
        except Exception as e:
            self._show_system_msg(f"Failed to switch model: {e}")

    def _switch_to_provider(self, slug: str):
        """Switch to a custom provider by reading its config from config.yaml."""
        try:
            import yaml
            cfg_path = os.path.join(
                os.environ.get("HERMES_HOME", "") or os.path.expanduser("~/.hermes"),
                "config.yaml",
            )
            with open(cfg_path) as f:
                cfg = yaml.safe_load(f) or {}
        except Exception as e:
            self._show_system_msg(f"Failed to read config: {e}")
            return

        # Find the custom provider entry
        custom = cfg.get("custom_providers", [])
        target = None
        if isinstance(custom, list):
            for p in custom:
                if isinstance(p, dict) and p.get("name") == slug:
                    target = p
                    break
        elif isinstance(custom, dict):
            target = custom.get(slug)
            if target and isinstance(target, dict):
                target = {**target, "name": slug}

        if not target:
            self._show_system_msg(f"Provider '{slug}' not found in config.yaml")
            return

        new_model = target.get("model", "")
        new_base_url = target.get("base_url", "")
        new_api_key = target.get("api_key", "")

        # Use agent.switch_model() for a clean in-place swap
        try:
            self._agent.switch_model(
                new_model=new_model,
                new_provider=slug,
                api_key=new_api_key,
                base_url=new_base_url,
            )
        except Exception:
            # Fallback: direct attribute patch
            self._agent.model = new_model
            self._agent.provider = slug
            if new_base_url:
                self._agent.base_url = new_base_url
            if new_api_key:
                self._agent.api_key = new_api_key
            self._agent._cached_system_prompt = None

        self._model_name = new_model
        self._render_status()
        self._refresh_sidebar()
        self._show_system_msg(f"Switched to provider: {slug} · {new_model}")

    def _cmd_compress(self, args: str) -> bool:
        if not self._history or len(self._history) < 4:
            self._show_system_msg("Not enough conversation to compress (need at least 4 messages).")
            return True
        if self.is_working:
            self._show_system_msg("Cannot compress while agent is working.")
            return True
        if not getattr(self._agent, "compression_enabled", False):
            self._show_system_msg("Compression is disabled in config.")
            return True

        focus = args.strip()
        original_count = len(self._history)

        def do_compress():
            try:
                from agent.model_metadata import estimate_request_tokens_rough
                from agent.manual_compression_feedback import summarize_manual_compression
                approx_tokens = estimate_request_tokens_rough(self._history)
                self.call_from_thread(
                    self._show_system_msg,
                    f"Compressing {original_count} messages (~{approx_tokens:,} tokens)...",
                )
                _sys_prompt = getattr(self._agent, "_cached_system_prompt", "") or ""
                _tools = getattr(self._agent, "tools", None)
                compressed, _ = self._agent._compress_context(
                    list(self._history),
                    None,
                    approx_tokens=approx_tokens,
                    focus_topic=focus or None,
                    force=True,
                )
                self._history = compressed
                self._agent._cached_system_prompt = None
                new_tokens = estimate_request_tokens_rough(
                    compressed, system_prompt=_sys_prompt, tools=_tools,
                )
                summary = summarize_manual_compression(
                    self._history, compressed, approx_tokens, new_tokens,
                )
                self.call_from_thread(self._show_system_msg, f"Compressed: {summary}")
            except Exception as e:
                import traceback
                traceback.print_exc()
                self.call_from_thread(self._show_system_msg, f"Compression failed: {e}")

        threading.Thread(target=do_compress, daemon=True).start()
        return True

    def _cmd_retry(self) -> bool:
        if not self._history:
            self._show_system_msg("No messages to retry.")
            return True
        if self.is_working:
            self._show_system_msg("Cannot retry while agent is working.")
            return True

        # Walk backwards to find the last user message.
        last_user_idx = None
        for i in range(len(self._history) - 1, -1, -1):
            if isinstance(self._history[i], dict) and self._history[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            self._show_system_msg("No user message found to retry.")
            return True

        last_message = self._history[last_user_idx].get("content", "")
        self._history = self._history[:last_user_idx]
        self._show_system_msg(f"Retrying: \"{last_message[:60]}{'...' if len(last_message) > 60 else ''}\"")

        # Re-run with the extracted message.
        self._stream_buf = ""
        self._response_widget = None
        self._active_reasoning = None
        self._reasoning_summary = None
        self._tool_blocks.clear()
        self._tool_summary = None
        self._turn_start_time = time.time()
        self.is_working = True
        self._run_turn(last_message)
        return True

    def _cmd_undo(self) -> bool:
        if not self._history:
            self._show_system_msg("No messages to undo.")
            return True
        if self.is_working:
            self._show_system_msg("Cannot undo while agent is working.")
            return True

        # Walk backwards to find the last user message.
        last_user_idx = None
        for i in range(len(self._history) - 1, -1, -1):
            if isinstance(self._history[i], dict) and self._history[i].get("role") == "user":
                last_user_idx = i
                break

        if last_user_idx is None:
            self._show_system_msg("No user message found to undo.")
            return True

        removed_count = len(self._history) - last_user_idx
        removed_msg = self._history[last_user_idx].get("content", "")
        self._history = self._history[:last_user_idx]
        if self._agent:
            self._agent._cached_system_prompt = None
        self._show_system_msg(
            f"Undid {removed_count} message(s). Removed: \"{removed_msg[:60]}{'...' if len(removed_msg) > 60 else ''}\""
        )
        return True

    def _cmd_reload(self) -> bool:
        try:
            from hermes_cli.config import reload_env
            count = reload_env()
            self._show_system_msg(f"Reloaded .env ({count} var(s) updated)")
        except Exception as e:
            self._show_system_msg(f"Reload failed: {e}")
        return True

    def _cmd_reload_mcp(self) -> bool:
        try:
            mgr = (
                getattr(self._agent, "mcp_manager", None)
                or getattr(self._agent, "_mcp_manager", None)
            )
            if mgr and hasattr(mgr, "reload"):
                mgr.reload()
                self._show_system_msg("MCP servers reloaded.")
            else:
                self._show_system_msg("No MCP manager available to reload.")
        except Exception as e:
            self._show_system_msg(f"MCP reload failed: {e}")
        self._refresh_sidebar()
        return True

    def _cmd_reload_skills(self) -> bool:
        try:
            from agent.skill_commands import reload_skills
            result = reload_skills()
            added = result.get("added", [])
            removed = result.get("removed", [])
            total = result.get("total", 0)
            if not added and not removed:
                self._show_system_msg(f"No skill changes detected. {total} skill(s) available.")
            else:
                parts = []
                if added:
                    parts.append(f"+{len(added)} added")
                if removed:
                    parts.append(f"-{len(removed)} removed")
                self._show_system_msg(f"Skills reloaded: {', '.join(parts)}. {total} available.")
        except Exception as e:
            self._show_system_msg(f"Skill reload failed: {e}")
        return True

    def _cmd_yolo(self, args: str) -> bool:
        """Toggle YOLO mode — auto-approve all dangerous commands."""
        session_key = getattr(self._agent, "session_id", "") or ""
        if not session_key:
            self._show_system_msg("No active session for yolo mode.")
            return True

        disable = args.strip().lower() in ("off", "0", "no", "disable")

        try:
            from tools.approval import enable_session_yolo, disable_session_yolo
            if disable:
                disable_session_yolo(session_key)
                self._show_system_msg("🔓 YOLO mode OFF — approvals required again.")
            else:
                enable_session_yolo(session_key)
                self._show_system_msg("🔥 YOLO mode ON — all commands auto-approved!")
        except Exception as e:
            self._show_system_msg(f"YOLO toggle failed: {e}")
        return True

    # ── Agent turn ──

    def _run_turn(self, text: str):
        def run():
            # Register thread-local callbacks on the agent worker thread.
            # _callback_tls is per-thread, so we must set them here (not on the
            # UI thread) for approval/sudo prompts to find the callbacks.
            self._register_thread_local_callbacks()

            # ── Plan mode: prepend plan-only instruction ──
            actual_text = text
            if getattr(self._agent, "_tui_plan_mode", False):
                actual_text = (
                    "[PLAN MODE — do NOT execute any code, edit any file, or run any "
                    "mutating command. Only analyze, plan, and describe what you would do. "
                    "Write the plan as a markdown document.]\n\n" + text
                )

            try:
                # NOTE: Do NOT pass stream_callback to run_conversation().
                # _fire_stream_delta() fires BOTH stream_delta_callback AND
                # _stream_callback — if both point to _on_delta, every delta
                # would be processed twice.  We already set stream_delta_callback
                # in _wire_agent_callbacks(), which is sufficient to enable the
                # streaming path (_has_stream_consumers() checks it).
                result = self._agent.run_conversation(
                    actual_text,
                    conversation_history=list(self._history),
                    task_id=getattr(self._agent, "session_id", None),
                )
                final_text = ""
                if isinstance(result, dict):
                    if isinstance(result.get("messages"), list):
                        self._history = result["messages"]
                    final_text = result.get("final_response", "")
                    if result.get("error"):
                        final_text = result.get("final_response", str(result["error"]))
                self.call_from_thread(self._on_turn_complete, final_text)
            except Exception as e:
                self.call_from_thread(self._show_error, str(e))

        threading.Thread(target=run, daemon=True).start()

    # ── Timeline callbacks (run on UI thread via call_from_thread) ──

    def _ensure_response_widget(self) -> Static:
        """Get or create the single response widget for this turn."""
        if self._response_widget is None:
            self._response_widget = Static(
                Text(f"{_SPINNER[0]} thinking…", style=C_DIM),
                classes="msg-assistant streaming",
            )
            self.query_one("#chat-area").mount(self._response_widget)
        return self._response_widget

    def _refresh_streaming_display(self):
        """Update the response widget with current streamed text + spinner."""
        w = self._response_widget
        if w is None:
            return
        display = _strip_inline_tool_xml(self._stream_buf)
        sp = _SPINNER[self._spinner_idx % len(_SPINNER)]
        if display:
            w.update(Group(
                Markdown(display),
                Text(f"{sp} …", style=C_DIM),
            ))
        elif self._tool_blocks:
            # No text yet but tools are running — show processing.
            w.update(Text(f"{sp} processing…", style=C_DIM))
        else:
            w.update(Text(f"{sp} thinking…", style=C_DIM))

    def _on_delta(self, text):
        if text is None:
            # Flush boundary — nothing to do, the buffer persists.
            return

        # Real text means any reasoning phase is over.
        self._finalize_reasoning()

        self._stream_buf += text
        self._ensure_response_widget()
        self._refresh_streaming_display()
        self._scroll_to_bottom()

    def _on_reasoning(self, text):
        if not text:
            return
        if self._active_reasoning is None:
            self._active_reasoning = ReasoningBlock()
            self.query_one("#chat-area").mount(self._active_reasoning)
        self._active_reasoning.append(text)
        self._scroll_to_bottom()

    def _finalize_reasoning(self):
        if self._active_reasoning is not None:
            # Collect the reasoning text before removing the block.
            text = self._active_reasoning._buffer.strip()
            self._active_reasoning.remove()
            self._active_reasoning = None
            if text:
                # Ensure ReasoningSummaryBlock exists, mounted before response widget.
                if self._reasoning_summary is None:
                    self._reasoning_summary = ReasoningSummaryBlock()
                    chat = self.query_one("#chat-area")
                    if self._tool_summary is not None:
                        chat.mount(self._reasoning_summary, before=self._tool_summary)
                    elif self._response_widget is not None:
                        chat.mount(self._reasoning_summary, before=self._response_widget)
                    else:
                        chat.mount(self._reasoning_summary)
                self._reasoning_summary.add_segment(text)

    def _detect_skill_usage(self, tool_name: str, args):
        """Detect when a skill is loaded and record it for the sidebar."""
        if tool_name in ("skill_view", "skill_manage"):
            if isinstance(args, dict):
                skill_name = args.get("name", "")
            elif isinstance(args, str):
                try:
                    import json as _json
                    skill_name = _json.loads(args).get("name", "")
                except Exception:
                    skill_name = ""
            else:
                skill_name = ""
            if skill_name and skill_name not in self._turn_skills:
                self._turn_skills.append(skill_name)

    def _update_tasks_from_todo(self, result):
        """Parse the result of a 'todo' tool call and extract the task list."""
        if not isinstance(result, str) or not result.strip():
            return
        try:
            import json as _json
            data = _json.loads(result)
        except Exception:
            return
        if isinstance(data, dict) and isinstance(data.get("todos"), list):
            self._tasks = data["todos"]
        elif isinstance(data, list):
            self._tasks = data

    def _on_tool_start(self, tc_id, name, args):
        # A tool interrupts any open reasoning.
        self._finalize_reasoning()

        # Track tool/skill usage for sidebar.
        name = str(name) if name else "?"
        if name and name not in self._turn_tools:
            self._turn_tools.append(name)
        self._detect_skill_usage(name, args)

        # ── Snapshot files before write_file / patch modifies them (for diff) ──
        if name in ("write_file", "patch") and isinstance(args, dict):
            file_path = args.get("path", "")
            if file_path:
                self._snapshot_file(file_path)

        # Ensure ToolSummaryBlock exists (mounted before the response widget).
        if self._tool_summary is None:
            self._tool_summary = ToolSummaryBlock()
            chat = self.query_one("#chat-area")
            # Mount before response widget so summary stays above the answer.
            if self._response_widget is not None:
                chat.mount(self._tool_summary, before=self._response_widget)
            else:
                chat.mount(self._tool_summary)

        # Mount tool block in chat area (still shown while running).
        block = ToolCallBlock(name, _tool_preview(name, args))
        self._tool_blocks[tc_id] = (block, time.time())
        self.query_one("#chat-area").mount(block)

        # Ensure the response widget exists (shows "processing…" during tools).
        self._ensure_response_widget()
        self._refresh_streaming_display()
        self._refresh_sidebar()
        self._scroll_to_bottom()

    def _on_tool_complete(self, tc_id, name, args, result):
        entry = self._tool_blocks.pop(tc_id, None)
        if entry is None:
            return
        block, started = entry
        ok = _tool_succeeded(result)
        duration = _fmt_duration(time.time() - started)
        icon = _TOOL_ICONS.get(str(name), "⚙")
        preview = _tool_preview(str(name), args)

        # Remove the individual ToolCallBlock from chat area.
        try:
            block.remove()
        except Exception:
            pass

        # Record completion in the summary block.
        if self._tool_summary is not None:
            self._tool_summary.add_completed(
                icon=icon,
                name=str(name),
                duration=duration,
                preview=preview if not ok else "",  # only show preview for failed tools
                ok=ok,
            )

        # ── Render inline diff for file modification tools ──
        if ok and str(name) in ("write_file", "patch") and isinstance(args, dict):
            file_path = args.get("path", "")
            if file_path:
                self._render_file_diff(file_path)

        # Extract plan tasks from todo tool results.
        if str(name) == "todo":
            self._update_tasks_from_todo(result)

        self._refresh_sidebar()
        self._scroll_to_bottom()

    def _on_tool_progress(self, event_type, name, preview, args, kwargs):
        self._scroll_to_bottom()

    def _on_status(self, kind: str, text: str | None):
        self._status_kind = kind
        self._status_text = text
        self._render_status()

        # ── Compression / context events → notify bar above input ──
        if text and any(kw in text.lower() for kw in ("compact", "compress", "compressi")):
            self._show_notify(f"🗜️ {text}")

    _NOTIFY_FADE_DELAY = 4.0  # seconds before auto-hide
    _notify_fade_timer = None

    def _show_notify(self, text: str):
        """Show a notification in the bar above the input; auto-fades after a few seconds."""
        try:
            nb = self.query_one("#notify-bar", Static)
            nb.update(Text(text, style=C_DIM))
            nb.add_class("active")
        except Exception:
            pass
        # Reset auto-hide timer
        if self._notify_fade_timer is not None:
            self._notify_fade_timer.stop()
        self._notify_fade_timer = self.set_timer(
            self._NOTIFY_FADE_DELAY, self._hide_notify
        )

    def _hide_notify(self):
        """Hide the notify bar."""
        try:
            nb = self.query_one("#notify-bar", Static)
            nb.remove_class("active")
            nb.update("")
        except Exception:
            pass
        self._notify_fade_timer = None

    def _on_clarify(self, question, choices):
        """Show clarify question in chat history (the modal already handled the answer)."""
        hint = question
        if choices:
            hint += " [" + " / ".join(str(c) for c in choices) + "]"
        self.query_one("#chat-area").mount(
            Static(Text(f"❓ {hint}", style=C_DIM), classes="system-msg")
        )
        self._scroll_to_bottom()

    def _on_turn_complete(self, text: str):
        try:
            self._finalize_reasoning()

            # Determine what to display as the final response.
            # Priority: final_response > streamed text > "(no response)"
            display = (text or "").strip()
            display = _strip_inline_tool_xml(display)
            if not display:
                display = _strip_inline_tool_xml(self._stream_buf)

            w = self._ensure_response_widget()
            w.remove_class("streaming")
            if display:
                code_blocks = _extract_code_blocks(display)
                if code_blocks:
                    # Replace the single response widget with multiple widgets
                    # (CopyButton + Markdown code block per code, Static for text)
                    # mounted in the chat area.
                    chat = self.query_one("#chat-area")
                    parts = _split_markdown_by_codeblocks(display)
                    # Remove the streaming response widget first
                    w.remove()
                    self._response_widget = None
                    for kind, content in parts:
                        if kind == "markdown":
                            if content.strip():
                                chat.mount(Static(
                                    Markdown(content.strip()),
                                    classes="msg-assistant",
                                ))
                        elif kind == "code":
                            lang, code = content
                            chat.mount(CopyButton(lang, code))
                            chat.mount(Static(
                                Markdown(f"```{lang}\n{code}\n```"),
                                classes="msg-assistant",
                            ))
                else:
                    w.update(Markdown(display))
            else:
                w.update(Text("(no response)", style=C_DIM))
        except Exception as e:
            import traceback
            print(f"[TUI] _on_turn_complete error: {e}", file=sys.stderr, flush=True)
            traceback.print_exc(file=sys.stderr)
            try:
                self.query_one("#chat-area").mount(
                    Static(Text(text or "(error)", style=C_TEXT), classes="msg-assistant")
                )
            except Exception:
                pass

        # Close any tool rows still marked running (defensive).
        for tc_id, (block, started) in list(self._tool_blocks.items()):
            duration = _fmt_duration(time.time() - started)
            icon = _TOOL_ICONS.get(getattr(block, "_name", "?"), "⚙")
            name = getattr(block, "_name", "?")
            preview = getattr(block, "_preview", "")
            try:
                block.remove()
            except Exception:
                pass
            if self._tool_summary is not None:
                self._tool_summary.add_completed(icon=icon, name=name, duration=duration, preview=preview)
        self._tool_blocks.clear()

        # Set total duration on summary and remove if no tools were used.
        if self._tool_summary is not None:
            if self._turn_start_time > 0:
                self._tool_summary.set_total_duration(time.time() - self._turn_start_time)
            if not self._tool_summary._completed:
                try:
                    self._tool_summary.remove()
                except Exception:
                    pass
                self._tool_summary = None

        # Remove reasoning summary if no segments were collected.
        if self._reasoning_summary is not None:
            if not self._reasoning_summary._segments:
                try:
                    self._reasoning_summary.remove()
                except Exception:
                    pass
                self._reasoning_summary = None

        self.is_working = False
        self._status_kind = ""
        self._status_text = None
        self._refresh_sidebar()
        self._scroll_to_bottom()
        self.query_one("#prompt-input").focus()

    def _show_error(self, msg: str):
        self._finalize_reasoning()
        if self._response_widget is not None:
            if self._stream_buf.strip():
                self._response_widget.remove_class("streaming")
                display = _strip_inline_tool_xml(self._stream_buf)
                code_blocks = _extract_code_blocks(display)
                if code_blocks:
                    chat = self.query_one("#chat-area")
                    parts = _split_markdown_by_codeblocks(display)
                    self._response_widget.remove()
                    self._response_widget = None
                    for kind, content in parts:
                        if kind == "markdown":
                            if content.strip():
                                chat.mount(Static(
                                    Markdown(content.strip()),
                                    classes="msg-assistant",
                                ))
                        elif kind == "code":
                            lang, code = content
                            chat.mount(CopyButton(lang, code))
                            chat.mount(Static(
                                Markdown(f"```{lang}\n{code}\n```"),
                                classes="msg-assistant",
                            ))
                else:
                    self._response_widget.update(Markdown(display))
            else:
                self._response_widget.remove()
            self._response_widget = None
        # Remove any running tool blocks and summary from this turn.
        for tc_id, (block, started) in list(self._tool_blocks.items()):
            try:
                block.remove()
            except Exception:
                pass
        self._tool_blocks.clear()
        if self._tool_summary is not None:
            try:
                self._tool_summary.remove()
            except Exception:
                pass
            self._tool_summary = None
        self.query_one("#chat-area").mount(
            Static(Text(f"⚠ {msg}", style=C_RED), classes="error-msg")
        )
        self.is_working = False
        self._render_status(error=True)
        self.query_one("#prompt-input").focus()

    # ── Sidebar / status refresh ──

    def _agent_tokens(self) -> int:
        return int(getattr(self._agent, "session_total_tokens", 0) or 0)

    def _agent_cost(self) -> float:
        return float(getattr(self._agent, "session_estimated_cost_usd", 0.0) or 0.0)

    def _agent_compressions(self) -> int:
        """Return compression count from both in-memory compressor and
        SessionDB parent chain, taking the maximum so historical
        compressions are never lost."""
        mem_count = 0
        compressor = getattr(self._agent, "context_compressor", None)
        if compressor is not None:
            mem_count = int(getattr(compressor, "compression_count", 0) or 0)

        db_count = 0
        session_id = getattr(self._agent, "session_id", None)
        # Avoid re-querying the DB for the same session.
        if session_id and session_id != getattr(self, "_cc_last_sid", None):
            self._cc_last_sid = session_id
            session_db = (
                getattr(self._agent, "_session_db", None)
                or getattr(self._agent, "session_db", None)
            )
            if session_db:
                try:
                    chain = session_db._session_lineage_root_to_tip(session_id)
                    self._cc_last_db = 0
                    for sid in chain[:-1]:
                        ancestor = session_db.get_session(sid)
                        if ancestor and ancestor.get("end_reason") == "compression":
                            self._cc_last_db += 1
                except Exception:
                    self._cc_last_db = 0
        db_count = getattr(self, "_cc_last_db", 0)

        count = max(mem_count, db_count)
        # Sync back to the compressor so it won't diverge on refresh.
        if compressor is not None and count > mem_count:
            compressor.compression_count = count
        return count

    def _agent_mcp(self) -> list[tuple[str, bool]]:
        """Best-effort MCP server discovery — degrades gracefully."""
        try:
            mgr = (
                getattr(self._agent, "mcp_manager", None)
                or getattr(self._agent, "_mcp_manager", None)
            )
            servers = getattr(mgr, "servers", None) if mgr else None
            if isinstance(servers, dict):
                return [(str(name), True) for name in servers.keys()]
            if isinstance(servers, (list, tuple)):
                return [(str(s), True) for s in servers]
        except Exception:
            pass
        return []

    def _refresh_sidebar(self):
        if self._agent is None:
            return
        sb = self.query_one(Sidebar)
        tokens = self._agent_tokens()
        sb.set_context(tokens, None, 0.0, self._agent_compressions(), self._context_len)
        sb.set_tasks(self._tasks)
        sb.set_usage(self._turn_tools, self._turn_skills)
        sb.set_mcp(self._agent_mcp())
        sb.set_footer(self._short_cwd(), self._version or self._hermes_version())

    def _render_status(self, error: bool = False):
        bar = self.query_one("#status-bar", Static)
        bar_w = bar.content_size.width
        if bar_w <= 0:
            bar_w = max(self.size.width - 4, 1)

        if self.is_working and not error:
            left = Text()
            left.append_text(self._work_bar())
            left.append("  ", style=C_DIM)
            if self._status_kind:
                left.append(self._status_kind, style=C_DIM)
                if self._status_text:
                    left.append(f" {self._status_text}", style=C_DIM)
                left.append("  ", style=C_DIM)
            left.append("esc", style=C_TEXT)
            left.append(" interrupt", style=C_DIM)

            right = Text()
            right.append(
                self._fmt_tokens(self._agent_tokens() if self._agent else 0),
                style=C_DIM,
            )
            gap = max(bar_w - left.cell_len - right.cell_len, 1)
            bar.update(Text.assemble(left, " " * gap, right))
            return

        model = self._model_name or "…"
        left = Text()
        if error:
            left.append("⚠ error", style=C_RED)
        elif self._agent is None:
            left.append("● connecting", style=C_DIM)
        else:
            left.append("● ", style=C_GREEN)
            left.append(self._mode, style=C_TEXT)
        left.append("  ·  ", style=C_DIM)
        left.append(model, style=C_BLUE)

        right = Text()
        right.append(self._fmt_tokens(self._agent_tokens() if self._agent else 0), style=C_DIM)
        right.append("   ", style=C_DIM)
        right.append("⇧⇥", style=C_TEXT)
        right.append(" mode", style=C_DIM)

        gap = max(bar_w - left.cell_len - right.cell_len, 1)
        line = Text.assemble(left, " " * gap, right)
        bar.update(line)

    def _snapshot_file(self, file_path: str):
        """Read and cache the current file contents before a tool modifies it."""
        try:
            p = Path(file_path)
            if not p.is_absolute():
                p = Path(os.getcwd()) / p
            if p.exists() and p.is_file():
                self._file_snapshots[str(p)] = p.read_text(errors="replace").splitlines(keepends=True)
            else:
                self._file_snapshots[str(p)] = []  # new file
        except Exception:
            pass

    def _render_file_diff(self, file_path: str):
        """Generate and display a unified diff for the modified file."""
        try:
            p = Path(file_path)
            if not p.is_absolute():
                p = Path(os.getcwd()) / p
            key = str(p)

            old_lines = self._file_snapshots.pop(key, None)
            if old_lines is None:
                return  # no snapshot — nothing to compare

            if p.exists() and p.is_file():
                new_lines = p.read_text(errors="replace").splitlines(keepends=True)
            else:
                new_lines = []  # file was deleted

            # Skip if unchanged
            if old_lines == new_lines:
                return

            diff_lines = list(difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{p.name}", tofile=f"b/{p.name}",
                lineterm="", n=3,
            ))

            if not diff_lines:
                return

            # Truncate very long diffs to avoid flooding the chat area
            max_diff_lines = 80
            truncated = len(diff_lines) > max_diff_lines + 5
            display_lines = diff_lines[:max_diff_lines]
            if truncated:
                display_lines.append(f"... ({len(diff_lines) - max_diff_lines} more lines)\n")

            diff_text = "".join(display_lines)
            chat = self.query_one("#chat-area")
            chat.mount(DiffBlock(str(p), diff_text))
            self._scroll_to_bottom()
        except Exception:
            pass

    def _scroll_to_bottom(self):
        self.query_one("#chat-area").scroll_end(animate=False)

    def action_clear_chat(self):
        self._history = []
        self._stream_buf = ""
        self._response_widget = None
        self._active_reasoning = None
        self._reasoning_summary = None
        self._tool_blocks.clear()
        self._tool_summary = None
        self._turn_start_time = 0.0
        self._file_snapshots.clear()
        msgs = self.query_one("#chat-area")
        msgs.remove_children()
        msgs.mount(Static(
            Markdown("**Chat cleared.** Send a message to start again."),
            classes="welcome",
        ))
        self._render_status()

    def action_new_session(self):
        """Ctrl+N — start a fresh session (clear + new session ID)."""
        if self.is_working:
            self._show_system_msg("Cannot start new session while agent is working.")
            return
        self.action_clear_chat()
        if self._agent:
            try:
                from hermes_state import SessionDB
                session_db = getattr(self._agent, "_session_db", None) or getattr(self._agent, "session_db", None) or SessionDB()
                new_id = session_db.create_session(source="textual-tui")
                self._agent.session_id = new_id
                self._agent._cached_system_prompt = None
                self._show_system_msg(f"New session: {new_id}")
            except Exception as e:
                self._show_system_msg(f"New session failed: {e}")

    def action_retry_last(self):
        """Ctrl+R — retry last user message."""
        self._cmd_retry()

    def action_undo_last(self):
        """Ctrl+Z — undo last exchange."""
        self._cmd_undo()

    def action_compress_context(self):
        """Ctrl+E — compress conversation context."""
        self._cmd_compress("")

    def action_pick_model(self):
        """Ctrl+M — open model picker."""
        if self.is_working:
            self._show_system_msg("Cannot switch model while agent is working.")
            return
        def on_pick(result: str | None):
            if result:
                self._apply_model_switch(result)
        self.push_screen(
            ModelPickerScreen(self._model_name, getattr(self._agent, "provider", "") or ""),
            on_pick,
        )

    def action_pick_provider(self):
        """Ctrl+P — open provider picker."""
        if self.is_working:
            self._show_system_msg("Cannot switch provider while agent is working.")
            return
        def on_pick(result: str | None):
            if result:
                self._apply_model_switch(f"provider:{result}")
        self.push_screen(
            ProviderPickerScreen(getattr(self._agent, "provider", "") or ""),
            on_pick,
        )

    def action_clear_input(self):
        """Ctrl+K — clear the input area."""
        prompt = self.query_one("#prompt-input", PromptArea)
        prompt.text = ""
        prompt.cursor_location = (0, 0)

    def action_cycle_permission(self):
        """Shift+Tab — cycle through permission modes: default → plan → auto → default."""
        idx = self._PERMISSION_MODES.index(self._permission_mode)
        self._permission_mode = self._PERMISSION_MODES[(idx + 1) % len(self._PERMISSION_MODES)]
        mode = self._permission_mode
        session_key = getattr(self._agent, "session_id", "") or ""

        if mode == "plan":
            # Disable YOLO so approvals still fire
            if session_key:
                try:
                    from tools.approval import disable_session_yolo
                    disable_session_yolo(session_key)
                except Exception:
                    pass
            # Set plan mode flag so _run_turn injects plan-only instruction
            if self._agent:
                self._agent._tui_plan_mode = True
            self._show_system_msg("🔒 Plan mode — agent will only plan, not execute")
        elif mode == "auto":
            # Enable YOLO for auto mode
            if session_key:
                try:
                    from tools.approval import enable_session_yolo
                    enable_session_yolo(session_key)
                except Exception:
                    pass
            if self._agent:
                self._agent._tui_plan_mode = False
            self._show_system_msg("🔥 Auto mode — all commands auto-approved")
        else:
            # Disable YOLO for default mode
            if session_key:
                try:
                    from tools.approval import disable_session_yolo
                    disable_session_yolo(session_key)
                except Exception:
                    pass
            if self._agent:
                self._agent._tui_plan_mode = False
            self._show_system_msg("✋ Default mode — approvals required")

        self._render_status()

    def action_open_sessions(self):
        """Ctrl+O — open session browser."""
        self._open_session_picker()

    def _open_session_picker(self):
        """Show a modal with recent sessions; on selection, resume that session."""
        if self.is_working:
            self._show_system_msg("Cannot switch sessions while agent is working.")
            return
        current_id = getattr(self._agent, "session_id", "") or ""

        def on_pick(session_id: str | None):
            if not session_id or session_id == current_id:
                return
            self._resume_session(session_id)

        self.push_screen(SessionPickerScreen(current_id), on_pick)

    def _resume_session(self, session_id: str):
        """Resume a different session by ID — clear chat and reload."""
        if not self._agent:
            return
        try:
            self.action_clear_chat()
            self._agent.session_id = session_id
            self._agent._tui_is_resume = True
            self._agent._cached_system_prompt = None
            self._load_resumed_session()
            self._restore_session_state()
            self._refresh_sidebar()
            self._render_status()
        except Exception as e:
            self._show_system_msg(f"Failed to resume session: {e}")

    def on_resize(self, event) -> None:
        self._render_status(error=False)

    def action_interrupt(self) -> None:
        """esc — ask the running agent to stop its current tool-calling loop."""
        if self.is_working and self._agent is not None:
            try:
                self._agent.interrupt()
            except Exception:
                pass

    def on_click(self, event: events.Click) -> None:
        """Toggle tool summary / reasoning summary block on click."""
        widget = event.widget
        while widget is not None:
            if isinstance(widget, (ToolSummaryBlock, ReasoningSummaryBlock)):
                widget.toggle()
                self._scroll_to_bottom()
                return
            widget = getattr(widget, "parent", None)

    def watch_is_working(self, working: bool):
        if working:
            self._anim_phase = 0
            self._spinner_idx = 0
            if self._anim_timer is None:
                self._anim_timer = self.set_interval(self._BAR_INTERVAL, self._tick_anim)
        else:
            if self._anim_timer is not None:
                self._anim_timer.stop()
                self._anim_timer = None
        self._render_status()


if __name__ == "__main__":
    app = HermesApp()
    app.run()
