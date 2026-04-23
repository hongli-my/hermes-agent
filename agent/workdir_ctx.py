"""
Per-agent working-directory context for in-process tool calls.

Problem
-------
Hermes' terminal / code_execution / file tools historically read the
working directory from the process-global ``os.environ["TERMINAL_CWD"]``
variable.  That worked fine for the CLI (one agent per process), but the
workflow engine runs multiple ``AIAgent`` instances concurrently inside
the same process (``iteration.max_concurrent > 1`` or ``parallel``
branches), and they race on the shared env var — one thread's tool call
would see another thread's worktree.

Solution
--------
We keep ``TERMINAL_CWD`` as the legacy source of truth for:

* child processes spawned by terminal_tool / code_execution_tool
  (they still need an env var since a fresh process can't read our
  Python ContextVar), and
* external integrations / CLI users that set it themselves.

But for **in-process** reads from the tools layer we introduce a
``contextvars.ContextVar`` that is:

* set per-agent by ``AIAgent._apply_working_dir`` before each turn,
* naturally inherited by ``ThreadPoolExecutor`` workers and asyncio
  tasks (Python 3.7+ semantics), so each concurrent agent sees its own
  value without any locking.

Read path:
    ``get_terminal_cwd()`` → ContextVar → fallback to
    ``os.environ["TERMINAL_CWD"]`` → fallback to caller's default.

Write path:
    ``set_terminal_cwd(value)`` → returns a token; caller can
    ``reset_terminal_cwd(token)`` to restore prior value.  Also syncs
    ``os.environ`` so subprocesses see the same value (best-effort —
    the env is still globally shared and may be overwritten by peers,
    but it's only used as a fallback).
"""
from __future__ import annotations

import contextvars
import os
from typing import Optional

# Sentinel distinguishing "not set in this context" from "set to None".
_UNSET = object()

_TERMINAL_CWD_CTX: contextvars.ContextVar = contextvars.ContextVar(
    "hermes_terminal_cwd", default=_UNSET,
)


def get_terminal_cwd(default: Optional[str] = None) -> Optional[str]:
    """Return the current working dir for the active agent context.

    Resolution order:
      1. ContextVar (set by the owning AIAgent for its turn).
      2. ``os.environ["TERMINAL_CWD"]`` (CLI / external integrations).
      3. ``default`` (caller-supplied fallback, usually ``os.getcwd()``).

    Returns ``None`` when the caller passes no default and nothing is
    configured, so that downstream callers who want ``os.getcwd()``
    can do ``get_terminal_cwd() or os.getcwd()``.
    """
    val = _TERMINAL_CWD_CTX.get()
    if val is not _UNSET and val:
        return val  # type: ignore[return-value]
    env_val = os.environ.get("TERMINAL_CWD")
    if env_val:
        return env_val
    return default


def set_terminal_cwd(value: Optional[str],
                     sync_env: bool = True) -> contextvars.Token:
    """Bind *value* to the terminal-cwd ContextVar for the current context.

    When ``sync_env`` is True (default) we also update
    ``os.environ["TERMINAL_CWD"]`` so that child processes launched via
    ``subprocess.Popen`` inherit the right cwd.  The env var is still
    globally shared between threads — use ``Popen(..., env=...)`` with
    an explicit copy when strict isolation is required for a spawn.

    Returns a ``contextvars.Token`` that can be passed to
    ``reset_terminal_cwd`` to restore the previous value.
    """
    token = _TERMINAL_CWD_CTX.set(value if value else _UNSET)
    if sync_env:
        try:
            if value:
                os.environ["TERMINAL_CWD"] = value
            elif "TERMINAL_CWD" in os.environ:
                # Only clear if we set it; never clobber caller's value
                # when called with None from code that didn't own it.
                # We leave the decision to callers via sync_env=False if
                # they don't want env touched.
                del os.environ["TERMINAL_CWD"]
        except Exception:
            pass
    return token


def reset_terminal_cwd(token: contextvars.Token) -> None:
    """Restore the ContextVar to its prior value (paired with ``set_terminal_cwd``)."""
    try:
        _TERMINAL_CWD_CTX.reset(token)
    except Exception:
        pass


def has_terminal_cwd() -> bool:
    """Return True when an in-process caller has bound a cwd."""
    return _TERMINAL_CWD_CTX.get() is not _UNSET
