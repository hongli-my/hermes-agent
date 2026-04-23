"""
Hermes Workflow CLI subcommand.

Usage:
  hermes workflow list                    List all workflows
  hermes workflow show <id>               Show workflow definition
  hermes workflow create <file.yaml>      Create workflow from YAML file
  hermes workflow delete <id>             Delete a workflow
  hermes workflow run <id> [key=val...]   Execute a workflow (with optional inputs)
  hermes workflow runs <id>               List runs for a workflow
  hermes workflow run-file <yaml_path>    Execute a YAML workflow file directly
  hermes workflow validate <file.yaml>    Validate a YAML workflow file
"""

import json
import os
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

def handle_workflow_command(args):
    """Dispatch workflow subcommands."""
    if not args:
        _print_help()
        return

    action = args[0]
    rest = args[1:]

    if action == "list":
        _cmd_list()
    elif action == "show":
        _cmd_show(rest)
    elif action == "create":
        _cmd_create(rest)
    elif action == "delete":
        _cmd_delete(rest)
    elif action == "run":
        _cmd_run(rest)
    elif action == "runs":
        _cmd_runs(rest)
    elif action == "run-file":
        _cmd_run_file(rest)
    elif action == "validate":
        _cmd_validate(rest)
    else:
        print(f"Unknown action: {action}")
        _print_help()


def _print_help():
    print("Usage: hermes workflow <action> [args]")
    print()
    print("Actions:")
    print("  list                       List all workflows")
    print("  show <id>                  Show workflow definition")
    print("  create <file.yaml>         Create workflow from YAML file")
    print("  delete <id>                Delete a workflow")
    print("  run <id> [key=val...]      Execute a workflow (with optional inputs)")
    print("  runs <id>                  List runs for a workflow")
    print("  run-file <yaml_path>       Execute a YAML workflow file directly")
    print("  validate <file.yaml>       Validate a YAML workflow file")


def _cmd_list():
    from workflow import list_workflows
    workflows = list_workflows()
    if not workflows:
        print("No workflows found.")
        return
    print(f"{'ID':<20} {'Name':<25} {'Nodes':<7} {'Description'}")
    print("-" * 80)
    for wf in workflows:
        desc = (wf.get("description") or "")[:30]
        print(f"{wf['id']:<20} {wf.get('name', '-'):<25} {wf.get('nodes_count', 0):<7} {desc}")


def _cmd_show(rest):
    if not rest:
        print("Usage: hermes workflow show <workflow_id>")
        return
    from workflow import get_workflow
    wf = get_workflow(rest[0])
    if not wf:
        print(f"Workflow '{rest[0]}' not found.")
        return
    # Pretty print, hide internal fields
    display = {k: v for k, v in wf.items() if not k.startswith("_")}
    print(json.dumps(display, indent=2, ensure_ascii=False))


def _cmd_create(rest):
    if not rest:
        print("Usage: hermes workflow create <file.yaml>")
        return
    filepath = Path(rest[0])
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    if filepath.suffix not in (".yaml", ".yml"):
        print(f"Only YAML files are supported (.yaml/.yml), got: {filepath.suffix}")
        return

    try:
        from workflow.store import create_workflow_from_yaml
        wf = create_workflow_from_yaml(str(filepath))
        print(f"Workflow created: {wf['id']} ({wf.get('name', '-')})")
    except Exception as e:
        print(f"Failed to create workflow: {e}")


def _cmd_delete(rest):
    if not rest:
        print("Usage: hermes workflow delete <workflow_id>")
        return
    from workflow import delete_workflow
    if delete_workflow(rest[0]):
        print(f"Workflow '{rest[0]}' deleted.")
    else:
        print(f"Workflow '{rest[0]}' not found.")


def _cmd_run(rest):
    if not rest:
        print("Usage: hermes workflow run <workflow_id> [key=val...]")
        return
    wf_id = rest[0]
    inputs = _parse_inputs(rest[1:])

    from workflow import get_workflow, run_workflow
    wf = get_workflow(wf_id)
    if not wf:
        print(f"Workflow '{wf_id}' not found.")
        return

    session_db = _get_session_db(wf_id)
    print(f"Running workflow '{wf.get('name', wf_id)}' ...")
    print()
    console_handler = _install_console_log_handler()
    monitor = _HeartbeatMonitor.start_default()
    try:
        result = run_workflow(wf, inputs=inputs, session_db=session_db,
                              progress_callback=_live_progress_callback)
    finally:
        monitor.stop()
        _remove_console_log_handler(console_handler)

    _print_run_result(result)


def _cmd_run_file(rest):
    """Execute a YAML workflow file directly (no registration needed)."""
    if not rest:
        print("Usage: hermes workflow run-file <yaml_path> [key=val...]")
        return
    filepath = Path(rest[0])
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    inputs = _parse_inputs(rest[1:])

    try:
        from workflow.schema import load_yaml_file
        from workflow.engine import run_workflow
        wf = load_yaml_file(str(filepath))
        wf_id = wf.get("id") or filepath.stem
        session_db = _get_session_db(wf_id)
        print(f"Running workflow '{wf.get('name', filepath.stem)}' from file ...")
        print()
        console_handler = _install_console_log_handler()
        monitor = _HeartbeatMonitor.start_default()
        try:
            result = run_workflow(wf, inputs=inputs, session_db=session_db,
                                  progress_callback=_live_progress_callback)
        finally:
            monitor.stop()
            _remove_console_log_handler(console_handler)
        _print_run_result(result)
    except Exception as e:
        print(f"Failed to run workflow: {e}")
        import traceback
        traceback.print_exc()


def _cmd_runs(rest):
    if not rest:
        print("Usage: hermes workflow runs <workflow_id>")
        return
    from workflow import list_workflow_runs
    runs = list_workflow_runs(rest[0])
    if not runs:
        print("No runs found.")
        return
    print(f"{'Run ID':<14} {'Status':<8} {'Duration':<10} {'Started'}")
    print("-" * 55)
    for run in runs:
        dur = f"{run.get('duration_seconds', 0):.1f}s"
        print(f"{run['run_id']:<14} {run.get('status', '-'):<8} {dur:<10} {run.get('started_at', '-')}")




# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _get_session_db(session_id_prefix: str = None):
    """Create a thread-safe SessionDB wrapper for workflow session tracking.

    Workflow execution is single-threaded at the step level, but iteration
    and parallel nodes can spawn concurrent branches that each need their
    own session DB connection to avoid sqlite3 concurrent-write errors.

    Each SessionDB instance holds its own sqlite3 connection (check_same_thread=False)
    and uses the instance-level lock for all writes.  The retry-with-jitter
    strategy in SessionDB handles SQLite BUSY from concurrent processes, but
    intra-process concurrent writes must be serialized by the lock.

    We return a wrapped object that uses a lock + separate connection per
    branch to ensure thread-safety.
    """
    try:
        from hermes_state import SessionDB
        import threading, uuid as _uuid

        run_id = _uuid.uuid4().hex[:12]
        wf_sid = f"wf:{session_id_prefix or 'run'}:{run_id}"

        # Create the primary session DB
        db = SessionDB(session_id=wf_sid)

        # Wrap with a lock so concurrent branches serialize their writes
        return _ThreadSafeSessionDB(db, session_id=wf_sid)

    except Exception as e:
        import logging
        logging.warning("Could not create SessionDB for workflow: %s", e)
        return None


class _ThreadSafeSessionDB:
    """Thread-safe wrapper around SessionDB.

    Uses a per-instance lock to serialize all writes.  Each concurrent
    branch in an iteration/parallel node gets its own wrapper that shares
    the underlying SessionDB instance but serializes access via the lock.
    """

    def __init__(self, db: "SessionDB", session_id: str):
        self._db = db
        self._sid = session_id
        self._lock = threading.Lock()

    def create_session(self, **kwargs):
        with self._lock:
            return self._db.create_session(
                session_id=self._sid,
                source=kwargs.get("source", "workflow"),
                model=kwargs.get("model"),
                model_config=kwargs.get("model_config", {}),
                parent_session_id=kwargs.get("parent_session_id"),
            )

    def append_message(self, session_id=None, **kwargs):
        # Use the wrapper's session_id; thread-safe via lock
        with self._lock:
            return self._db.append_message(
                session_id=self._sid,
                role=kwargs.get("role"),
                content=kwargs.get("content"),
                tool_name=kwargs.get("tool_name"),
                tool_calls=kwargs.get("tool_calls"),
                tool_call_id=kwargs.get("tool_call_id"),
                token_count=kwargs.get("token_count"),
                finish_reason=kwargs.get("finish_reason"),
                reasoning=kwargs.get("reasoning"),
                metadata=kwargs.get("metadata"),
            )

    def flush(self, session_id=None):
        with self._lock:
            return self._db.flush(session_id=self._sid)

    def get_session(self, session_id=None):
        return self._db.get_session(session_id=self._sid)

    def get_messages(self, session_id=None, limit=None, after_seq=None):
        return self._db.get_messages(session_id=self._sid, limit=limit, after_seq=after_seq)


def _cmd_validate(rest):
    """Validate a YAML workflow file."""
    if not rest:
        print("Usage: hermes workflow validate <file.yaml>")
        return
    filepath = Path(rest[0])
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    if filepath.suffix not in (".yaml", ".yml"):
        print(f"Only YAML files are supported (.yaml/.yml), got: {filepath.suffix}")
        return

    try:
        from workflow.schema import load_yaml_file, validate_workflow
        wf = load_yaml_file(str(filepath))

        errors = validate_workflow(wf)
        if errors:
            print("❌ Validation failed:")
            for err in errors:
                print(f"  • {err}")
        else:
            print(f"✅ Valid workflow: {wf.get('name', filepath.stem)}")
            print(f"   Steps: {len(wf.get('nodes', {}))}")
            print(f"   Edges: {len(wf.get('edges', []))}")
    except Exception as e:
        print(f"❌ Parse error: {e}")


# ---------------------------------------------------------------------------
# Console log handler —— workflow 运行期间把 logger 输出也打到终端
# ---------------------------------------------------------------------------
#
# hermes_logging.setup_logging(mode="cli") 默认只给 root logger 挂
# RotatingFileHandler（写入 ~/.hermes/logs/agent.log），终端并不会看到
# logger.info() 的输出。但 `workflow run` 是交互式命令，用户希望实时
# 看到每一步进度，因此我们在进入 workflow 执行前，临时给 root logger
# 添加一个只输出简洁消息的 StreamHandler，结束后再移除，不影响其他
# 子命令的日志行为。
# ---------------------------------------------------------------------------

_WF_CONSOLE_FMT = "[%(asctime)s] %(message)s"
_WF_CONSOLE_DATEFMT = "%Y-%m-%d %H:%M:%S"


def _install_console_log_handler() -> logging.Handler:
    """为 root logger 安装一个终端 StreamHandler，供 workflow 运行期间使用。

    输出格式为 `[YYYY-MM-DD HH:MM:SS] <message>`，带时间戳但不带 logger 名，
    既能看到进度节拍又保持视觉整洁。
    返回的 handler 需由 _remove_console_log_handler() 在结束时移除。
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    handler.setFormatter(logging.Formatter(_WF_CONSOLE_FMT, datefmt=_WF_CONSOLE_DATEFMT))
    # 标记，便于识别/清理
    handler._hermes_wf_console = True  # type: ignore[attr-defined]

    root = logging.getLogger()
    # 如果 root 级别高于 INFO，降到 INFO 否则我们的 INFO 日志会被过滤掉
    if root.level == logging.NOTSET or root.level > logging.INFO:
        handler._hermes_wf_prev_root_level = root.level  # type: ignore[attr-defined]
        root.setLevel(logging.INFO)
    root.addHandler(handler)
    return handler


def _remove_console_log_handler(handler: logging.Handler) -> None:
    """移除 _install_console_log_handler() 安装的终端 handler。"""
    if handler is None:
        return
    root = logging.getLogger()
    try:
        root.removeHandler(handler)
    except Exception:
        pass
    try:
        handler.close()
    except Exception:
        pass
    # 恢复 root level（如果我们改过）
    prev = getattr(handler, "_hermes_wf_prev_root_level", None)
    if prev is not None:
        try:
            root.setLevel(prev)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Live progress callback —— 用 logger 输出节点进度（保留树形 + emoji）
# ---------------------------------------------------------------------------

_NODE_EMOJI = {
    "script": "📜",
    "shell": "🐚",
    "agent": "🤖",
    "iteration": "🔄",
    "iteration_item": "📦",
    "if": "❓",
    "parallel": "⚡",
    "template": "📝",
}


def _live_progress_callback(node_id: str, node_type: str, status: str,
                             result: dict, duration: float,
                             depth: int = 0):
    """Emit workflow step progress as log lines.

    输出形式（保留树形前缀 + emoji，[TYPE]/[OK]/[FAIL] 更醒目）：

      顶层 start :  🔄 [ITERATION] collect_stocks ...
      顶层 ok    :      ✅ [OK] 🔄 collect_stocks (8.56s) → count: 103
      顶层 fail  :      ❌ [FAIL] 🔄 collect_stocks (8.56s) → __error__: timeout
      iter_item  :  │  ├─ [ITEM] idx=3 ...
                    │  └─ ✅ [OK] [ITEM] idx=3 (0.31s)
      sub_step   :  │  │  ├─ 📜 [SCRIPT] fetch_data ...
                    │  │  └─ ✅ [OK] 📜 fetch_data (1.23s) → count: 42

    任何异常都不会抛出（吞掉），避免把 callback 错误误判为 item 失败。
    """
    try:
        is_iter_item = node_type == "iteration_item"
        is_sub_step = depth > 0 and not is_iter_item

        emoji = _NODE_EMOJI.get(node_type, "📦")
        type_tag = f"[{node_type.upper()}]"

        if status == "start":
            if is_iter_item:
                line = f"│  ├─ [ITEM] {node_id} ..."
            elif is_sub_step:
                line = f"│  │  ├─ {emoji} {type_tag} {node_id} ..."
            else:
                line = f"{emoji} {type_tag} {node_id} ..."
            logger.info(line)
            return

        # status == "ok" | "error"
        is_ok = status == "ok"
        state_tag = "[OK]" if is_ok else "[FAIL]"
        icon = "✅" if is_ok else "❌"

        # 预览（取第一个非 __ key）或错误信息
        preview = ""
        if result and isinstance(result, dict):
            if not is_ok and "__error__" in result:
                err_str = str(result["__error__"]).replace("\n", " ")
                if len(err_str) > 80:
                    err_str = err_str[:80] + "..."
                preview = f" → __error__: {err_str}"
            else:
                for k, v in result.items():
                    if k.startswith("__"):
                        continue
                    s = str(v).replace("\n", " ")
                    if len(s) > 50:
                        s = s[:50] + "..."
                    preview = f" → {k}: {s}"
                    break

        if is_iter_item:
            line = f"│  └─ {icon} {state_tag} [ITEM] {node_id} ({duration}s){preview}"
        elif is_sub_step:
            line = f"│  │  └─ {icon} {state_tag} {emoji} {node_id} ({duration}s){preview}"
        else:
            line = f"    {icon} {state_tag} {emoji} {node_id} ({duration}s){preview}"

        if is_ok:
            logger.info(line)
        else:
            logger.error(line)
    except Exception as _e:
        # callback 出错绝不能向上抛（engine 的 future.result() 会把它当作 item 失败）
        try:
            logger.warning("[progress-callback-error] %s", _e)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_inputs(args: list) -> dict:
    """Parse key=val input arguments."""
    inputs = {}
    for arg in args:
        if "=" in arg:
            key, val = arg.split("=", 1)
            # Try to parse value as JSON, fall back to string
            try:
                parsed = json.loads(val)
                # If the parsed value is a number but the original string
                # looks like it could be an identifier (e.g., date, id, code),
                # keep it as string to avoid type confusion
                if isinstance(parsed, (int, float)) and val.isdigit():
                    # For pure numeric strings that might be identifiers,
                    # keep them as strings to preserve leading zeros and
                    # avoid type issues in scripts
                    inputs[key] = val
                else:
                    inputs[key] = parsed
            except json.JSONDecodeError:
                inputs[key] = val
    return inputs


def _is_iteration_result(step_result: dict) -> bool:
    return "results" in step_result and isinstance(step_result["results"], list)


def _innermost_output(output: dict) -> str:
    """Follow the 'output' chain to get the deepest non-dict value."""
    while isinstance(output, dict) and "output" in output:
        output = output["output"]
    if isinstance(output, dict):
        # just show first key
        first_key = next(iter(output), None)
        if first_key is not None:
            v = output[first_key]
            s = json.dumps(v, ensure_ascii=False)
            return s[:60] + ("..." if len(s) > 60 else "")
        return "{}"
    s = json.dumps(output, ensure_ascii=False)
    return s[:60] + ("..." if len(s) > 60 else "")


def _print_run_result(result: dict):
    """Print a workflow run result."""
    status = result["status"]
    duration = result["duration_seconds"]
    run_id = result["run_id"]

    status_icon = "✅" if status == "ok" else "❌"
    print(f"\n{'='*60}")
    print(f"{status_icon} Status: {status} | Duration: {duration}s | Run ID: {run_id}")

    if result.get("errors"):
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  ✗ {err}")

    # Show step-by-step results
    outputs = result.get("outputs", {})
    if outputs:
        print("\nSteps:")
        for step_id, step_data in outputs.items():
            step_type = step_data.get("type", "?")
            step_dur = step_data.get("duration_seconds", 0)
            step_result = step_data.get("result", {})
            has_error = "__error__" in step_result
            icon = "❌" if has_error else "✅"

            print(f"  {icon} {step_id} ({step_type}, {step_dur}s)")

            # Special formatting for iteration results (don't indent the whole dict)
            if not has_error and _is_iteration_result(step_result):
                # Collect preview lines without base indent
                preview_lines = []
                results = step_result["results"]
                shown = results[:3]
                for item in shown:
                    idx = item["index"]
                    item_val = item["item"]
                    output = item.get("output", {})
                    display_val = _innermost_output(output)
                    item_str = json.dumps(item_val, ensure_ascii=False)
                    if len(item_str) > 40:
                        item_str = item_str[:40] + "..."
                    preview_lines.append(f"[{idx}] {item_str} → {display_val}")
                for line in preview_lines:
                    print(f"    {line}")
                if len(results) > 3:
                    print(f"    ... (+{len(results) - 3} more)")
                if step_result.get("__errors__"):
                    print(f"    ⚠ {len(step_result['__errors__'])} errors")
                continue

            # Truncate output for display
            preview = ""
            if has_error:
                preview = step_result["__error__"][:100]
            else:
                for k, val in step_result.items():
                    if k.startswith("__"):
                        continue
                    val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                    preview += f"  {k}: {val_str[:80]}{'...' if len(val_str) > 80 else ''}\n"

            if preview:
                for line in preview.strip().split("\n"):
                    print(f"    {line}")

    # Show final output
    final_output = result.get("final_output", {})
    if final_output and status == "ok":
        print("\nFinal Output:")
        for key, val in final_output.items():
            if key.startswith("__"):
                continue
            val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            if len(val_str) > 300:
                print(f"  {key}: {val_str[:300]}...")
            else:
                print(f"  {key}: {val_str}")


# ---------------------------------------------------------------------------
# Heartbeat Monitor —— 基于 agent_registry 的活体心跳（日志形式）
# ---------------------------------------------------------------------------

class _HeartbeatMonitor:
    """Workflow 执行期间的活体心跳监视器。

    信号来源是 `workflow.agent_registry.snapshot()`，读取每个正在运行的
    AIAgent 的 `_last_activity_ts` / `_last_activity_desc`——只要 agent
    真的卡住（HTTP 挂起 / 死锁 / 无限循环），这个时间戳就会冻住，心跳
    行的 idle 秒数就会不断上涨，用户立刻能看出是"真卡"还是"正常慢"。

    输出形式：每 _LOG_INTERVAL 秒通过 logger.info 打一行聚合状态，例如：
        💓 ⏳ 3 running · 🟢 2 active · 💤 1 slow · oldest 52s
        💓 ⚠ 2 running · 🔴 1 STUCK 145s · 🟢 1 active · oldest 145s
    """

    # 刷新频率（秒）——日志形式下不需要高频刷新，10s 一行即可
    _LOG_INTERVAL = 10.0
    # idle 告警阈值（秒）——超过视为"疑似卡住"
    _STALL_WARN_SECONDS = 60.0

    # 全局单例（保留以便未来扩展，不强制依赖）
    _current: "Optional[_HeartbeatMonitor]" = None
    _current_lock = threading.Lock()

    def __init__(self):
        self._stop_event = threading.Event()
        self._thread: "Optional[threading.Thread]" = None
        self._last_emit_text: str = ""
        self._last_emit_ts: float = 0.0

    @classmethod
    def current(cls) -> "Optional[_HeartbeatMonitor]":
        with cls._current_lock:
            return cls._current

    @classmethod
    def start_default(cls) -> "_HeartbeatMonitor":
        m = cls()
        m.start()
        with cls._current_lock:
            cls._current = m
        return m

    # ---- 生命周期 ----------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="wf-heartbeat", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        t = self._thread
        if t is not None:
            t.join(timeout=2.0)
        self._thread = None
        with self._current_lock:
            if _HeartbeatMonitor._current is self:
                _HeartbeatMonitor._current = None

    # ---- 渲染 --------------------------------------------------------------

    def _run_loop(self) -> None:
        # 每 0.5 秒 wake 一次以便快速响应 stop，但只在 _LOG_INTERVAL
        # 到期时才真正 emit 一行日志。
        tick_interval = 0.5
        while not self._stop_event.is_set():
            try:
                self._tick()
            except Exception:
                logger.debug("heartbeat tick failed", exc_info=True)
            self._stop_event.wait(tick_interval)

    def _tick(self) -> None:
        try:
            from workflow.agent_registry import snapshot
        except Exception:
            return

        try:
            snap = snapshot()
        except Exception:
            logger.debug("heartbeat snapshot() raised", exc_info=True)
            return

        text = self._format_snapshot(snap)
        if not text:
            # 没有 agent 活体，不输出
            return

        now = time.time()
        # 控制频率：仅在 _LOG_INTERVAL 到期或文本关键状态变化时 emit
        due = (now - self._last_emit_ts) >= self._LOG_INTERVAL
        changed = text != self._last_emit_text
        # 关键状态变化（出现/消失 STUCK）立刻 emit，不等 interval
        key_state_changed = ("STUCK" in text) != ("STUCK" in self._last_emit_text)

        if not (due or key_state_changed):
            return
        if not due and not key_state_changed:
            return
        # 文本完全没变且未到 interval，也不重复打
        if not changed and not due:
            return

        logger.info("💓 %s", text)
        self._last_emit_text = text
        self._last_emit_ts = now

    def _format_snapshot(self, snap) -> str:
        """聚合渲染：不逐条列 agent，只显示总数 + active/slow/stuck 计数 + oldest 秒数。

        示例：
          全健康：  ⏳ 3 running · 🟢 2 active · 💤 1 slow · oldest 52s
          有卡住：  ⚠ 3 running · 🔴 1 STUCK 145s · 💤 1 slow · 🟢 1 active · oldest 145s
          仅 1 个:  ⏳ 1 running · 🟢 active · 18s
          卡住 1 个: ⚠ 1 running · 🔴 STUCK · idle 132s · 145s
        """
        if not snap:
            return ""

        total = len(snap)
        active = slow = stuck = 0
        oldest_run = 0.0
        max_stuck_run = 0.0
        max_stuck_idle = 0.0

        for rec in snap:
            idle = rec.get("idle_seconds")
            run_s = rec.get("running_seconds") or 0.0
            if run_s > oldest_run:
                oldest_run = run_s

            if idle is None or idle < 3.0:
                active += 1
            elif idle < self._STALL_WARN_SECONDS:
                slow += 1
            else:
                stuck += 1
                if run_s > max_stuck_run:
                    max_stuck_run = run_s
                if idle > max_stuck_idle:
                    max_stuck_idle = idle

        # 单个 agent 走紧凑分支，避免冗余
        if total == 1:
            rec = snap[0]
            run_s = int(rec.get("running_seconds") or 0)
            idle = rec.get("idle_seconds")
            if idle is None or idle < 3.0:
                return f"⏳ 1 running · 🟢 active · {run_s}s"
            if idle < self._STALL_WARN_SECONDS:
                return f"⏳ 1 running · 💤 slow · idle {int(idle)}s · {run_s}s"
            return f"⚠ 1 running · 🔴 STUCK · idle {int(idle)}s · {run_s}s"

        # 多 agent 聚合
        head = f"⚠ {total} running" if stuck else f"⏳ {total} running"

        parts = []
        if stuck:
            parts.append(f"🔴 {stuck} STUCK {int(max_stuck_run)}s")
        if slow:
            parts.append(f"💤 {slow} slow")
        if active:
            parts.append(f"🟢 {active} active")

        parts.append(f"oldest {int(oldest_run)}s")
        return head + " · " + " · ".join(parts)