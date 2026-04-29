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
import sys
import threading
from pathlib import Path


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
    result = run_workflow(wf, inputs=inputs, session_db=session_db,
                          progress_callback=_live_progress_callback)

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
        result = run_workflow(wf, inputs=inputs, session_db=session_db,
                              progress_callback=_live_progress_callback)
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
# Live progress callback
# ---------------------------------------------------------------------------

_NODE_EMOJI = {
    "script": "📜",
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
    """Print live progress during workflow execution with hierarchy."""
    
    # Determine if this is a sub-step within iteration
    is_iter_item = node_type == "iteration_item"
    is_sub_step = depth > 0 and not is_iter_item
    
    if status == "start":
        if is_iter_item:
            print(f"│  ├─ {node_id}...", flush=True)
        elif is_sub_step:
            emoji = _NODE_EMOJI.get(node_type, "📦")
            print(f"│  │  ├─ {emoji} {node_id}...", flush=True)
        else:
            emoji = _NODE_EMOJI.get(node_type, "📦")
            print(f"{emoji} {node_id} ({node_type})...", flush=True)
    else:
        if status == "ok":
            icon = "✅"
            # Show output preview
            if result and not any(k.startswith("__") for k in result):
                for k, v in result.items():
                    preview = str(v)
                    if len(preview) > 50:
                        preview = preview[:50] + "..."
                    preview = preview.replace("\n", " ")
                    if is_iter_item:
                        print(f"│  │  └─ {icon} {node_id} ({duration}s)")
                    elif is_sub_step:
                        print(f"│  │  │  {icon} {node_id} → {k}: {preview}")
                    else:
                        print(f"    {icon} {node_id} → {k}: {preview}")
                return
        else:
            icon = "❌"
        
        if is_iter_item:
            print(f"│  │  └─ {icon} {node_id} ({duration}s)")
        elif is_sub_step:
            print(f"│  │  │  {icon} {node_id} ({node_type}, {duration}s)")
        else:
            print(f"    {icon} {node_id} ({node_type}, {duration}s)")


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