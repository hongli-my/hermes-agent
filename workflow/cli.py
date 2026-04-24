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

    print(f"Running workflow '{wf.get('name', wf_id)}' ...")
    result = run_workflow(wf, inputs=inputs)

    _print_run_result(result)


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
        print(f"Running workflow '{wf.get('name', filepath.stem)}' from file ...")
        result = run_workflow(wf, inputs=inputs)
        _print_run_result(result)
    except Exception as e:
        print(f"Failed to run workflow: {e}")
        import traceback
        traceback.print_exc()


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
                inputs[key] = json.loads(val)
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