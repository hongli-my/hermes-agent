"""
Hermes Workflow CLI subcommand.
Usage: hermes workflow list
       hermes workflow run <workflow_id>
       hermes workflow show <workflow_id>
       hermes workflow create <file.json>
       hermes workflow delete <workflow_id>
       hermes workflow runs <workflow_id>
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
    else:
        print(f"Unknown action: {action}")
        _print_help()


def _print_help():
    print("Usage: hermes workflow <action> [args]")
    print()
    print("Actions:")
    print("  list                  List all workflows")
    print("  show <id>             Show workflow definition")
    print("  create <file.json>    Create workflow from JSON file")
    print("  delete <id>           Delete a workflow")
    print("  run <id>              Execute a workflow")
    print("  runs <id>             List runs for a workflow")


def _cmd_list():
    from workflow import list_workflows
    workflows = list_workflows()
    if not workflows:
        print("No workflows found.")
        return
    print(f"{'ID':<14} {'Name':<25} {'Nodes':<7} {'Created'}")
    print("-" * 70)
    for wf in workflows:
        print(f"{wf['id']:<14} {wf.get('name', '-'):<25} {wf.get('nodes_count', 0):<7} {wf.get('created_at', '-')}")


def _cmd_show(rest):
    if not rest:
        print("Usage: hermes workflow show <workflow_id>")
        return
    from workflow import get_workflow
    wf = get_workflow(rest[0])
    if not wf:
        print(f"Workflow '{rest[0]}' not found.")
        return
    print(json.dumps(wf, indent=2, ensure_ascii=False))


def _cmd_create(rest):
    if not rest:
        print("Usage: hermes workflow create <file.json>")
        return
    filepath = Path(rest[0])
    if not filepath.exists():
        print(f"File not found: {filepath}")
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            wf_def = json.load(f)
    except Exception as e:
        print(f"Failed to parse JSON: {e}")
        return
    from workflow import create_workflow
    wf = create_workflow(wf_def)
    print(f"Workflow created: {wf['id']} ({wf.get('name', '-')})")


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
        print("Usage: hermes workflow run <workflow_id>")
        return
    from workflow import get_workflow, run_workflow
    wf = get_workflow(rest[0])
    if not wf:
        print(f"Workflow '{rest[0]}' not found.")
        return
    print(f"Running workflow '{wf.get('name', rest[0])}' ...")
    result = run_workflow(wf)
    status = result["status"]
    duration = result["duration_seconds"]
    print(f"\n{'='*60}")
    print(f"Status: {status} | Duration: {duration}s | Run ID: {result['run_id']}")
    if result.get("errors"):
        print("\nErrors:")
        for err in result["errors"]:
            print(f"  ✗ {err}")
    if result.get("final_output"):
        print("\nFinal Output:")
        for key, val in result["final_output"].items():
            if key.startswith("__"):
                continue
            val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
            print(f"  {key}: {val_str[:200]}{'...' if len(val_str) > 200 else ''}")


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
