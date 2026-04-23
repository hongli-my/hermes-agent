"""
Hermes Workflow Store — CRUD for workflow definitions and run history.

Storage layout:
  ~/.hermes/workflows/
    ├── <workflow_id>/
    │   ├── workflow.json    (internal parsed representation)
    │   └── runs/
    │       └── <run_id>/
    │           └── output.json
    └── ...

Users define workflows in YAML, which is parsed and stored as JSON internally.
The YAML source is the human-authorable format; JSON is the engine's internal format.
"""

import json
import os
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = None
try:
    import logging
    logger = logging.getLogger(__name__)
except Exception:
    pass


def _get_workflows_dir():
    try:
        from hermes_constants import get_hermes_home
        d = get_hermes_home() / "workflows"
    except Exception:
        d = Path.home() / ".hermes" / "workflows"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _hermes_now():
    try:
        from hermes_time import now
        return now()
    except Exception:
        return datetime.now().astimezone()


# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------

def create_workflow(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new workflow. Returns the stored workflow with id.

    Accepts a YAML-parsed workflow (from schema.parse_yaml_workflow or load_yaml_file).
    The workflow_def must have 'nodes' and 'edges' keys (produced by schema parser).
    """
    _ensure_dirs()

    # If the definition has steps but no nodes, parse it first
    if "steps" in workflow_def and "nodes" not in workflow_def:
        import yaml
        yaml_text = yaml.dump(workflow_def, allow_unicode=True, default_flow_style=False)
        from workflow.schema import parse_yaml_workflow
        workflow_def = parse_yaml_workflow(yaml_text)

    wf_id = workflow_def.get("id") or uuid.uuid4().hex[:12]
    workflow_def["id"] = wf_id
    workflow_def.setdefault("name", wf_id)
    workflow_def.setdefault("nodes", {})
    workflow_def.setdefault("edges", [])
    workflow_def["created_at"] = _hermes_now().isoformat()
    workflow_def["updated_at"] = _hermes_now().isoformat()

    wf_dir = _get_workflows_dir() / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)

    _save_workflow_file(wf_id, workflow_def)

    return workflow_def


def create_workflow_from_yaml(yaml_path: str) -> Dict[str, Any]:
    """Create a workflow from a YAML file."""
    from workflow.schema import load_yaml_file
    wf = load_yaml_file(yaml_path)
    return create_workflow(wf)


def get_workflow(wf_id: str) -> Optional[Dict[str, Any]]:
    """Get a workflow by ID.

    Reads the internal JSON representation (nodes + edges format).
    Users create workflows via YAML files, which are parsed and
    stored as JSON internally.
    """
    wf_dir = _get_workflows_dir() / wf_id

    # Read internal JSON representation
    wf_file = wf_dir / "workflow.json"
    if wf_file.exists():
        with open(wf_file, "r", encoding="utf-8") as f:
            return json.load(f)

    return None


def list_workflows() -> List[Dict[str, Any]]:
    """List all workflows (metadata only)."""
    _ensure_dirs()
    workflows = []

    for wf_dir in sorted(_get_workflows_dir().iterdir()):
        if not wf_dir.is_dir():
            continue
        if wf_dir.name.startswith("."):
            continue

        wf = None
        # Read internal JSON representation
        wf_file = wf_dir / "workflow.json"
        if wf_file.exists():
            try:
                with open(wf_file, "r", encoding="utf-8") as f:
                    wf = json.load(f)
            except Exception:
                pass

        if wf is not None:
            workflows.append({
                "id": wf.get("id", wf_dir.name),
                "name": wf.get("name", wf_dir.name),
                "description": wf.get("description", ""),
                "nodes_count": len(wf.get("nodes", {})),
                "created_at": wf.get("created_at"),
                "updated_at": wf.get("updated_at"),
            })
        else:
            # Directory exists but no valid workflow file
            workflows.append({
                "id": wf_dir.name,
                "name": wf_dir.name,
                "description": "",
                "nodes_count": 0,
                "created_at": None,
                "updated_at": None,
            })

    return workflows


def update_workflow(wf_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a workflow definition."""
    wf = get_workflow(wf_id)
    if not wf:
        return None
    wf.update(updates)
    wf["updated_at"] = _hermes_now().isoformat()
    _save_workflow_file(wf_id, wf)
    return wf


def delete_workflow(wf_id: str) -> bool:
    """Delete a workflow and all its run history."""
    wf_dir = _get_workflows_dir() / wf_id
    if wf_dir.exists() and wf_dir.is_dir():
        shutil.rmtree(wf_dir)
        return True
    return False


# ---------------------------------------------------------------------------
# Run history
# ---------------------------------------------------------------------------

def list_workflow_runs(wf_id: str) -> List[Dict[str, Any]]:
    """List runs for a workflow."""
    runs_dir = _get_workflows_dir() / wf_id / "runs"
    if not runs_dir.exists():
        return []

    runs = []
    for run_dir in sorted(runs_dir.iterdir(), reverse=True):
        if not run_dir.is_dir():
            continue
        output_file = run_dir / "output.json"
        if output_file.exists():
            try:
                with open(output_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                runs.append({
                    "run_id": data.get("run_id", run_dir.name),
                    "status": data.get("status", "unknown"),
                    "started_at": data.get("started_at"),
                    "duration_seconds": data.get("duration_seconds"),
                    "errors_count": len(data.get("errors", [])),
                })
            except Exception:
                pass

    return runs


def get_workflow_run(wf_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific run's output."""
    output_file = _get_workflows_dir() / wf_id / "runs" / run_id / "output.json"
    if not output_file.exists():
        return None
    with open(output_file, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    _get_workflows_dir().mkdir(parents=True, exist_ok=True)


def _save_workflow_file(wf_id: str, wf_def: Dict[str, Any]):
    """Atomically save workflow definition.

    Saves the internal representation (nodes + edges) as JSON for
    programmatic read/write. Users create workflows via YAML files,
    but the store keeps the parsed internal format for efficiency.
    """
    wf_dir = _get_workflows_dir() / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)

    wf_file = wf_dir / "workflow.json"
    tmp_file = wf_dir / ".workflow.json.tmp"

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(wf_def, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, wf_file)
