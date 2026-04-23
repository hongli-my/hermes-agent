"""
Hermes Workflow Engine — DAG-based workflow execution.

Supports:
  - script nodes:   run Python scripts, stdin JSON → stdout JSON
  - shell nodes:    run arbitrary shell commands, capture stdout/stderr/returncode
  - agent nodes:    run AIAgent, prompt → final_response
  - iteration nodes: loop over items, execute sub_steps per item
  - if nodes:       conditional branch (then/else)
  - parallel nodes: run branches concurrently
  - template nodes: render string templates with variables

Workflow definitions: YAML only (steps-based).
  Users author YAML, which is parsed into internal format (nodes + edges)
  and stored as JSON for programmatic access.

Storage: ~/.hermes/workflows/{workflow_id}/workflow.json
Output:  ~/.hermes/workflows/{workflow_id}/runs/{run_id}/output.json
Scripts: ~/.hermes/scripts/
"""

# Engine — execution
from workflow.engine import run_workflow

# Store — CRUD
from workflow.store import (
    create_workflow,
    create_workflow_from_yaml,
    delete_workflow,
    get_workflow,
    get_workflow_run,
    list_workflow_runs,
    list_workflows,
    update_workflow,
)

# Schema — YAML parsing
from workflow.schema import (
    load_yaml_file,
    parse_yaml_workflow,
    validate_workflow,
)

# Renderer — variable resolution
from workflow.renderer import VarPool, render, evaluate_condition

# Runner — Lua API 调用入口
from workflow.runner import run_workflow as run

__all__ = [
    # Engine
    "run_workflow",
    # Store
    "create_workflow",
    "create_workflow_from_yaml",
    "delete_workflow",
    "get_workflow",
    "get_workflow_run",
    "list_workflow_runs",
    "list_workflows",
    "update_workflow",
    # Schema
    "load_yaml_file",
    "parse_yaml_workflow",
    "validate_workflow",
    # Renderer
    "VarPool",
    "render",
    "evaluate_condition",
    # Runner
    "run",
]
