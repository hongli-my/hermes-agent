"""
Hermes Workflow Engine — DAG-based workflow execution.

Supports:
  - script nodes: run Python scripts, stdout → variables
  - agent nodes:  run AIAgent, final_response → variables
  - iterate nodes: loop over a list, execute body per item
  - parallel nodes: run sub-nodes concurrently (future)

Workflow is defined as JSON (nodes + edges).
Edges define execution order; the engine topologically sorts and runs them.

Storage: ~/.hermes/workflows/{workflow_id}/workflow.json
Output:  ~/.hermes/workflows/{workflow_id}/runs/{run_id}/
"""

import copy
import json
import logging
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage paths — lazy init to avoid import-time errors with hermes_constants
# ---------------------------------------------------------------------------

_wf_dir_cache = None

def _get_workflows_dir():
    global _wf_dir_cache
    if _wf_dir_cache is not None:
        return _wf_dir_cache
    try:
        from hermes_constants import get_hermes_home
        _wf_dir_cache = get_hermes_home() / "workflows"
    except Exception:
        from pathlib import Path as _P
        _wf_dir_cache = _P.home() / ".hermes" / "workflows"
    _wf_dir_cache.mkdir(parents=True, exist_ok=True)
    return _wf_dir_cache

def _hermes_now():
    try:
        from hermes_time import now
        return now()
    except Exception:
        from datetime import datetime as _dt
        return _dt.now().astimezone()

def _ensure_dirs():
    _get_workflows_dir().mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Variable resolution — replace {{node_id.key}} with values from var_pool
# ---------------------------------------------------------------------------

_VAR_PATTERN = re.compile(r'\{\{(\w+)\.(\w+)\}\}')

def resolve_variables(text: str, var_pool: Dict[str, Any]) -> str:
    """Replace {{node_id.key}} placeholders with actual values from var_pool."""
    if not text:
        return text

    def _replacer(match):
        node_id = match.group(1)
        key = match.group(2)
        node_vars = var_pool.get(node_id, {})
        value = node_vars.get(key) if isinstance(node_vars, dict) else None
        if value is None:
            # Try stringifying the whole node output
            node_output = var_pool.get(node_id)
            if node_output is not None:
                return json.dumps(node_output, ensure_ascii=False)
            return match.group(0)  # Leave unresolved
        if isinstance(value, str):
            return value
        return json.dumps(value, ensure_ascii=False)

    return _VAR_PATTERN.sub(_replacer, text)


def _parse_script_output(stdout: str) -> Dict[str, Any]:
    """Try to parse script stdout as JSON; fall back to {'output': raw}."""
    stdout = stdout.strip()
    if not stdout:
        return {"output": ""}
    try:
        parsed = json.loads(stdout)
        if isinstance(parsed, dict):
            return parsed
        return {"output": parsed}
    except json.JSONDecodeError:
        return {"output": stdout}

# ---------------------------------------------------------------------------
# Topological sort
# ---------------------------------------------------------------------------

def _topological_sort(nodes: Dict[str, Any], edges: List[List[str]]) -> List[str]:
    """Return node IDs in execution order. Raises on cycles."""
    in_degree = {nid: 0 for nid in nodes}
    adjacency = {nid: [] for nid in nodes}

    for src, dst in edges:
        if src in adjacency:
            adjacency[src].append(dst)
        if dst in in_degree:
            in_degree[dst] += 1

    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    order = []

    while queue:
        # Sort for deterministic order
        queue.sort()
        nid = queue.pop(0)
        order.append(nid)
        for neighbor in adjacency.get(nid, []):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(order) != len(nodes):
        raise ValueError("Workflow contains a cycle")

    return order

# ---------------------------------------------------------------------------
# Node executors
# ---------------------------------------------------------------------------

def _run_script_node(node: Dict[str, Any], var_pool: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a Python script node. Script stdout → variables."""
    scripts_dir = _get_workflows_dir().parent / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)

    script_path = node.get("script", "")
    raw = Path(script_path).expanduser()
    if raw.is_absolute():
        resolved = raw.resolve()
    else:
        resolved = (scripts_dir / raw).resolve()

    # Security: must be within scripts dir
    try:
        resolved.relative_to(scripts_dir.resolve())
    except ValueError:
        return {"__error__": f"Script path escapes scripts directory: {script_path}"}

    if not resolved.exists():
        return {"__error__": f"Script not found: {resolved}"}

    # Build env with upstream variables available
    env = os.environ.copy()
    env["HERMES_WORKFLOW_VARS"] = json.dumps(var_pool, ensure_ascii=False)

    timeout = node.get("timeout", 120)

    try:
        result = subprocess.run(
            [sys.executable, str(resolved)],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(resolved.parent),
            env=env,
        )
        if result.returncode != 0:
            return {"__error__": f"Script exited with code {result.returncode}: {result.stderr.strip()}"}

        outputs = _parse_script_output(result.stdout)

        # If node defines output_keys, filter
        output_keys = node.get("outputs")
        if output_keys and isinstance(output_keys, list):
            filtered = {}
            for key in output_keys:
                if key in outputs:
                    filtered[key] = outputs[key]
            # Keep any __error__ key
            if "__error__" in outputs:
                filtered["__error__"] = outputs["__error__"]
            return filtered

        return outputs

    except subprocess.TimeoutExpired:
        return {"__error__": f"Script timed out after {timeout}s"}
    except Exception as e:
        return {"__error__": f"Script execution failed: {e}"}


def _run_agent_node(node: Dict[str, Any], var_pool: Dict[str, Any],
                    session_db=None) -> Dict[str, Any]:
    """Execute an agent node. AIAgent → final_response → variables."""
    from run_agent import AIAgent
    from dotenv import load_dotenv

    prompt_template = node.get("prompt", "")
    prompt = resolve_variables(prompt_template, var_pool)

    if not prompt.strip():
        return {"__error__": "Agent prompt is empty after variable resolution"}

    # Load config
    _hermes_home = get_hermes_home()
    try:
        load_dotenv(str(_hermes_home / ".env"), override=True, encoding="utf-8")
    except Exception:
        pass

    _cfg = {}
    try:
        import yaml
        _cfg_path = str(_hermes_home / "config.yaml")
        if os.path.exists(_cfg_path):
            with open(_cfg_path) as f:
                _cfg = yaml.safe_load(f) or {}
    except Exception:
        pass

    model = node.get("model") or os.getenv("HERMES_MODEL", "")
    if not model:
        _model_cfg = _cfg.get("model", {})
        if isinstance(_model_cfg, str):
            model = _model_cfg
        elif isinstance(_model_cfg, dict):
            model = _model_cfg.get("default", model)

    # Provider resolution
    from hermes_cli.runtime_provider import resolve_runtime_provider
    try:
        runtime_kwargs = {
            "requested": node.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
        }
        if node.get("base_url"):
            runtime_kwargs["explicit_base_url"] = node.get("base_url")
        runtime = resolve_runtime_provider(**runtime_kwargs)
    except Exception as exc:
        return {"__error__": f"Provider resolution failed: {exc}"}

    # Smart routing
    from agent.smart_model_routing import resolve_turn_route
    smart_routing = _cfg.get("smart_model_routing", {}) or {}
    turn_route = resolve_turn_route(
        prompt, smart_routing,
        {
            "model": model,
            "api_key": runtime.get("api_key"),
            "base_url": runtime.get("base_url"),
            "provider": runtime.get("provider"),
            "api_mode": runtime.get("api_mode"),
            "command": runtime.get("command"),
            "args": list(runtime.get("args") or []),
        },
    )

    max_iterations = node.get("max_iterations", 30)
    enabled_toolsets = node.get("toolsets")  # None = all
    disabled_toolsets = node.get("disabled_toolsets", ["cronjob", "messaging", "clarify"])

    # Load skills if specified
    skill_names = node.get("skills") or []
    if skill_names:
        from tools.skills_tool import skill_view
        skill_parts = []
        for sn in skill_names:
            loaded = json.loads(skill_view(sn))
            if loaded.get("success"):
                content = str(loaded.get("content", "")).strip()
                if content:
                    skill_parts.append(
                        f'[Skill: {sn}]\n{content}'
                    )
        if skill_parts:
            prompt = "\n\n".join(skill_parts) + "\n\n---\n\n" + prompt

    # System prompt
    system_prompt = node.get("system_prompt")

    agent = AIAgent(
        model=turn_route["model"],
        api_key=turn_route["runtime"].get("api_key"),
        base_url=turn_route["runtime"].get("base_url"),
        provider=turn_route["runtime"].get("provider"),
        api_mode=turn_route["runtime"].get("api_mode"),
        acp_command=turn_route["runtime"].get("command"),
        acp_args=turn_route["runtime"].get("args"),
        max_iterations=max_iterations,
        enabled_toolsets=enabled_toolsets,
        disabled_toolsets=disabled_toolsets,
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="workflow",
        session_db=session_db,
    )

    try:
        result = agent.run_conversation(prompt, system_message=system_prompt)
        final_response = result.get("final_response", "") or ""

        output_keys = node.get("outputs", ["response"])
        outputs = {}
        if len(output_keys) == 1:
            outputs[output_keys[0]] = final_response
        elif len(output_keys) > 1:
            # Try parse as JSON for multi-output
            try:
                parsed = json.loads(final_response)
                if isinstance(parsed, dict):
                    for key in output_keys:
                        outputs[key] = parsed.get(key, "")
                else:
                    outputs[output_keys[0]] = final_response
            except json.JSONDecodeError:
                outputs[output_keys[0]] = final_response
        else:
            outputs["response"] = final_response

        return outputs

    except Exception as e:
        return {"__error__": f"Agent execution failed: {e}"}


def _run_iterate_node(node: Dict[str, Any], var_pool: Dict[str, Any],
                      session_db=None) -> Dict[str, Any]:
    """Execute an iterate node: loop over a list, run body per item."""
    over_template = node.get("over", "")
    over_expr = resolve_variables(over_template, var_pool)

    # Resolve the list
    items = None
    # Try parsing as JSON
    try:
        items = json.loads(over_expr)
    except json.JSONDecodeError:
        pass

    # Try dot-notation from var_pool
    if items is None:
        match = _VAR_PATTERN.match(over_template)
        if match:
            node_id, key = match.group(1), match.group(2)
            node_vars = var_pool.get(node_id, {})
            if isinstance(node_vars, dict):
                items = node_vars.get(key)

    if items is None:
        return {"__error__": f"Cannot resolve iterate source: {over_template}"}

    if not isinstance(items, list):
        items = [items]

    item_var = node.get("item_var", "item")
    body = node.get("body", {})
    output_key = (node.get("outputs") or ["results"])[0]

    results = []
    errors = []

    for i, item in enumerate(items):
        logger.info("Iterate[%d/%d]: processing item", i + 1, len(items))

        # Create a child var_pool with the current item
        child_pool = copy.deepcopy(var_pool)
        child_pool["__iter__"] = {
            item_var: item,
            "index": i,
            "total": len(items),
        }

        # Execute the body node
        body_result = _execute_single_node(body, child_pool, session_db=session_db)

        if "__error__" in body_result:
            errors.append({"index": i, "error": body_result["__error__"]})
            # Continue or stop based on config
            if node.get("stop_on_error", False):
                break
        else:
            results.append(body_result)

    output = {output_key: results}
    if errors:
        output["__errors__"] = errors

    return output


def _execute_single_node(node: Dict[str, Any], var_pool: Dict[str, Any],
                         session_db=None) -> Dict[str, Any]:
    """Dispatch to the right executor based on node type."""
    node_type = node.get("type", "agent")

    if node_type == "script":
        return _run_script_node(node, var_pool)
    elif node_type == "agent":
        return _run_agent_node(node, var_pool, session_db=session_db)
    elif node_type == "iterate":
        return _run_iterate_node(node, var_pool, session_db=session_db)
    else:
        return {"__error__": f"Unknown node type: {node_type}"}

# ---------------------------------------------------------------------------
# Workflow execution engine
# ---------------------------------------------------------------------------

def run_workflow(workflow: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None,
                 session_db=None) -> Dict[str, Any]:
    """
    Execute a workflow definition.

    Args:
        workflow: The workflow JSON (nodes + edges).
        inputs: Optional input variables (available as {{inputs.key}}).
        session_db: Optional SessionDB for agent session persistence.

    Returns:
        {
            "status": "ok" | "error",
            "outputs": { node_id: { key: value } },
            "errors": [ ... ],
            "duration_seconds": float,
            "run_id": str,
        }
    """
    run_id = uuid.uuid4().hex[:12]
    start_time = time.time()

    nodes = workflow.get("nodes", {})
    edges = workflow.get("edges", [])

    # Validate nodes
    if not nodes:
        return {
            "status": "error",
            "outputs": {},
            "errors": ["Workflow has no nodes"],
            "duration_seconds": 0,
            "run_id": run_id,
        }

    # Topological sort
    try:
        execution_order = _topological_sort(nodes, edges)
    except ValueError as e:
        return {
            "status": "error",
            "outputs": {},
            "errors": [str(e)],
            "duration_seconds": 0,
            "run_id": run_id,
        }

    # Variable pool: stores outputs of completed nodes
    var_pool = {"inputs": inputs or {}}

    errors = []
    all_outputs = {}

    for node_id in execution_order:
        node_def = nodes[node_id]
        node_type = node_def.get("type", "agent")

        logger.info("Workflow[%s]: running node '%s' (type=%s)", run_id, node_id, node_type)

        node_start = time.time()

        result = _execute_single_node(node_def, var_pool, session_db=session_db)

        node_duration = time.time() - node_start

        # Store result in var_pool and all_outputs
        var_pool[node_id] = result
        all_outputs[node_id] = {
            "result": result,
            "duration_seconds": round(node_duration, 2),
            "type": node_type,
        }

        if "__error__" in result:
            error_msg = f"Node '{node_id}' failed: {result['__error__']}"
            errors.append(error_msg)
            logger.error("Workflow[%s]: %s", run_id, error_msg)

            # Stop on error by default
            if not workflow.get("continue_on_error", False):
                break

        logger.info("Workflow[%s]: node '%s' completed in %.2fs", run_id, node_id, node_duration)

    # Find the last node's output as the workflow's final output
    final_node_id = execution_order[-1] if execution_order else None
    final_output = var_pool.get(final_node_id, {}) if final_node_id else {}

    # Save run output
    _save_run_output(workflow.get("id", "unknown"), run_id, {
        "run_id": run_id,
        "workflow_name": workflow.get("name", "unnamed"),
        "status": "error" if errors else "ok",
        "started_at": datetime.fromtimestamp(start_time).isoformat(),
        "duration_seconds": round(time.time() - start_time, 2),
        "execution_order": execution_order,
        "outputs": all_outputs,
        "final_output": final_output,
        "errors": errors,
    })

    return {
        "status": "error" if errors else "ok",
        "outputs": all_outputs,
        "final_output": final_output,
        "errors": errors,
        "duration_seconds": round(time.time() - start_time, 2),
        "run_id": run_id,
    }


def _save_run_output(workflow_id: str, run_id: str, data: Dict[str, Any]):
    """Save run output to file."""
    _ensure_dirs()
    run_dir = _get_workflows_dir() / workflow_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / "output.json"
    tmp_file = run_dir / ".output.json.tmp"

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, output_file)

# ---------------------------------------------------------------------------
# Workflow CRUD
# ---------------------------------------------------------------------------

def create_workflow(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new workflow. Returns the stored workflow with id."""
    _ensure_dirs()

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


def get_workflow(wf_id: str) -> Optional[Dict[str, Any]]:
    """Get a workflow by ID."""
    wf_file = _get_workflows_dir() / wf_id / "workflow.json"
    if not wf_file.exists():
        return None
    with open(wf_file, "r", encoding="utf-8") as f:
        return json.load(f)


def list_workflows() -> List[Dict[str, Any]]:
    """List all workflows (metadata only)."""
    _ensure_dirs()
    workflows = []
    for wf_dir in sorted(_get_workflows_dir().iterdir()):
        if not wf_dir.is_dir():
            continue
        wf_file = wf_dir / "workflow.json"
        if wf_file.exists():
            try:
                with open(wf_file, "r", encoding="utf-8") as f:
                    wf = json.load(f)
                workflows.append({
                    "id": wf.get("id", wf_dir.name),
                    "name": wf.get("name", wf_dir.name),
                    "nodes_count": len(wf.get("nodes", {})),
                    "created_at": wf.get("created_at"),
                    "updated_at": wf.get("updated_at"),
                })
            except Exception:
                pass
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
    import shutil
    wf_dir = _get_workflows_dir() / wf_id
    if wf_dir.exists() and wf_dir.is_dir():
        shutil.rmtree(wf_dir)
        return True
    return False


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


def _save_workflow_file(wf_id: str, wf_def: Dict[str, Any]):
    """Atomically save workflow.json."""
    wf_dir = _get_workflows_dir() / wf_id
    wf_dir.mkdir(parents=True, exist_ok=True)

    wf_file = wf_dir / "workflow.json"
    tmp_file = wf_dir / ".workflow.json.tmp"

    with open(tmp_file, "w", encoding="utf-8") as f:
        json.dump(wf_def, f, ensure_ascii=False, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())

    os.replace(tmp_file, wf_file)
