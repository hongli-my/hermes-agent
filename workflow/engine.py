"""
Hermes Workflow Engine — step-based execution engine.

Supports node types:
  - script:    run Python script, stdin JSON → stdout JSON
  - agent:     run AIAgent, prompt → final_response
  - iteration: loop over items, execute sub_steps per item
  - if:        conditional branch (then/else)
  - parallel:  run branches concurrently
  - template:  render string template with variables

Execution model:
  YAML steps are parsed into a DAG by schema.py.
  The engine executes nodes in topological order, but also supports
  step-by-step execution for workflows with if-node branching.

Variable flow:
  Each step's output is stored in VarPool and accessible as
  {{steps.<id>.output.<key>}} by downstream steps.
"""

import copy
import json
import logging
import os
import subprocess
import sys
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from workflow.renderer import VarPool, render, evaluate_condition
from workflow.schema import parse_yaml_workflow, load_yaml_file, validate_workflow

logger = logging.getLogger(__name__)

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
# Storage paths
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
# Script node executor
# ---------------------------------------------------------------------------

def _run_script_node(node: Dict[str, Any], pool: VarPool) -> Dict[str, Any]:
    """Execute a Python script node.

    The script receives its input as JSON on stdin.
    The script must write its output as JSON to stdout.
    stderr is captured for logging.
    """
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

    # Render input variables
    input_data = render(node.get("input", {}), pool)

    timeout = node.get("timeout", 120)

    try:
        result = subprocess.run(
            [sys.executable, str(resolved)],
            input=json.dumps(input_data, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=str(resolved.parent),
            env=os.environ.copy(),
        )

        # Log stderr if present
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.info("Script[%s] stderr: %s", node.get("id", "?"), line)

        if result.returncode != 0:
            return {"__error__": f"Script exited with code {result.returncode}: {result.stderr.strip()}"}

        # Parse stdout as JSON
        outputs = _parse_script_output(result.stdout)

        # If node defines output_keys, filter
        output_keys = node.get("outputs")
        if output_keys and isinstance(output_keys, list):
            filtered = {}
            for key in output_keys:
                if key in outputs:
                    filtered[key] = outputs[key]
            if "__error__" in outputs:
                filtered["__error__"] = outputs["__error__"]
            return filtered

        return outputs

    except subprocess.TimeoutExpired:
        return {"__error__": f"Script timed out after {timeout}s"}
    except Exception as e:
        return {"__error__": f"Script execution failed: {e}"}


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
# Agent node executor
# ---------------------------------------------------------------------------

def _run_agent_node(node: Dict[str, Any], pool: VarPool,
                    session_db=None) -> Dict[str, Any]:
    """Execute an agent node. AIAgent → final_response → variables."""
    try:
        from run_agent import AIAgent
        from dotenv import load_dotenv
    except ImportError:
        return {"__error__": "AIAgent not available (run_agent.py not importable)"}

    prompt_template = node.get("prompt", "")
    prompt = render(prompt_template, pool)

    if not prompt.strip():
        return {"__error__": "Agent prompt is empty after variable resolution"}

    # Load config
    _hermes_home = _get_hermes_home()
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
    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime_kwargs = {
            "requested": node.get("provider") or os.getenv("HERMES_INFERENCE_PROVIDER"),
        }
        if node.get("base_url"):
            runtime_kwargs["explicit_base_url"] = node.get("base_url")
        runtime = resolve_runtime_provider(**runtime_kwargs)
    except Exception as exc:
        return {"__error__": f"Provider resolution failed: {exc}"}

    # Smart routing
    try:
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
    except Exception:
        # Fallback without smart routing
        turn_route = {
            "model": model,
            "runtime": runtime,
        }

    max_iterations = node.get("max_iterations", 30)
    enabled_toolsets = node.get("toolsets")
    disabled_toolsets = node.get("disabled_toolsets", ["cronjob", "messaging", "clarify"])

    # Load skills if specified
    skill_names = node.get("skills") or []
    if skill_names:
        try:
            from tools.skills_tool import skill_view
            skill_parts = []
            for sn in skill_names:
                loaded = json.loads(skill_view(sn))
                if loaded.get("success"):
                    content = str(loaded.get("content", "")).strip()
                    if content:
                        skill_parts.append(f'[Skill: {sn}]\n{content}')
            if skill_parts:
                prompt = "\n\n".join(skill_parts) + "\n\n---\n\n" + prompt
        except Exception:
            pass

    system_prompt = node.get("system_prompt")

    agent = AIAgent(
        model=turn_route.get("model", model),
        api_key=turn_route.get("runtime", runtime).get("api_key"),
        base_url=turn_route.get("runtime", runtime).get("base_url"),
        provider=turn_route.get("runtime", runtime).get("provider"),
        api_mode=turn_route.get("runtime", runtime).get("api_mode"),
        acp_command=turn_route.get("runtime", runtime).get("command"),
        acp_args=turn_route.get("runtime", runtime).get("args"),
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

        output_keys = node.get("outputs", ["text"])
        outputs = {}
        if len(output_keys) == 1:
            outputs[output_keys[0]] = final_response
        elif len(output_keys) > 1:
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
            outputs["text"] = final_response

        return outputs

    except Exception as e:
        return {"__error__": f"Agent execution failed: {e}"}


# ---------------------------------------------------------------------------
# Template node executor
# ---------------------------------------------------------------------------

def _run_template_node(node: Dict[str, Any], pool: VarPool) -> Dict[str, Any]:
    """Render a string template with variable substitution."""
    template_str = node.get("template", "")
    rendered = render(template_str, pool)

    output_keys = node.get("outputs", ["text"])
    outputs = {}
    if len(output_keys) == 1:
        outputs[output_keys[0]] = rendered
    else:
        outputs["text"] = rendered

    return outputs


# ---------------------------------------------------------------------------
# If node executor
# ---------------------------------------------------------------------------

def _run_if_node(node: Dict[str, Any], pool: VarPool) -> Dict[str, Any]:
    """Evaluate a condition and record which branch to take.

    Returns:
        {"__branch__": "then"|"else", "__target__": step_id}
    """
    condition = node.get("condition", "")
    result = evaluate_condition(condition, pool)

    if result:
        return {"__branch__": "then", "__target__": node.get("then", "")}
    else:
        return {"__branch__": "else", "__target__": node.get("else", "")}


# ---------------------------------------------------------------------------
# Iteration node executor
# ---------------------------------------------------------------------------

def _run_iteration_node(node: Dict[str, Any], pool: VarPool,
                        session_db=None) -> Dict[str, Any]:
    """Loop over a list and execute sub_steps for each item."""
    # Resolve items list
    items_expr = node.get("items", "")
    items = render(items_expr, pool)

    # If items is a string, try parsing as JSON
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            return {"__error__": f"Cannot resolve iteration items: {items_expr}"}

    if not isinstance(items, list):
        items = [items]

    item_var = node.get("item_var", "item")
    sub_steps = node.get("sub_steps", [])
    max_concurrent = node.get("max_concurrent", 1)
    stop_on_error = node.get("stop_on_error", False)
    output_key = (node.get("outputs") or ["results"])[0]

    results = []
    errors = []

    for i, item in enumerate(items):
        logger.info("Iteration[%d/%d]: processing item", i + 1, len(items))

        # Push iteration context
        pool.push_iter_context(item_var, item, i, len(items))

        # Execute sub_steps sequentially
        iter_output = {}
        iter_error = None

        for sub_step in sub_steps:
            sub_type = sub_step.get("type", "agent")

            if sub_type == "script":
                sub_result = _run_script_node(sub_step, pool)
            elif sub_type == "agent":
                sub_result = _run_agent_node(sub_step, pool, session_db=session_db)
            elif sub_type == "template":
                sub_result = _run_template_node(sub_step, pool)
            else:
                sub_result = {"__error__": f"Unknown sub-step type: {sub_type}"}

            sub_id = sub_step.get("id", f"sub_{i}")
            iter_output[sub_id] = sub_result
            pool.set_step_output(sub_id, sub_result)

            if "__error__" in sub_result:
                iter_error = sub_result["__error__"]
                if stop_on_error:
                    break

        # Pop iteration context
        pool.pop_iter_context()

        if iter_error:
            errors.append({"index": i, "item": item, "error": iter_error})
            if stop_on_error:
                break
        else:
            results.append({
                "index": i,
                "item": item,
                "output": iter_output,
            })

    output = {output_key: results}
    if errors:
        output["__errors__"] = errors

    return output


# ---------------------------------------------------------------------------
# Parallel node executor
# ---------------------------------------------------------------------------

def _run_parallel_node(node: Dict[str, Any], pool: VarPool,
                       session_db=None) -> Dict[str, Any]:
    """Run multiple branches concurrently."""
    branches = node.get("branches", [])
    if not branches:
        return {}

    results = {}

    if len(branches) == 1:
        # No need for threading with a single branch
        branch = branches[0]
        branch_id = branch.get("id", "branch_0")
        result = _execute_single_node(branch, pool, session_db=session_db)
        results[branch_id] = result
    else:
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            futures = {}
            for idx, branch in enumerate(branches):
                branch_id = branch.get("id", f"branch_{idx}")
                # Each branch gets a copy of the pool
                branch_pool = copy.deepcopy(pool)
                future = executor.submit(
                    _execute_single_node, branch, branch_pool, session_db
                )
                futures[future] = branch_id

            for future in as_completed(futures):
                branch_id = futures[future]
                try:
                    results[branch_id] = future.result()
                except Exception as e:
                    results[branch_id] = {"__error__": str(e)}

    return results


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------

def _execute_single_node(node: Dict[str, Any], pool: VarPool,
                         session_db=None) -> Dict[str, Any]:
    """Dispatch to the right executor based on node type."""
    node_type = node.get("type", "agent")

    if node_type == "script":
        return _run_script_node(node, pool)
    elif node_type == "agent":
        return _run_agent_node(node, pool, session_db=session_db)
    elif node_type == "iteration":
        return _run_iteration_node(node, pool, session_db=session_db)
    elif node_type == "if":
        return _run_if_node(node, pool)
    elif node_type == "template":
        return _run_template_node(node, pool)
    elif node_type == "parallel":
        return _run_parallel_node(node, pool, session_db=session_db)
    else:
        return {"__error__": f"Unknown node type: {node_type}"}


# ---------------------------------------------------------------------------
# Workflow execution engine
# ---------------------------------------------------------------------------

def run_workflow(workflow: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None,
                 session_db=None) -> Dict[str, Any]:
    """Execute a workflow definition.

    The workflow must be a YAML-parsed format with 'nodes' and 'edges'
    (produced by schema.parse_yaml_workflow or load_yaml_file).

    Args:
        workflow: The workflow definition (from schema.parse_yaml_workflow).
        inputs: Optional input variables (override workflow.inputs).
        session_db: Optional SessionDB for agent session persistence.

    Returns:
        {
            "status": "ok" | "error",
            "outputs": { step_id: { key: value } },
            "errors": [ ... ],
            "duration_seconds": float,
            "run_id": str,
        }
    """
    run_id = uuid.uuid4().hex[:12]
    start_time = time.time()

    nodes = workflow.get("nodes", {})
    edges = workflow.get("edges", [])

    # Merge inputs: workflow defaults + runtime overrides
    wf_inputs = workflow.get("inputs", {})
    merged_inputs = {**wf_inputs, **(inputs or {})}

    # Validate
    if not nodes:
        return {
            "status": "error",
            "outputs": {},
            "errors": ["Workflow has no nodes"],
            "duration_seconds": 0,
            "run_id": run_id,
        }

    # Initialize variable pool
    pool = VarPool(inputs=merged_inputs)

    # Determine execution order
    # For if-node workflows, we use step-by-step execution
    # (if nodes redirect to different next steps)
    has_if_nodes = any(n.get("type") == "if" for n in nodes.values())

    if has_if_nodes:
        execution_result = _run_workflow_with_branches(
            workflow, nodes, edges, pool, run_id, session_db
        )
    else:
        execution_result = _run_workflow_linear(
            nodes, edges, pool, run_id, session_db
        )

    # Save run output
    duration = round(time.time() - start_time, 2)
    execution_result["duration_seconds"] = duration
    execution_result["run_id"] = run_id

    _save_run_output(workflow.get("id", "unknown"), run_id, {
        "run_id": run_id,
        "workflow_name": workflow.get("name", "unnamed"),
        "workflow_id": workflow.get("id", ""),
        "status": execution_result["status"],
        "started_at": datetime.fromtimestamp(start_time).isoformat(),
        "duration_seconds": duration,
        "inputs": merged_inputs,
        "outputs": execution_result.get("outputs", {}),
        "final_output": execution_result.get("final_output", {}),
        "errors": execution_result.get("errors", []),
    })

    return execution_result


def _run_workflow_linear(nodes: Dict, edges: List, pool: VarPool,
                         run_id: str, session_db=None) -> Dict[str, Any]:
    """Execute a workflow with no branching (topological sort, linear execution)."""
    try:
        execution_order = _topological_sort(nodes, edges)
    except ValueError as e:
        return {
            "status": "error",
            "outputs": {},
            "errors": [str(e)],
        }

    errors = []
    all_outputs = {}

    for node_id in execution_order:
        node_def = nodes[node_id]
        node_type = node_def.get("type", "agent")

        logger.info("Workflow[%s]: running node '%s' (type=%s)", run_id, node_id, node_type)
        node_start = time.time()

        result = _execute_single_node(node_def, pool, session_db=session_db)

        node_duration = time.time() - node_start
        pool.set_step_output(node_id, result)

        all_outputs[node_id] = {
            "result": result,
            "duration_seconds": round(node_duration, 2),
            "type": node_type,
        }

        if "__error__" in result:
            error_msg = f"Node '{node_id}' failed: {result['__error__']}"
            errors.append(error_msg)
            logger.error("Workflow[%s]: %s", run_id, error_msg)
            break

    # Last node output is the final output
    final_node_id = execution_order[-1] if execution_order else None
    final_output = pool.get_step_output(final_node_id) if final_node_id else {}

    return {
        "status": "error" if errors else "ok",
        "outputs": all_outputs,
        "final_output": final_output,
        "errors": errors,
    }


def _run_workflow_with_branches(workflow: Dict, nodes: Dict, edges: List,
                                pool: VarPool, run_id: str,
                                session_db=None) -> Dict[str, Any]:
    """Execute a workflow with if-node branching.

    Uses step-by-step execution, following the branch chosen by each if node.
    """
    # Build adjacency map
    adjacency = {nid: [] for nid in nodes}
    for src, dst in edges:
        if src in adjacency:
            adjacency[src].append(dst)

    # Find entry points
    entry_points = workflow.get("entry_points", [])
    if not entry_points:
        # Fall back: find nodes with no incoming edges
        in_degree = {nid: 0 for nid in nodes}
        for _, dst in edges:
            if dst in in_degree:
                in_degree[dst] += 1
        entry_points = [nid for nid, deg in in_degree.items() if deg == 0]

    if not entry_points:
        return {
            "status": "error",
            "outputs": {},
            "errors": ["Cannot find entry point in workflow"],
        }

    errors = []
    all_outputs = {}
    visited = set()
    queue = list(entry_points)

    while queue:
        node_id = queue.pop(0)
        if node_id in visited:
            continue
        visited.add(node_id)

        node_def = nodes.get(node_id)
        if node_def is None:
            continue

        node_type = node_def.get("type", "agent")
        logger.info("Workflow[%s]: running node '%s' (type=%s)", run_id, node_id, node_type)

        node_start = time.time()
        result = _execute_single_node(node_def, pool, session_db=session_db)
        node_duration = time.time() - node_start

        pool.set_step_output(node_id, result)
        all_outputs[node_id] = {
            "result": result,
            "duration_seconds": round(node_duration, 2),
            "type": node_type,
        }

        if "__error__" in result:
            error_msg = f"Node '{node_id}' failed: {result['__error__']}"
            errors.append(error_msg)
            logger.error("Workflow[%s]: %s", run_id, error_msg)
            if not workflow.get("continue_on_error", False):
                break

        # Determine next nodes
        if node_type == "if" and "__target__" in result:
            # Follow the chosen branch
            target = result["__target__"]
            if target and target in nodes:
                queue.append(target)
        else:
            # Follow all outgoing edges
            for next_id in adjacency.get(node_id, []):
                if next_id not in visited:
                    queue.append(next_id)

    # Final output = last executed node's output
    final_output = {}
    if visited:
        last_node = list(visited)[-1]
        final_output = pool.get_step_output(last_node) or {}

    return {
        "status": "error" if errors else "ok",
        "outputs": all_outputs,
        "final_output": final_output,
        "errors": errors,
    }


# ---------------------------------------------------------------------------
# Run output persistence
# ---------------------------------------------------------------------------

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
# Helper: get hermes home
# ---------------------------------------------------------------------------

def _get_hermes_home():
    try:
        from hermes_constants import get_hermes_home
        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


# ---------------------------------------------------------------------------
# Public API — convenience wrappers delegating to store
# ---------------------------------------------------------------------------

def create_workflow(workflow_def: Dict[str, Any]) -> Dict[str, Any]:
    """Create a new workflow. Returns the stored workflow with id."""
    from workflow.store import create_workflow as _store_create
    return _store_create(workflow_def)


def get_workflow(wf_id: str) -> Optional[Dict[str, Any]]:
    """Get a workflow by ID."""
    from workflow.store import get_workflow as _store_get
    return _store_get(wf_id)


def list_workflows() -> List[Dict[str, Any]]:
    """List all workflows (metadata only)."""
    from workflow.store import list_workflows as _store_list
    return _store_list()


def update_workflow(wf_id: str, updates: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Update a workflow definition."""
    from workflow.store import update_workflow as _store_update
    return _store_update(wf_id, updates)


def delete_workflow(wf_id: str) -> bool:
    """Delete a workflow and all its run history."""
    from workflow.store import delete_workflow as _store_delete
    return _store_delete(wf_id)


def list_workflow_runs(wf_id: str) -> List[Dict[str, Any]]:
    """List runs for a workflow."""
    from workflow.store import list_workflow_runs as _store_list_runs
    return _store_list_runs(wf_id)


def get_workflow_run(wf_id: str, run_id: str) -> Optional[Dict[str, Any]]:
    """Get a specific run's output."""
    from workflow.store import get_workflow_run as _store_get_run
    return _store_get_run(wf_id, run_id)
