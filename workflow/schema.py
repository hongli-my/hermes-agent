"""
Hermes Workflow Schema — YAML → internal representation parser.

Converts a human-friendly YAML workflow definition (steps-based, sequential)
into the engine's internal graph format (nodes dict + edges list).

Supports node types:
  - script:  run a Python script, stdin JSON → stdout JSON
  - shell:   run an arbitrary shell command, capture stdout/stderr/returncode
  - agent:   run AIAgent with a prompt
  - iteration: loop over a list, execute sub_steps per item
  - if:      conditional branch (then/else)
  - parallel: run branches concurrently
  - template: string template rendering

YAML format example:

    name: My Workflow
    inputs:
      date: "2026-04-24"
    steps:
      - id: collect
        type: script
        script: stocks/collect.py
        input:
          date: "{{inputs.date}}"
      - id: analyze
        type: agent
        prompt: "Analyze: {{steps.collect.output}}"
      - id: check
        type: if
        condition: "len({{steps.analyze.output.items}}) > 0"
        then: report
        else: skip
      - id: report
        type: template
        template: "Found {{steps.analyze.output.count}} items"
      - id: skip
        type: template
        template: "Nothing found"
"""

import re
from typing import Any, Dict, List, Optional, Tuple

import yaml


def parse_yaml_workflow(yaml_text: str) -> Dict[str, Any]:
    """Parse a YAML workflow definition into internal representation.

    Returns:
        {
            "id": str,           # from filename or name field
            "name": str,
            "description": str,
            "inputs": dict,      # default input values
            "nodes": {id: node_def},  # flat node dictionary
            "edges": [[src, dst]],    # edge list for DAG
            "entry_points": [id],     # first step(s) to execute
        }
    """
    raw = yaml.safe_load(yaml_text)
    if not isinstance(raw, dict):
        raise ValueError("YAML workflow must be a mapping")

    name = raw.get("name", "unnamed")
    description = raw.get("description", "")
    inputs = raw.get("inputs", {})
    steps = raw.get("steps", [])

    if not steps:
        raise ValueError("Workflow must have at least one step")

    nodes = {}
    edges = []
    entry_points = []

    # Track step ids for edge building
    step_ids = []

    # First pass: collect all if-node targets to avoid adding wrong seq edges
    if_targets = set()
    for step in steps:
        if step.get("type") == "if":
            then_id = step.get("then")
            else_id = step.get("else")
            if then_id:
                if_targets.add(then_id)
            if else_id:
                if_targets.add(else_id)

    for i, step in enumerate(steps):
        step_id = step.get("id")
        if not step_id:
            raise ValueError(f"Step {i} missing 'id' field")

        if step_id in nodes:
            raise ValueError(f"Duplicate step id: {step_id}")

        step_ids.append(step_id)
        node_type = step.get("type")
        if not node_type:
            raise ValueError(f"Step '{step_id}' missing 'type' field")

        # Build node definition based on type
        node_def = _build_node_def(step_id, step)
        nodes[step_id] = node_def

        # Build edges
        if i == 0:
            entry_points.append(step_id)

        # Sequential edge: previous step → current step
        # Skip if:
        #   - Previous node is 'if' (its then/else edges handle routing)
        #   - Current step is an if-branch target (then/else)
        if i > 0:
            prev_id = step_ids[i - 1]
            prev_type = nodes[prev_id].get("type")

            if prev_type == "if":
                pass  # edges handled by if node
            elif step_id in if_targets:
                pass  # this step is a branch target, if node will connect to it
            else:
                edges.append([prev_id, step_id])

    # Process if-node then/else edges
    _process_if_edges(nodes, edges, step_ids)
    # Process iteration edges
    _process_iteration_edges(nodes, edges)
    # Process parallel edges
    _process_parallel_edges(nodes, edges)

    return {
        "name": name,
        "description": description,
        "inputs": inputs,
        "nodes": nodes,
        "edges": edges,
        "entry_points": entry_points,
    }


def _build_node_def(step_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
    """Build internal node definition from a YAML step."""
    node_type = step["type"]

    base = {
        "type": node_type,
        "id": step_id,
    }

    # Copy common optional fields
    for key in ("comment", "timeout"):
        if key in step:
            base[key] = step[key]

    if node_type == "script":
        base["script"] = step.get("script", "")
        base["input"] = step.get("input", {})
        base["outputs"] = step.get("outputs")
        base["timeout"] = step.get("timeout", 120)
        base["workdir"] = step.get("workdir")  # Working directory for the script

    elif node_type == "shell":
        base["cmd"] = step.get("cmd", "")
        base["workdir"] = step.get("workdir")
        base["timeout"] = step.get("timeout", 60)
        base["env"] = step.get("env") or {}
        # shell=True 默认随 cmd 类型推导（str → True，list → False），
        # 显式设置会覆盖。此处只保留用户值，由 engine 读不到时再推断。
        if "shell" in step:
            base["shell"] = bool(step.get("shell"))
        base["allow_nonzero"] = bool(step.get("allow_nonzero", False))
        base["outputs"] = step.get("outputs")

    elif node_type == "agent":
        base["prompt"] = step.get("prompt", "")
        base["model"] = step.get("model")
        base["provider"] = step.get("provider")
        base["system_prompt"] = step.get("system_prompt")
        base["skills"] = step.get("skills", [])
        base["toolsets"] = step.get("toolsets")
        base["disabled_toolsets"] = step.get("disabled_toolsets")
        base["max_iterations"] = step.get("max_iterations", 30)
        base["output_format"] = step.get("output_format")
        base["outputs"] = step.get("outputs", ["text"])
        base["workdir"] = step.get("workdir")  # Working directory for terminal commands

    elif node_type == "iteration":
        base["items"] = step.get("items", "")
        base["item_var"] = step.get("item_var", "item")
        base["max_concurrent"] = step.get("max_concurrent", 1)
        base["stop_on_error"] = step.get("stop_on_error", False)
        base["sub_steps"] = step.get("sub_steps", [])
        base["outputs"] = step.get("outputs", ["results"])
        base["workdir"] = step.get("workdir")  # Working directory for sub_steps

    elif node_type == "if":
        base["condition"] = step.get("condition", "")
        base["then"] = step.get("then", "")
        base["else"] = step.get("else", "")

    elif node_type == "parallel":
        base["branches"] = step.get("branches", [])
        base["outputs"] = step.get("outputs")

    elif node_type == "template":
        base["template"] = step.get("template", "")
        base["outputs"] = step.get("outputs", ["text"])

    else:
        raise ValueError(f"Unknown node type: {node_type}")

    return base


def _process_if_edges(nodes: Dict, edges: List, step_ids: List[str]):
    """Add then/else edges for if nodes. Remove sequential edges that skip."""
    for step_id, node in nodes.items():
        if node.get("type") != "if":
            continue

        then_id = node.get("then")
        else_id = node.get("else")

        # Add edges from if node to then/else targets
        if then_id and then_id in nodes:
            edges.append([step_id, then_id])
        if else_id and else_id in nodes:
            edges.append([step_id, else_id])

        # Remove any sequential edge from if → next step in YAML order
        # ONLY if the next step is NOT a then/else target
        # (if it is a target, the edge we just added is the then/else edge)
        idx = step_ids.index(step_id)
        if idx + 1 < len(step_ids):
            next_id = step_ids[idx + 1]
            # Only remove if next is not a then/else target
            # (the then/else edges are the correct ones to keep)
            if next_id not in (then_id, else_id):
                edge = [step_id, next_id]
                while edge in edges:
                    edges.remove(edge)

    # Add merge edges: if then/else branches need to rejoin at a common
    # downstream step, add edges from the terminal of each branch to that step.
    _add_if_merge_edges(nodes, edges, step_ids)


def _add_if_merge_edges(nodes: Dict, edges: List, step_ids: List[str]):
    """After an if branch, if both branches need to rejoin at a downstream
    step, add edges from the terminal of each branch to that merge step.

    Strategy:
      1. For each if node, find the then/else target step ids.
      2. Find the "merge point" — the first step after the if node in YAML
         order that is NOT a then/else target or a descendant of one.
      3. For each branch (then/else), find its terminal node (the last node
         in the branch before the merge point).
      4. Add edge: terminal_node → merge_point (if not already present).

    If there's no merge point (branches go to the end), no edges are added.
    """
    for i, step_id in enumerate(step_ids):
        node = nodes.get(step_id, {})
        if node.get("type") != "if":
            continue

        then_id = node.get("then")
        else_id = node.get("else")

        # Collect all step ids that belong to the then/else branches
        branch_steps = set()
        for branch_id in [then_id, else_id]:
            if not branch_id or branch_id not in nodes:
                continue
            branch_steps.add(branch_id)
            # Walk downstream from branch_id via existing edges
            queue = [branch_id]
            while queue:
                current = queue.pop(0)
                for src, dst in edges:
                    if src == current and dst not in branch_steps:
                        branch_steps.add(dst)
                        queue.append(dst)

        # Find merge point: first step after the if node (in YAML order)
        # that is NOT in the branch_steps set
        merge_id = None
        for j in range(i + 1, len(step_ids)):
            if step_ids[j] not in branch_steps:
                merge_id = step_ids[j]
                break

        if not merge_id:
            # No merge point — branches go to the end of the workflow
            continue

        # For each branch, find the terminal node (the node in the branch
        # that has no outgoing edge leading to another branch step)
        for branch_id in [then_id, else_id]:
            if not branch_id or branch_id not in nodes:
                continue

            # Walk the branch to find the terminal
            terminal = branch_id
            visited = set()
            while True:
                if terminal in visited:
                    break
                visited.add(terminal)
                # Find downstream nodes
                downstream = [dst for src, dst in edges if src == terminal]
                if not downstream:
                    break
                # Follow the first downstream that's in branch_steps
                found = False
                for d in downstream:
                    if d in branch_steps:
                        terminal = d
                        found = True
                        break
                if not found:
                    break

            # Add edge from terminal to merge if not already there
            if terminal != merge_id and terminal in branch_steps:
                edge = [terminal, merge_id]
                if edge not in edges:
                    edges.append(edge)


def _process_iteration_edges(nodes: Dict, edges: List):
    """Flatten iteration sub_steps into the node graph."""
    for step_id, node in nodes.items():
        if node.get("type") != "iteration":
            continue

        sub_steps = node.get("sub_steps", [])
        if not sub_steps:
            continue

        # Sub-steps are executed internally by the iteration executor.
        # They are NOT separate nodes in the graph — they are embedded
        # inside the iteration node definition.
        # No edges needed — the iteration executor handles execution order.


def _process_parallel_edges(nodes: Dict, edges: List):
    """Process parallel branches. They run concurrently, not as graph nodes."""
    # Parallel branches are handled by the parallel executor internally.
    # No additional edges needed.


def load_yaml_file(path: str) -> Dict[str, Any]:
    """Load and parse a YAML workflow file."""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    wf = parse_yaml_workflow(content)
    # Use filename (without extension) as ID if not set
    if not wf.get("id"):
        import os
        basename = os.path.splitext(os.path.basename(path))[0]
        wf["id"] = basename
    return wf


def validate_workflow(wf: Dict[str, Any]) -> List[str]:
    """Validate a parsed workflow. Returns list of error messages."""
    errors = []

    nodes = wf.get("nodes", {})
    edges = wf.get("edges", [])

    if not nodes:
        errors.append("Workflow has no nodes")
        return errors

    # Check all edge endpoints exist
    for src, dst in edges:
        if src not in nodes:
            errors.append(f"Edge source '{src}' not found in nodes")
        if dst not in nodes:
            errors.append(f"Edge target '{dst}' not found in nodes")

    # Check if node targets exist
    for step_id, node in nodes.items():
        if node.get("type") == "if":
            then_id = node.get("then")
            else_id = node.get("else")
            if then_id and then_id not in nodes:
                errors.append(f"If node '{step_id}': then target '{then_id}' not found")
            if else_id and else_id not in nodes:
                errors.append(f"If node '{step_id}': else target '{else_id}' not found")

        if node.get("type") == "script":
            if not node.get("script"):
                errors.append(f"Script node '{step_id}' missing 'script' field")

        if node.get("type") == "shell":
            cmd = node.get("cmd")
            if cmd is None or cmd == "" or (isinstance(cmd, list) and not cmd):
                errors.append(f"Shell node '{step_id}' missing 'cmd' field")
            elif not isinstance(cmd, (str, list)):
                errors.append(f"Shell node '{step_id}': 'cmd' must be string or list, got {type(cmd).__name__}")

        if node.get("type") == "agent":
            if not node.get("prompt"):
                errors.append(f"Agent node '{step_id}' missing 'prompt' field")

        if node.get("type") == "iteration":
            if not node.get("items"):
                errors.append(f"Iteration node '{step_id}' missing 'items' field")
            if not node.get("sub_steps"):
                errors.append(f"Iteration node '{step_id}' missing 'sub_steps'")

    # Check for cycles (simple DFS)
    adjacency = {nid: [] for nid in nodes}
    for src, dst in edges:
        if src in adjacency:
            adjacency[src].append(dst)

    visited = set()
    rec_stack = set()

    def _has_cycle(node_id):
        visited.add(node_id)
        rec_stack.add(node_id)
        for neighbor in adjacency.get(node_id, []):
            if neighbor not in visited:
                if _has_cycle(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        rec_stack.remove(node_id)
        return False

    for nid in nodes:
        if nid not in visited:
            if _has_cycle(nid):
                errors.append("Workflow contains a cycle")
                break

    return errors
