"""
Hermes Workflow Renderer — variable resolution engine.

Replaces {{...}} placeholders in strings and dicts with actual values
from the execution context.

Supported variable patterns:
  {{inputs.key}}          — workflow input parameters
  {{steps.node_id.output.key}}  — output of a completed step
  {{steps.node_id.output}}      — full output of a step (as JSON)
  {{env.VAR_NAME}}        — environment variable
  {{item}}                — current iteration item (inside iteration body)
  {{__iter__.item_var}}   — iteration metadata (item, index, total)

Variable resolution is deep — it processes all string values in nested
dicts and lists, not just top-level strings.
"""

import json
import os
import re
from typing import Any, Dict, List, Optional, Union


# Pattern: {{namespace.key1.key2...}}
_VAR_PATTERN = re.compile(r'\{\{([\w.]+)\}\}')

# Shorthand patterns
_INPUTS_PATTERN = re.compile(r'\{\{inputs\.(\w+)\}\}')
_ENV_PATTERN = re.compile(r'\{\{env\.(\w+)\}\}')
_STEPS_PATTERN = re.compile(r'\{\{steps\.(\w+)\.output(?:\.(\w+))?\}\}')
_ITER_PATTERN = re.compile(r'\{\{__iter__\.(\w+)\}\}')
# Bare variable inside iteration body: {{stock}}, {{item}}, etc.
# These are replaced with __iter__.<item_var> values


class VarPool:
    """Execution-time variable pool. Stores outputs of completed steps."""

    def __init__(self, inputs: Optional[Dict[str, Any]] = None):
        self.inputs = inputs or {}
        self.steps: Dict[str, Any] = {}  # step_id → output dict
        self._iter_context: Optional[Dict[str, Any]] = None

    def set_step_output(self, step_id: str, output: Dict[str, Any]):
        """Store a completed step's output."""
        self.steps[step_id] = output

    def get_step_output(self, step_id: str) -> Optional[Dict[str, Any]]:
        """Get a step's output, or None if not yet executed."""
        return self.steps.get(step_id)

    def push_iter_context(self, item_var: str, item: Any, index: int, total: int):
        """Enter an iteration context."""
        self._iter_context = {
            item_var: item,
            "index": index,
            "total": total,
        }

    def pop_iter_context(self):
        """Leave an iteration context."""
        self._iter_context = None

    @property
    def iter_context(self) -> Optional[Dict[str, Any]]:
        return self._iter_context


def render(template: Union[str, Dict, List, Any], pool: VarPool) -> Any:
    """Render all {{...}} variables in a template using the var pool.

    Handles strings, dicts, and lists recursively.
    Non-string types are returned as-is unless they're containers.

    After string substitution, if the result looks like a JSON value
    (list, dict, number, bool), it is automatically parsed back to
    the corresponding Python type. This ensures that:
      "stocks": "{{steps.collect.output.stocks}}"
    becomes:
      "stocks": [{"symbol": "sz002714", ...}]
    instead of:
      "stocks": '[{"symbol": "sz002714", ...}]'
    """
    if isinstance(template, str):
        return _render_string(template, pool)
    elif isinstance(template, dict):
        return {k: render(v, pool) for k, v in template.items()}
    elif isinstance(template, list):
        return [render(item, pool) for item in template]
    else:
        return template


def _render_string(text: str, pool: VarPool) -> Any:
    """Replace all ``{{...}}`` patterns in *text*.

    Return-type rules (intentional, to avoid surprising coercions):

    1. If the whole input is a single variable reference (optionally
       surrounded by whitespace) — e.g. ``"{{steps.x.output.stocks}}"`` —
       return the resolved value **as-is**, preserving its original type
       (``list`` / ``dict`` / ``int`` / ``bool`` / ...).  This is the
       common "pass-through" case used to forward a structured output
       from one step to the next.

    2. Otherwise (mixed text / multi-var template / prompt body) the
       result is returned as a ``str``.  Non-string substitutions are
       JSON-encoded so they embed cleanly, but we do **not** attempt to
       ``json.loads`` the final string back into a Python object — doing
       so silently mis-typed legitimate prose that merely happened to
       start with ``[``, ``{`` or ``"`` (Markdown lists, prompts that
       describe JSON, narrative quotes, etc.).

    Unresolved placeholders (``{{missing}}``) are left intact so the
    caller can detect them instead of silently dropping content.
    """
    # Case 1 — whole string is a single variable reference.
    single_var = _VAR_PATTERN.fullmatch(text.strip())
    if single_var:
        value = _resolve_expr(single_var.group(1), pool)
        if value is not None and not isinstance(value, str):
            return value  # pass through list/dict/number/bool unchanged
        if value is not None:
            return value  # raw string value, no further coercion

    # Case 2 — mixed content; do placeholder substitution only.
    def _replacer(match):
        expr = match.group(1)
        value = _resolve_expr(expr, pool)
        if value is None:
            return match.group(0)  # leave unresolved so caller can tell
        if isinstance(value, str):
            return value
        # JSON-encode structured values so they embed without breaking
        # quoting in the surrounding prose.
        return json.dumps(value, ensure_ascii=False)

    return _VAR_PATTERN.sub(_replacer, text)


def _resolve_expr(expr: str, pool: VarPool) -> Any:
    """Resolve a variable expression like 'steps.collect.output.stocks'."""
    parts = expr.split(".")

    # {{inputs.key}}
    if parts[0] == "inputs" and len(parts) >= 2:
        key = ".".join(parts[1:])
        return _deep_get(pool.inputs, key)

    # {{steps.node_id.output.key}} or {{steps.node_id.output}}
    if parts[0] == "steps" and len(parts) >= 3:
        step_id = parts[1]
        step_output = pool.get_step_output(step_id)
        if step_output is None:
            return None
        if parts[2] == "output":
            if len(parts) == 3:
                # {{steps.X.output}} — full output
                # If step_output has an "output" key, unwrap it;
                # otherwise return the whole step_output dict.
                if isinstance(step_output, dict) and "output" in step_output:
                    return step_output["output"]
                return step_output
            else:
                # {{steps.X.output.key.subkey...}}
                # step_output may be {"output": {"key": val}} or {"key": val}
                # Try with "output." prefix first, then without
                key = ".".join(parts[3:])
                result = _deep_get(step_output, "output." + key)
                if result is not None:
                    return result
                return _deep_get(step_output, key)
        else:
            # {{steps.X.key...}} — direct key access on step output
            key = ".".join(parts[2:])
            return _deep_get(step_output, key)

    # {{env.VAR}}
    if parts[0] == "env" and len(parts) >= 2:
        var_name = ".".join(parts[1:])
        return os.environ.get(var_name)

    # {{__iter__.item_var}} or {{__iter__.index}} etc.
    if parts[0] == "__iter__" and len(parts) >= 2:
        if pool.iter_context is None:
            return None
        key = ".".join(parts[1:])
        return _deep_get(pool.iter_context, key)

    # Bare variable or dotted access: check iteration context first, then steps
    # e.g. {{stock}} or {{stock.name}} inside iteration body
    var_name = parts[0]
    # Check iteration context
    if pool.iter_context and var_name in pool.iter_context:
        obj = pool.iter_context[var_name]
        if len(parts) == 1:
            return obj
        else:
            # Dotted access: stock.name, stock.symbol, etc.
            return _deep_get(obj, ".".join(parts[1:]))

    # Check steps (could be a step id → full output)
    if len(parts) == 1 and var_name in pool.steps:
        return pool.steps[var_name]

    return None


def _deep_get(obj: Any, key_path: str) -> Any:
    """Navigate a nested dict/list by dot-separated key path.

    Supports:
      "stocks"          → obj["stocks"]
      "stocks.0.name"   → obj["stocks"][0]["name"]
      "result.text"     → obj["result"]["text"]
    """
    parts = key_path.split(".")
    current = obj

    for part in parts:
        if current is None:
            return None

        # Try dict key
        if isinstance(current, dict):
            if part in current:
                current = current[part]
            else:
                return None
        # Try list index
        elif isinstance(current, list):
            try:
                idx = int(part)
                current = current[idx]
            except (ValueError, IndexError):
                return None
        else:
            return None

    return current


def evaluate_condition(condition: str, pool: VarPool) -> bool:
    """Evaluate a condition string with variable substitution.

    Supports:
      "len({{steps.X.output.items}}) > 0"
      "{{steps.X.output.count}} >= 5"
      "{{steps.X.output.success}} == true"

    The condition is first rendered (variables replaced), then evaluated
    as a Python expression in a restricted namespace.

    Variable values that are lists/dicts are serialized as JSON, so
    `len([1,2,3])` becomes `len([1, 2, 3])` which Python can eval.
    """
    rendered = _render_string(condition, pool)

    # Restricted evaluation namespace
    safe_builtins = {
        "len": len,
        "abs": abs,
        "int": int,
        "float": float,
        "str": str,
        "bool": bool,
        "True": True,
        "False": False,
        "None": None,
        "true": True,
        "false": False,
        "null": None,
        "__builtins__": {},
    }

    try:
        result = eval(rendered, safe_builtins, {})
        return bool(result)
    except Exception:
        # If evaluation fails, log and return False
        return False
