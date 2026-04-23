"""Quick smoke test for workflow engine (no agent calls)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow.engine import (
    _topological_sort,
    _parse_script_output,
)
from workflow.store import (
    create_workflow,
    create_workflow_from_yaml,
    get_workflow,
    list_workflows,
    delete_workflow,
)
from workflow.schema import parse_yaml_workflow, validate_workflow
from workflow.renderer import VarPool, render, evaluate_condition


def test_topological_sort():
    nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
    edges = [["a", "b"], ["b", "c"], ["c", "d"]]
    order = _topological_sort(nodes, edges)
    assert order == ["a", "b", "c", "d"], f"Expected [a,b,c,d], got {order}"
    print("✓ topological_sort (linear)")


def test_topological_sort_branch():
    nodes = {"a": {}, "b": {}, "c": {}, "d": {}}
    edges = [["a", "b"], ["a", "c"], ["b", "d"], ["c", "d"]]
    order = _topological_sort(nodes, edges)
    assert order.index("a") < order.index("b")
    assert order.index("a") < order.index("c")
    assert order.index("b") < order.index("d")
    assert order.index("c") < order.index("d")
    print("✓ topological_sort (branch+merge)")


def test_topological_sort_cycle():
    nodes = {"a": {}, "b": {}, "c": {}}
    edges = [["a", "b"], ["b", "c"], ["c", "a"]]
    try:
        _topological_sort(nodes, edges)
        assert False, "Should have raised ValueError"
    except ValueError:
        print("✓ topological_sort (cycle detected)")


def test_parse_script_output():
    # JSON dict
    out = _parse_script_output('{"items": [1,2]}')
    assert out == {"items": [1, 2]}

    # JSON array
    out = _parse_script_output('[1,2,3]')
    assert out == {"output": [1, 2, 3]}

    # Plain text
    out = _parse_script_output("hello world")
    assert out == {"output": "hello world"}

    print("✓ parse_script_output")


def test_yaml_parsing():
    yaml_text = """
name: Test Workflow
description: A simple test
inputs:
  date: "2026-04-24"
steps:
  - id: step1
    type: script
    script: test.py
    input:
      date: "{{inputs.date}}"
  - id: step2
    type: template
    template: "Result: {{steps.step1.output}}"
"""
    wf = parse_yaml_workflow(yaml_text)
    assert wf["name"] == "Test Workflow"
    assert "step1" in wf["nodes"]
    assert "step2" in wf["nodes"]
    assert len(wf["edges"]) >= 1
    errors = validate_workflow(wf)
    assert not errors, f"Validation errors: {errors}"
    print("✓ YAML parsing + validation")


def test_render():
    pool = VarPool(inputs={"date": "2026-04-24", "threshold": 1.5})
    pool.set_step_output("collect", {"output": {"stocks": [{"symbol": "sz002714", "name": "牧原股份"}]}})

    result = render("Today is {{inputs.date}}", pool)
    assert result == "Today is 2026-04-24"

    result2 = render("{{steps.collect.output.stocks}}", pool)
    assert isinstance(result2, list)
    assert result2[0]["symbol"] == "sz002714"

    print("✓ Variable rendering")


def test_evaluate_condition():
    pool = VarPool()
    pool.set_step_output("filter", {"output": {"stocks": [1, 2, 3]}})
    assert evaluate_condition("len({{steps.filter.output.stocks}}) > 0", pool) is True

    pool2 = VarPool()
    pool2.set_step_output("filter", {"output": {"stocks": []}})
    assert evaluate_condition("len({{steps.filter.output.stocks}}) > 0", pool2) is False

    print("✓ Condition evaluation")


def test_crud():
    wf_id = "test_wf_" + __import__("uuid").uuid4().hex[:6]
    # Create from YAML-parsed dict
    wf_def = parse_yaml_workflow(f"""
name: CRUD Test
steps:
  - id: hello
    type: script
    script: dummy.py
""")
    wf_def["id"] = wf_id
    wf = create_workflow(wf_def)
    assert wf["id"] == wf_id

    fetched = get_workflow(wf_id)
    assert fetched is not None
    assert fetched["name"] == "CRUD Test"

    wfs = list_workflows()
    assert any(w["id"] == wf_id for w in wfs)

    delete_workflow(wf_id)
    assert get_workflow(wf_id) is None
    print("✓ CRUD operations")


if __name__ == "__main__":
    test_topological_sort()
    test_topological_sort_branch()
    test_topological_sort_cycle()
    test_parse_script_output()
    test_yaml_parsing()
    test_render()
    test_evaluate_condition()
    test_crud()
    print("\n✅ All tests passed!")
