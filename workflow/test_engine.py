"""Quick smoke test for workflow engine (no agent calls)."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from workflow.engine import (
    _topological_sort,
    resolve_variables,
    _parse_script_output,
    create_workflow,
    get_workflow,
    list_workflows,
    delete_workflow,
)


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


def test_resolve_variables():
    var_pool = {
        "node_a": {"items": ["x", "y"]},
        "node_b": {"text": "hello"},
    }
    result = resolve_variables("Process {{node_a.items}} and {{node_b.text}}", var_pool)
    assert '["x", "y"]' in result
    assert "hello" in result
    print("✓ resolve_variables")


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


def test_crud():
    wf_id = "test_wf_" + __import__("uuid").uuid4().hex[:6]
    wf = create_workflow({"id": wf_id, "name": "Test Workflow", "nodes": {"a": {"type": "agent", "prompt": "hi"}}, "edges": []})
    assert wf["id"] == wf_id

    fetched = get_workflow(wf_id)
    assert fetched is not None
    assert fetched["name"] == "Test Workflow"

    wfs = list_workflows()
    assert any(w["id"] == wf_id for w in wfs)

    delete_workflow(wf_id)
    assert get_workflow(wf_id) is None
    print("✓ CRUD operations")


if __name__ == "__main__":
    test_topological_sort()
    test_topological_sort_branch()
    test_topological_sort_cycle()
    test_resolve_variables()
    test_parse_script_output()
    test_crud()
    print("\n✅ All tests passed!")
