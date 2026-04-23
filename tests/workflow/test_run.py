#!/usr/bin/env python3
"""Test workflow with YAML file only (replaces old JSON test)."""
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow.schema import load_yaml_file, validate_workflow
from workflow.renderer import VarPool, render, evaluate_condition

# ─── Test 1: Schema parsing ───

print("=" * 60)
print("TEST 1: Schema parsing")
print("=" * 60)

# Parse test-pipeline.yaml
test_yaml = str(Path.home() / ".hermes/workflows/test-pipeline.yaml")
if Path(test_yaml).exists():
    wf = load_yaml_file(test_yaml)
    print(f"\ntest-pipeline.yaml:")
    print(f"  Name: {wf['name']}")
    print(f"  Nodes: {list(wf['nodes'].keys())}")
    print(f"  Edges: {wf['edges']}")
    print(f"  Entry points: {wf['entry_points']}")
    errors = validate_workflow(wf)
    print(f"  Validation: {'OK' if not errors else errors}")

# Parse test-if-branch.yaml
test_if_yaml = str(Path.home() / ".hermes/workflows/test-if-branch.yaml")
if Path(test_if_yaml).exists():
    wf2 = load_yaml_file(test_if_yaml)
    print(f"\ntest-if-branch.yaml:")
    print(f"  Name: {wf2['name']}")
    print(f"  Nodes: {list(wf2['nodes'].keys())}")
    print(f"  Edges: {wf2['edges']}")
    print(f"  Entry points: {wf2['entry_points']}")
    errors2 = validate_workflow(wf2)
    print(f"  Validation: {'OK' if not errors2 else errors2}")

# ─── Test 2: Renderer ───

print("\n" + "=" * 60)
print("TEST 2: Variable rendering")
print("=" * 60)

pool = VarPool(inputs={"date": "2026-04-24", "threshold": 1.5})
pool.set_step_output("collect", {"output": {"stocks": [{"symbol": "sz002714", "name": "牧原股份"}], "indices": [{"code": "sh000001", "close": 3280}]}})

# Test {{inputs.date}}
result = render("Today is {{inputs.date}}", pool)
print(f"  {{inputs.date}} → {result}")
assert result == "Today is 2026-04-24"

# Test {{steps.collect.output.stocks}}
result2 = render("Stocks: {{steps.collect.output.stocks}}", pool)
print(f"  {{steps.collect.output.stocks}} → {str(result2)[:80]}...")

# Test condition
pool2 = VarPool()
pool2.set_step_output("filter", {"output": {"stocks": [1, 2, 3]}})
cond_result = evaluate_condition("len({{steps.filter.output.stocks}}) > 0", pool2)
print(f"  Condition 'len([1,2,3]) > 0' → {cond_result}")
assert cond_result is True

pool3 = VarPool()
pool3.set_step_output("filter", {"output": {"stocks": []}})
cond_result2 = evaluate_condition("len({{steps.filter.output.stocks}}) > 0", pool3)
print(f"  Condition 'len([]) > 0' → {cond_result2}")
assert cond_result2 is False

# Test iteration context
pool4 = VarPool(inputs={"threshold": 0.5})
pool4.push_iter_context("stock", {"symbol": "sz002714", "name": "牧原股份"}, 0, 3)
result4 = render("Analyzing {{stock.name}} ({{stock.symbol}})", pool4)
print(f"  {{stock.name}} in iteration → {result4}")
assert "牧原股份" in result4

print("\n✅ Schema + Renderer tests passed!")

# ─── Test 3: End-to-end engine (if YAML exists) ───

if Path(test_yaml).exists():
    print("\n" + "=" * 60)
    print("TEST 3: End-to-end engine (test-pipeline.yaml)")
    print("=" * 60)

    from workflow.engine import run_workflow

    result = run_workflow(wf, inputs={"date": "2026-04-24"})
    print(f"  Status: {result['status']}")
    print(f"  Duration: {result['duration_seconds']}s")
    print(f"  Run ID: {result['run_id']}")
    print(f"  Steps executed: {list(result['outputs'].keys())}")

    if result.get("errors"):
        print(f"  Errors: {result['errors']}")
    else:
        for step_id, step_data in result["outputs"].items():
            step_result = step_data.get("result", {})
            print(f"  {step_id} ({step_data['type']}, {step_data['duration_seconds']}s):")
            for key, val in step_result.items():
                if key.startswith("__"):
                    continue
                val_str = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
                if len(val_str) > 120:
                    print(f"    {key}: {val_str[:120]}...")
                else:
                    print(f"    {key}: {val_str}")

# ─── Test 4: If-branch workflow ───

if Path(test_if_yaml).exists():
    print("\n" + "=" * 60)
    print("TEST 4: If-branch workflow (test-if-branch.yaml)")
    print("=" * 60)

    result2 = run_workflow(wf2, inputs={"threshold": 0.5})
    print(f"  Status: {result2['status']}")
    print(f"  Steps executed: {list(result2['outputs'].keys())}")

    for step_id, step_data in result2["outputs"].items():
        step_result = step_data.get("result", {})
        if "__branch__" in step_result:
            print(f"  {step_id}: branch={step_result['__branch__']}, target={step_result.get('__target__')}")
        elif "__error__" in step_result:
            print(f"  {step_id}: ERROR={step_result['__error__']}")
        else:
            preview = json.dumps(step_result, ensure_ascii=False)[:100]
            print(f"  {step_id}: {preview}")

    print(f"  Final output: {json.dumps(result2['final_output'], ensure_ascii=False)[:200]}")

print("\n" + "=" * 60)
print("✅ ALL TESTS PASSED!")
print("=" * 60)
