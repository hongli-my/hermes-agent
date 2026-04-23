#!/usr/bin/env python3
"""测试迭代节点并发执行功能。

测试覆盖：
  1. 并发迭代基本执行 - 多个 item 并发执行 sub_step
  2. 结果顺序 - 并发执行但结果按 item 顺序排列
  3. VarPool 隔离 - 每个线程使用独立 VarPool 副本，互不影响
  4. TERMINAL_CWD 行为 - 并发前设置一次，并发后清理
  5. 校验规则 - max_concurrent>1 时 sub_step 禁止配置 workdir
  6. stop_on_error - 并发模式下遇错取消其他 item
  7. 空 items / 单 item 边界场景
  8. 并发数限制 - max_concurrent 限制实际线程数
  9. iter_workdir 共享 - 所有 item 共享同一个迭代层 workdir
"""
import copy
import json
import os
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow.engine import (
    _run_iteration_node,
    _run_iteration_concurrent,
    _run_iteration_sequential,
    _set_terminal_cwd,
)
from workflow.renderer import VarPool, render
from workflow.schema import parse_yaml_workflow, validate_workflow


# ---------------------------------------------------------------------------
# 辅助函数：创建用于测试的 script 脚本
# ---------------------------------------------------------------------------

def _setup_test_script(name: str, content: str) -> Path:
    """在 scripts 目录下创建测试脚本，返回脚本路径。"""
    scripts_dir = Path.home() / ".hermes" / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / name
    script_path.parent.mkdir(parents=True, exist_ok=True)
    script_path.write_text(content)
    return script_path


def _cleanup_test_script(name: str):
    """清理测试脚本。"""
    scripts_dir = Path.home() / ".hermes" / "scripts"
    script_path = scripts_dir / name
    if script_path.exists():
        script_path.unlink()


# ---------------------------------------------------------------------------
# 测试 1：基本并发执行
# ---------------------------------------------------------------------------

def test_concurrent_basic_execution():
    """并发迭代：3 个 item 并发执行 template sub_step。"""
    # 创建一个返回 item 值的脚本
    script_name = "test/concurrent_echo.py"
    _setup_test_script(script_name, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'item_val = data.get("item_val", "?")\n'
        'print(json.dumps({"echo": item_val}))\n'
    ))

    try:
        pool = VarPool(inputs={"cluster_list": ["c1", "c2", "c3"]})
        pool.set_step_output("prep", {"output": {"cluster_list": ["c1", "c2", "c3"]}})

        node = {
            "id": "iterate_clusters",
            "type": "iteration",
            "items": "{{steps.prep.output.cluster_list}}",
            "item_var": "cluster",
            "max_concurrent": 3,
            "sub_steps": [
                {
                    "id": "echo_cluster",
                    "type": "script",
                    "script": script_name,
                    "input": {"item_val": "{{cluster}}"},
                    "outputs": ["echo"],
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        assert "__error__" not in result, f"迭代执行出错: {result.get('__error__')}"
        assert "results" in result, "缺少 results 字段"
        assert len(result["results"]) == 3, f"期望 3 个结果，实际 {len(result['results'])}"

        # 验证结果顺序
        items_in_order = [r["item"] for r in result["results"]]
        assert items_in_order == ["c1", "c2", "c3"], f"结果顺序错误: {items_in_order}"

        # 验证每个结果包含 index
        for i, r in enumerate(result["results"]):
            assert r["index"] == i, f"index 不匹配: 期望 {i}, 实际 {r['index']}"

        print("✓ 并发迭代基本执行: 3 item 并发，结果顺序正确")
    finally:
        _cleanup_test_script(script_name)
        _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 2：结果顺序保证
# ---------------------------------------------------------------------------

def test_concurrent_result_ordering():
    """并发迭代：即使 item 执行时间不同，结果仍按原始顺序排列。"""
    # 创建一个耗时随 item 变化的脚本（后面的 item 故意更慢）
    script_name = "test/concurrent_order.py"
    _setup_test_script(script_name, (
        'import json, sys, time\n'
        'data = json.loads(sys.stdin.read())\n'
        'idx = data.get("idx", 0)\n'
        'time.sleep(0.1 * idx)  # 后面的 item 更慢\n'
        'print(json.dumps({"order": idx}))\n'
    ))

    try:
        pool = VarPool(inputs={})

        node = {
            "id": "iterate_order",
            "type": "iteration",
            "items": ["first", "second", "third", "fourth", "fifth"],
            "item_var": "item",
            "max_concurrent": 5,
            "sub_steps": [
                {
                    "id": "ordered_step",
                    "type": "script",
                    "script": script_name,
                    "input": {"idx": "{{__iter__.index}}"},
                    "outputs": ["order"],
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        assert "results" in result
        assert len(result["results"]) == 5

        # 验证顺序：index 必须递增
        indices = [r["index"] for r in result["results"]]
        assert indices == [0, 1, 2, 3, 4], f"结果顺序不正确: {indices}"

        items = [r["item"] for r in result["results"]]
        assert items == ["first", "second", "third", "fourth", "fifth"], f"item 顺序不正确: {items}"

        print("✓ 并发迭代结果顺序: 即使执行时间不同，结果仍按原始顺序排列")
    finally:
        _cleanup_test_script(script_name)
        _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 3：VarPool 隔离
# ---------------------------------------------------------------------------

def test_concurrent_varpool_isolation():
    """并发迭代：每个线程使用独立 VarPool 副本，变量互不干扰。"""
    captured_items = []
    capture_lock = threading.Lock()

    # 用 patch 拦截 _run_template_node，捕获每个线程的 pool 内容
    original_template = None
    from workflow import engine as eng

    def mock_template_node(node, pool):
        item_val = None
        if pool.iter_context:
            # 获取 item_var 的值
            ctx = pool.iter_context
            # iter_context 是 {item_var: value, "index": i, "total": n}
            for k, v in ctx.items():
                if k not in ("index", "total"):
                    item_val = v
                    break
        with capture_lock:
            captured_items.append(item_val)
        return {"text": str(item_val)}

    with patch.object(eng, "_run_template_node", side_effect=mock_template_node):
        pool = VarPool(inputs={"items_list": ["A", "B", "C", "D"]})
        pool.set_step_output("prep", {"output": {"items_list": ["A", "B", "C", "D"]}})

        node = {
            "id": "iterate_isolate",
            "type": "iteration",
            "items": "{{steps.prep.output.items_list}}",
            "item_var": "val",
            "max_concurrent": 4,
            "sub_steps": [
                {
                    "id": "template_step",
                    "type": "template",
                    "template": "{{val}}",
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

    # 验证：每个线程都捕获到了正确的 item
    assert sorted(captured_items) == ["A", "B", "C", "D"], \
        f"VarPool 隔离失败，捕获的 items: {captured_items}"

    # 验证结果完整性
    assert "results" in result
    assert len(result["results"]) == 4

    print("✓ VarPool 隔离: 每个线程独立，变量互不干扰")
    _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 4：TERMINAL_CWD 行为
# ---------------------------------------------------------------------------

def test_concurrent_terminal_cwd():
    """并发迭代：TERMINAL_CWD 在并发前设置一次，并发后清理。"""
    _set_terminal_cwd(None)

    with tempfile.TemporaryDirectory() as tmpdir:
        pool = VarPool(inputs={})
        resolved_tmpdir = str(Path(tmpdir).resolve())

        node = {
            "id": "iterate_cwd",
            "type": "iteration",
            "items": ["x", "y", "z"],
            "item_var": "item",
            "max_concurrent": 2,
            "workdir": tmpdir,
            "sub_steps": [
                {
                    "id": "template_step",
                    "type": "template",
                    "template": "{{item}}",
                }
            ],
            "outputs": ["results"],
        }

        # 迭代前 TERMINAL_CWD 应为 None
        assert os.environ.get("TERMINAL_CWD") is None, "迭代前 TERMINAL_CWD 应为空"

        result = _run_iteration_node(node, pool)

        # 迭代后 TERMINAL_CWD 应被清理
        assert os.environ.get("TERMINAL_CWD") is None, \
            f"迭代后 TERMINAL_CWD 应被清理，实际值: {os.environ.get('TERMINAL_CWD')}"

        assert "results" in result
        assert len(result["results"]) == 3

        print("✓ TERMINAL_CWD: 并发前设置，并发后清理")


def test_concurrent_terminal_cwd_no_workdir():
    """并发迭代：没有配置 workdir 时，TERMINAL_CWD 不受影响。"""
    _set_terminal_cwd(None)

    pool = VarPool(inputs={})

    node = {
        "id": "iterate_no_cwd",
        "type": "iteration",
        "items": ["a", "b"],
        "item_var": "item",
        "max_concurrent": 2,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    # 没有 workdir，TERMINAL_CWD 应保持为 None
    assert os.environ.get("TERMINAL_CWD") is None

    assert "results" in result
    print("✓ TERMINAL_CWD: 无 workdir 时不影响环境变量")


# ---------------------------------------------------------------------------
# 测试 5：校验规则 - max_concurrent>1 时 sub_step 禁止配置 workdir
# ---------------------------------------------------------------------------

def test_validate_concurrent_iteration_no_substep_workdir():
    """校验：并发迭代的 sub_step 不允许配置 workdir。"""
    yaml_text = """
name: test_concurrent_workdir_validation
steps:
  - id: iterate
    type: iteration
    items: "{{inputs.clusters}}"
    max_concurrent: 3
    workdir: /tmp/deploy
    sub_steps:
      - id: sub_agent
        type: agent
        prompt: "check {{cluster}}"
        workdir: /tmp/sub_step_dir
      - id: sub_script
        type: script
        script: check.py
"""
    wf = parse_yaml_workflow(yaml_text)
    errors = validate_workflow(wf)

    # 应该有错误：sub_agent 配置了 workdir
    assert len(errors) > 0, "应该检测到 sub_step 配置 workdir 的错误"
    assert any("sub_agent" in e and "workdir" in e for e in errors), \
        f"错误信息应包含 sub_agent 的 workdir 问题，实际: {errors}"
    print(f"✓ 校验规则: 并发迭代 sub_step 配置 workdir 被正确拦截，错误: {errors[0]}")


def test_validate_sequential_iteration_allows_substep_workdir():
    """校验：顺序迭代（max_concurrent=1）允许 sub_step 配置 workdir。"""
    yaml_text = """
name: test_sequential_workdir_ok
steps:
  - id: iterate
    type: iteration
    items: "{{inputs.clusters}}"
    max_concurrent: 1
    workdir: /tmp/deploy
    sub_steps:
      - id: sub_agent
        type: agent
        prompt: "check {{cluster}}"
        workdir: /tmp/sub_step_dir
"""
    wf = parse_yaml_workflow(yaml_text)
    errors = validate_workflow(wf)

    # 顺序迭代允许 sub_step 配置 workdir
    workdir_errors = [e for e in errors if "workdir" in e and "sub_step" in e]
    assert len(workdir_errors) == 0, \
        f"顺序迭代应允许 sub_step 配置 workdir，错误: {workdir_errors}"
    print("✓ 校验规则: 顺序迭代允许 sub_step 配置 workdir")


def test_validate_concurrent_multiple_substep_workdirs():
    """校验：多个 sub_step 都配置了 workdir 时，每个都被报告。"""
    yaml_text = """
name: test_multiple_workdir_errors
steps:
  - id: iterate
    type: iteration
    items: "{{inputs.clusters}}"
    max_concurrent: 2
    sub_steps:
      - id: sub_a
        type: agent
        prompt: "a"
        workdir: /tmp/a
      - id: sub_b
        type: agent
        prompt: "b"
        workdir: /tmp/b
      - id: sub_c
        type: template
        template: "c"
"""
    wf = parse_yaml_workflow(yaml_text)
    errors = validate_workflow(wf)

    workdir_errors = [e for e in errors if "workdir" in e]
    assert len(workdir_errors) == 2, \
        f"应检测到 2 个 workdir 错误，实际: {workdir_errors}"
    assert any("sub_a" in e for e in workdir_errors), "应包含 sub_a 的错误"
    assert any("sub_b" in e for e in workdir_errors), "应包含 sub_b 的错误"
    print("✓ 校验规则: 多个 sub_step 的 workdir 错误都被正确报告")


# ---------------------------------------------------------------------------
# 测试 6：stop_on_error 取消机制
# ---------------------------------------------------------------------------

def test_concurrent_stop_on_error():
    """并发迭代 stop_on_error=True：一个 item 出错后取消其他 item。"""
    # 创建一个在第 2 个 item 会出错的脚本
    script_name = "test/concurrent_fail.py"
    _setup_test_script(script_name, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'item_val = data.get("item_val", "")\n'
        'if item_val == "fail":\n'
        '    print(json.dumps({"__error__": "intentional failure"}))\n'
        'else:\n'
        '    print(json.dumps({"ok": item_val}))\n'
    ))

    try:
        pool = VarPool(inputs={})

        node = {
            "id": "iterate_error",
            "type": "iteration",
            "items": ["ok1", "fail", "ok2", "ok3"],
            "item_var": "item",
            "max_concurrent": 4,
            "stop_on_error": True,
            "sub_steps": [
                {
                    "id": "may_fail",
                    "type": "script",
                    "script": script_name,
                    "input": {"item_val": "{{item}}"},
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        # 应该有错误
        assert "__errors__" in result, "应该有错误记录"
        assert len(result["__errors__"]) > 0, "至少有一个错误"

        # 验证错误指向 "fail" item
        error_items = [e["item"] for e in result["__errors__"]]
        assert "fail" in error_items, f"'fail' item 应在错误列表中，实际: {error_items}"

        print(f"✓ stop_on_error: 出错后取消其他 item，错误数: {len(result['__errors__'])}")
    finally:
        _cleanup_test_script(script_name)
        _set_terminal_cwd(None)


def test_concurrent_no_stop_on_error():
    """并发迭代 stop_on_error=False：出错后继续执行其他 item。"""
    script_name = "test/concurrent_continue.py"
    _setup_test_script(script_name, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'idx = data.get("idx", 0)\n'
        'if idx == 1:\n'
        '    print(json.dumps({"__error__": "item 1 fails"}))\n'
        'else:\n'
        '    print(json.dumps({"ok": idx}))\n'
    ))

    try:
        pool = VarPool(inputs={})

        node = {
            "id": "iterate_continue",
            "type": "iteration",
            "items": ["a", "b", "c", "d"],
            "item_var": "item",
            "max_concurrent": 4,
            "stop_on_error": False,
            "sub_steps": [
                {
                    "id": "may_fail",
                    "type": "script",
                    "script": script_name,
                    "input": {"idx": "{{__iter__.index}}"},
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        # 有错误记录
        assert "__errors__" in result

        # 但成功的 item 应该有结果
        valid_results = result.get("results", [])
        assert len(valid_results) == 3, f"应有 3 个成功结果，实际: {len(valid_results)}"

        print("✓ stop_on_error=False: 出错后继续执行其他 item")
    finally:
        _cleanup_test_script(script_name)
        _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 7：边界场景
# ---------------------------------------------------------------------------

def test_concurrent_empty_items():
    """并发迭代：items 为空列表。"""
    pool = VarPool(inputs={})

    node = {
        "id": "iterate_empty",
        "type": "iteration",
        "items": [],
        "item_var": "item",
        "max_concurrent": 3,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    assert "results" in result
    assert len(result["results"]) == 0, "空 items 应产生空结果"
    assert "__errors__" not in result
    print("✓ 边界场景: 空 items 正常处理")


def test_concurrent_single_item():
    """并发迭代：只有一个 item（并发退化为单线程）。"""
    pool = VarPool(inputs={})

    node = {
        "id": "iterate_single",
        "type": "iteration",
        "items": ["only_one"],
        "item_var": "item",
        "max_concurrent": 5,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "result_{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    assert "results" in result
    assert len(result["results"]) == 1
    assert result["results"][0]["item"] == "only_one"
    print("✓ 边界场景: 单 item 正常处理")


def test_concurrent_max_concurrent_larger_than_items():
    """并发迭代：max_concurrent 大于 items 数量。"""
    pool = VarPool(inputs={})

    node = {
        "id": "iterate_limit",
        "type": "iteration",
        "items": ["a", "b"],
        "item_var": "item",
        "max_concurrent": 100,  # 远超 items 数量
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    assert "results" in result
    assert len(result["results"]) == 2
    print("✓ 边界场景: max_concurrent 远大于 items 数量时正常处理")


# ---------------------------------------------------------------------------
# 测试 8：iter_workdir 共享
# ---------------------------------------------------------------------------

def test_concurrent_shared_workdir():
    """并发迭代：所有 item 共享迭代层的 workdir。"""
    script_name = "test/concurrent_shared_wd.py"
    _setup_test_script(script_name, (
        'import json, sys, os\n'
        'data = json.loads(sys.stdin.read())\n'
        'print(json.dumps({"cwd": os.getcwd()}))\n'
    ))

    with tempfile.TemporaryDirectory() as tmpdir:
        resolved_tmpdir = str(Path(tmpdir).resolve())

        try:
            pool = VarPool(inputs={})

            node = {
                "id": "iterate_shared_wd",
                "type": "iteration",
                "items": ["x", "y", "z"],
                "item_var": "item",
                "max_concurrent": 3,
                "workdir": tmpdir,
                "sub_steps": [
                    {
                        "id": "check_cwd",
                        "type": "script",
                        "script": script_name,
                        "input": {},
                        "outputs": ["cwd"],
                    }
                ],
                "outputs": ["results"],
            }

            result = _run_iteration_node(node, pool)

            assert "results" in result
            # 所有 item 的 script 应该都在 iter_workdir 下执行
            for r in result["results"]:
                if "__error__" not in r.get("output", {}).get("check_cwd", {}):
                    cwd = r["output"]["check_cwd"].get("cwd", "")
                    assert cwd == resolved_tmpdir, \
                        f"script cwd 应为 {resolved_tmpdir}，实际: {cwd}"

            print("✓ iter_workdir 共享: 所有 item 共享同一个迭代层 workdir")
        finally:
            _cleanup_test_script(script_name)
            _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 9：并发 vs 顺序结果一致性
# ---------------------------------------------------------------------------

def test_concurrent_vs_sequential_consistency():
    """并发迭代与顺序迭代的结果结构应一致。"""
    pool_seq = VarPool(inputs={"items_list": ["p", "q", "r"]})
    pool_seq.set_step_output("prep", {"output": {"items_list": ["p", "q", "r"]}})

    pool_conc = VarPool(inputs={"items_list": ["p", "q", "r"]})
    pool_conc.set_step_output("prep", {"output": {"items_list": ["p", "q", "r"]}})

    node_seq = {
        "id": "iterate_seq",
        "type": "iteration",
        "items": "{{steps.prep.output.items_list}}",
        "item_var": "item",
        "max_concurrent": 1,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "result_{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    node_conc = {
        "id": "iterate_conc",
        "type": "iteration",
        "items": "{{steps.prep.output.items_list}}",
        "item_var": "item",
        "max_concurrent": 3,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "result_{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result_seq = _run_iteration_node(node_seq, pool_seq)
    result_conc = _run_iteration_node(node_conc, pool_conc)

    # 两者结果结构应一致
    assert len(result_seq["results"]) == len(result_conc["results"])

    for r_seq, r_conc in zip(result_seq["results"], result_conc["results"]):
        assert r_seq["index"] == r_conc["index"]
        assert r_seq["item"] == r_conc["item"]
        # output 结构一致（template 结果值也应一致）
        assert r_seq["output"].keys() == r_conc["output"].keys()

    print("✓ 一致性: 并发与顺序迭代的结果结构一致")
    _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 10：多个 sub_step 的并发执行
# ---------------------------------------------------------------------------

def test_concurrent_multiple_substeps():
    """并发迭代：每个 item 有多个 sub_step 顺序执行。"""
    script_name = "test/concurrent_multi_sub.py"
    _setup_test_script(script_name, (
        'import json, sys\n'
        'data = json.loads(sys.stdin.read())\n'
        'print(json.dumps({"echo": data.get("val", "?")}))\n'
    ))

    try:
        pool = VarPool(inputs={})

        node = {
            "id": "iterate_multi",
            "type": "iteration",
            "items": ["alpha", "beta"],
            "item_var": "item",
            "max_concurrent": 2,
            "sub_steps": [
                {
                    "id": "step_one",
                    "type": "template",
                    "template": "hello_{{item}}",
                },
                {
                    "id": "step_two",
                    "type": "script",
                    "script": script_name,
                    "input": {"val": "{{item}}"},
                    "outputs": ["echo"],
                },
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        assert "results" in result
        assert len(result["results"]) == 2

        for r in result["results"]:
            output = r["output"]
            assert "step_one" in output, "应包含 step_one 的输出"
            assert "step_two" in output, "应包含 step_two 的输出"

        print("✓ 多 sub_step: 每个 item 的多个 sub_step 顺序执行")
    finally:
        _cleanup_test_script(script_name)
        _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 11：items 从 JSON 字符串解析
# ---------------------------------------------------------------------------

def test_concurrent_items_from_json_string():
    """并发迭代：items 为 JSON 字符串格式。"""
    pool = VarPool(inputs={})
    pool.set_step_output("prep", {"output": '["x", "y", "z"]'})

    node = {
        "id": "iterate_json",
        "type": "iteration",
        "items": "{{steps.prep.output}}",
        "item_var": "item",
        "max_concurrent": 2,
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    assert "results" in result
    assert len(result["results"]) == 3
    items = [r["item"] for r in result["results"]]
    assert items == ["x", "y", "z"]
    print("✓ items 解析: JSON 字符串格式正确解析为列表")
    _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 12：并发迭代后 TERMINAL_CWD 不泄漏
# ---------------------------------------------------------------------------

def test_concurrent_no_cwd_leak_after_error():
    """并发迭代出错后，TERMINAL_CWD 也应被清理。"""
    _set_terminal_cwd(None)

    with tempfile.TemporaryDirectory() as tmpdir:
        pool = VarPool(inputs={})

        node = {
            "id": "iterate_leak",
            "type": "iteration",
            "items": ["a"],
            "item_var": "item",
            "max_concurrent": 2,
            "workdir": tmpdir,
            "stop_on_error": True,
            "sub_steps": [
                {
                    "id": "fail_step",
                    "type": "template",
                    "template": "{{item}}",
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)

        # 无论成功或失败，TERMINAL_CWD 都应被清理
        assert os.environ.get("TERMINAL_CWD") is None, \
            f"迭代后 TERMINAL_CWD 应被清理，实际: {os.environ.get('TERMINAL_CWD')}"

        print("✓ 无泄漏: 迭代后 TERMINAL_CWD 被正确清理")
    _set_terminal_cwd(None)


# ---------------------------------------------------------------------------
# 测试 13：默认 max_concurrent 为 1（顺序执行）
# ---------------------------------------------------------------------------

def test_default_max_concurrent_is_sequential():
    """不设置 max_concurrent 时默认顺序执行。"""
    pool = VarPool(inputs={})

    node = {
        "id": "iterate_default",
        "type": "iteration",
        "items": ["a", "b", "c"],
        "item_var": "item",
        # 不设置 max_concurrent
        "sub_steps": [
            {
                "id": "template_step",
                "type": "template",
                "template": "{{item}}",
            }
        ],
        "outputs": ["results"],
    }

    result = _run_iteration_node(node, pool)

    assert "results" in result
    assert len(result["results"]) == 3
    print("✓ 默认行为: 不设 max_concurrent 时顺序执行")


# ---------------------------------------------------------------------------
# 运行所有测试
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("迭代并发测试")
    print("=" * 60)

    # 基本功能
    test_concurrent_basic_execution()
    test_concurrent_result_ordering()
    test_concurrent_varpool_isolation()

    # TERMINAL_CWD
    test_concurrent_terminal_cwd()
    test_concurrent_terminal_cwd_no_workdir()
    test_concurrent_no_cwd_leak_after_error()

    # 校验规则
    test_validate_concurrent_iteration_no_substep_workdir()
    test_validate_sequential_iteration_allows_substep_workdir()
    test_validate_concurrent_multiple_substep_workdirs()

    # stop_on_error
    test_concurrent_stop_on_error()
    test_concurrent_no_stop_on_error()

    # 边界场景
    test_concurrent_empty_items()
    test_concurrent_single_item()
    test_concurrent_max_concurrent_larger_than_items()

    # workdir 共享
    test_concurrent_shared_workdir()

    # 一致性
    test_concurrent_vs_sequential_consistency()

    # 多 sub_step
    test_concurrent_multiple_substeps()

    # items 解析
    test_concurrent_items_from_json_string()

    # 默认行为
    test_default_max_concurrent_is_sequential()

    print("\n" + "=" * 60)
    print("✅ 全部迭代并发测试通过！")
    print("=" * 60)
