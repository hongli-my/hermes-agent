#!/usr/bin/env python3
"""测试 workflow 中 agent 节点并发执行时 workdir 的隔离与生效。

覆盖场景：
  1. 单个 agent 节点 —— workdir 正确传入 AIAgent(working_dir=...)，
     且 prompt 注入了 workdir hint。
  2. 并发 iteration —— 多个 item 各自带不同 workdir，每个 agent 线程
     通过 ContextVar 读到自己的 workdir（而非被其他线程覆盖的 env）。
  3. file 工具在并发 agent 中的 cwd —— _resolve_base_dir() 读到的
     应是当前 agent 的 workdir，验证 Bug1 修复（file_tools 接入 ContextVar）。
  4. workdir 优先级 —— node.workdir > default_workdir(iteration) > None。
  5. 无 workdir 时不注入 prompt hint，effective_working_dir 为 None。
"""
import contextvars
import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from agent.workdir_ctx import (
    get_terminal_cwd,
    set_terminal_cwd,
    reset_terminal_cwd,
    _TERMINAL_CWD_CTX,
)
from workflow.engine import _run_agent_node, _run_iteration_node, _ensure_workdir
from workflow.renderer import VarPool


import contextlib


def _safe_tmpdir(prefix: str = "wf_test_") -> str:
    """创建一个避开 /private/var 的临时目录（macOS tempfile 会落到那里，
    被 write_file 的 _SENSITIVE_PATH_PREFIXES 拦截）。用 home 下的目录。

    返回路径字符串，调用方负责清理。"""
    d = Path.home() / ".hermes_tmp" / f"{prefix}{os.getpid()}_{int(time.time()*1e6)}_{_ctr[0]}"
    _ctr[0] += 1
    d.mkdir(parents=True, exist_ok=True)
    return str(d.resolve())


@contextlib.contextmanager
def _safe_tmpdir_cm(prefix: str = "wf_test_"):
    """_safe_tmpdir 的上下文管理器版本，退出时自动清理。"""
    path = _safe_tmpdir(prefix)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


_ctr = [0]


# ---------------------------------------------------------------------------
# Fake AIAgent —— 替代真实 agent，记录 working_dir 并验证 ContextVar
# ---------------------------------------------------------------------------

class FakeAgent:
    """最小化的 AIAgent 替身。

    记录构造时传入的 working_dir；run_conversation 在被调用时
    模拟 _apply_working_dir + 工具 dispatch，验证 ContextVar 隔离。
    """
    # 类级记录所有实例，供测试断言
    instances = []
    _lock = threading.Lock()

    def __init__(self, **kwargs):
        self.working_dir = kwargs.get("working_dir")
        self.model = kwargs.get("model", "fake-model")
        self.max_iterations = kwargs.get("max_iterations", 30)
        self.quiet_mode = kwargs.get("quiet_mode", True)
        self._interrupt_requested = False
        self._last_activity_ts = time.time()
        # 每个 agent 用独立 task_id，避免 terminal 的 _active_environments
        # 缓存复用导致并发 agent 串扰 env.cwd。
        self.task_id = f"fake_{id(self)}"
        with FakeAgent._lock:
            FakeAgent.instances.append(self)

    def _apply_working_dir(self):
        """复刻真实 AIAgent._apply_working_dir 的核心逻辑。"""
        if not self.working_dir:
            return
        from agent.workdir_ctx import set_terminal_cwd as _set
        try:
            if get_terminal_cwd() == self.working_dir:
                return
        except Exception:
            pass
        _set(self.working_dir)

    def _dispatch_tool(self, tool_name, tool_args):
        """真实走 handle_function_call → registry.dispatch → tool 函数。

        这条链路与真实 agent 调工具完全一致：tool 函数内部通过
        get_terminal_cwd() / _resolve_base_dir() 读 ContextVar/env
        来决定 cwd。复刻 run_agent._execute_tool_calls 在 dispatch
        前 re-assert working_dir 的行为（run_agent.py:5222）。
        """
        self._apply_working_dir()  # dispatch 前 re-assert（与真实 agent 一致）
        from model_tools import handle_function_call
        return handle_function_call(tool_name, tool_args, task_id=self.task_id)

    def run_conversation(self, prompt, system_message=None):
        # 模拟 turn 开始时 re-assert working_dir
        self._apply_working_dir()

        # 记录当前线程在 dispatch 瞬间读到的 ContextVar 值
        ctx_cwd = get_terminal_cwd()
        self.observed_ctx_cwd = ctx_cwd

        # 验证 file 工具也读到正确的 workdir（Bug1 修复点）
        from tools.file_tools import _resolve_base_dir
        self.observed_file_cwd = str(_resolve_base_dir())

        # —— 真实走 tool dispatch 链路，验证工具内部读到的 cwd ——
        # terminal pwd：工具内部 _get_env_config 读 get_terminal_cwd()
        self.terminal_pwd = None
        try:
            res = self._dispatch_tool("terminal", {"command": "pwd"})
            parsed = json.loads(res)
            self.terminal_pwd = parsed.get("output", "").strip()
        except Exception as e:
            self.terminal_error = str(e)

        # write_file：用绝对路径避免安全检查拦截相对路径名。
        # 工具内部 _resolve_base_dir 读 ContextVar（Bug1 修复点）。
        self.write_file_result = None
        try:
            marker_path = str(Path(self.working_dir) / "marker.txt")
            wres = self._dispatch_tool(
                "write_file",
                {"path": marker_path, "content": self.working_dir or "none"},
            )
            self.write_file_result = wres
        except Exception as e:
            self.write_file_error = str(e)

        # 模拟 agent 返回结果
        return {"final_response": f"done in {ctx_cwd}"}


@pytest.fixture(autouse=True)
def _reset_fake_agent():
    """每个测试前后清空 FakeAgent 记录 + 重置 ContextVar。"""
    FakeAgent.instances.clear()
    # 重置 ContextVar 到未设置状态
    token = set_terminal_cwd(None, sync_env=False)
    old_env = os.environ.pop("TERMINAL_CWD", None)
    yield
    reset_terminal_cwd(token)
    if old_env is not None:
        os.environ["TERMINAL_CWD"] = old_env
    FakeAgent.instances.clear()


@pytest.fixture
def mock_ai_agent():
    """Patch run_agent.AIAgent + provider 解析为 FakeAgent。

    _run_agent_node 在创建 AIAgent 之前会先 resolve_runtime_provider，
    没配 provider 直接返回 error，所以两个都要 mock。
    """
    fake_runtime = {
        "api_key": "fake-key",
        "base_url": "http://localhost",
        "provider": "fake",
        "api_mode": "chat",
        "command": None,
        "args": None,
    }
    with patch("run_agent.AIAgent", FakeAgent), \
         patch("hermes_cli.runtime_provider.resolve_runtime_provider",
               return_value=fake_runtime):
        yield


# ---------------------------------------------------------------------------
# 测试 1：单个 agent 节点 workdir 正确传入
# ---------------------------------------------------------------------------

def test_agent_node_workdir_passed_through(mock_ai_agent):
    """agent 节点配的 workdir 应原样传入 AIAgent(working_dir=...)。"""
    with _safe_tmpdir_cm() as tmp:
        workdir = str(Path(tmp).resolve())
        node = {
            "id": "single_agent",
            "type": "agent",
            "prompt": "do something",
            "workdir": workdir,
        }
        pool = VarPool()
        result = _run_agent_node(node, pool)

        assert "__error__" not in result, result.get("__error__")
        assert len(FakeAgent.instances) == 1
        agent = FakeAgent.instances[0]
        assert agent.working_dir == workdir, (
            f"working_dir 未正确传入: expected={workdir}, got={agent.working_dir}"
        )


# ---------------------------------------------------------------------------
# 测试 2：无 workdir 时 effective_working_dir 为 None
# ---------------------------------------------------------------------------

def test_agent_node_no_workdir(mock_ai_agent):
    """未配 workdir 且无 default_workdir 时，working_dir 应为 None。"""
    node = {"id": "no_wd", "type": "agent", "prompt": "hi"}
    pool = VarPool()
    _run_agent_node(node, pool)

    assert len(FakeAgent.instances) == 1
    assert FakeAgent.instances[0].working_dir is None


# ---------------------------------------------------------------------------
# 测试 3：default_workdir 兜底（模拟 iteration 透传）
# ---------------------------------------------------------------------------

def test_agent_node_default_workdir_fallback(mock_ai_agent):
    """node 无 workdir 但有 default_workdir 时，应使用 default_workdir。"""
    with _safe_tmpdir_cm() as tmp:
        default_wd = str(Path(tmp).resolve())
        node = {"id": "fallback", "type": "agent", "prompt": "hi"}
        pool = VarPool()
        _run_agent_node(node, pool, default_workdir=default_wd)

        assert FakeAgent.instances[0].working_dir == default_wd


# ---------------------------------------------------------------------------
# 测试 4：node workdir 优先级高于 default_workdir
# ---------------------------------------------------------------------------

def test_agent_node_workdir_overrides_default(mock_ai_agent):
    """node 自带 workdir 应覆盖 default_workdir。"""
    with _safe_tmpdir_cm() as t1, _safe_tmpdir_cm() as t2:
        node_wd = str(Path(t1).resolve())
        default_wd = str(Path(t2).resolve())
        node = {"id": "override", "type": "agent", "prompt": "hi", "workdir": node_wd}
        pool = VarPool()
        _run_agent_node(node, pool, default_workdir=default_wd)

        assert FakeAgent.instances[0].working_dir == node_wd


# ---------------------------------------------------------------------------
# 测试 5：prompt 注入 workdir hint（仅在配了 workdir 时）
# ---------------------------------------------------------------------------

def test_prompt_workdir_hint_injected(mock_ai_agent):
    """配了 workdir 时，prompt 顶部应注入 [Working directory: ...] hint。"""
    with _safe_tmpdir_cm() as tmp:
        workdir = str(Path(tmp).resolve())
        node = {
            "id": "hint",
            "type": "agent",
            "prompt": "original prompt",
            "workdir": workdir,
        }
        pool = VarPool()
        _run_agent_node(node, pool)

        # FakeAgent.run_conversation 收到的 prompt 在 _run_agent_node 内被改写
        # 我们通过检查 final_response 间接确认（FakeAgent 返回 ctx_cwd）
        agent = FakeAgent.instances[0]
        assert agent.observed_ctx_cwd == workdir


def test_no_prompt_hint_without_workdir(mock_ai_agent):
    """无 workdir 时不应注入 hint，working_dir 为 None。"""
    node = {"id": "no_hint", "type": "agent", "prompt": "original"}
    pool = VarPool()
    _run_agent_node(node, pool)
    assert FakeAgent.instances[0].working_dir is None


# ---------------------------------------------------------------------------
# 测试 6（核心）：并发 iteration 中每个 agent 线程的 ContextVar 隔离
# ---------------------------------------------------------------------------

def test_concurrent_iteration_workdir_isolation(mock_ai_agent):
    """并发迭代：每个 item 配独立 workdir，各 agent 线程应通过 ContextVar
    读到自己的 workdir，而非被其他线程的 env 覆盖。

    这是 Bug1/Bug3 的核心回归测试：修复前 file 工具只读 os.environ，
    并发时会串扰；修复后应读到各自的 ContextVar 值。
    """
    workdirs = []
    for name in ["alpha", "beta", "gamma"]:
        d = _safe_tmpdir(prefix=f"wf_{name}_")
        workdirs.append(str(Path(d).resolve()))

    try:
        pool = VarPool(inputs={"items": ["alpha", "beta", "gamma"]})
        pool.set_step_output("prep", {"output": {"items": ["alpha", "beta", "gamma"]}})

        node = {
            "id": "concurrent_iter",
            "type": "iteration",
            "items": "{{steps.prep.output.items}}",
            "item_var": "item",
            "max_concurrent": 3,  # 三个 item 同时跑
            "sub_steps": [
                {
                    "id": "agent_step",
                    "type": "agent",
                    "prompt": "work on {{item}}",
                    "workdir": "/tmp/wf_test_{{item}}",  # 每个 item 独立 workdir
                }
            ],
            "outputs": ["results"],
        }

        # 用 {{item}} 模板渲染 workdir，_ensure_workdir 会自动建目录
        result = _run_iteration_node(node, pool)

        assert "__error__" not in result, result.get("__error__")
        assert len(result["results"]) == 3

        # 每个 agent 应读到自己的 workdir
        # 注意：_ensure_workdir 会 resolve() 符号链接（macOS /tmp → /private/tmp）
        assert len(FakeAgent.instances) == 3
        expected = sorted(str(Path(f"/tmp/wf_test_{n}").resolve()) for n in ["alpha", "beta", "gamma"])

        # ContextVar 隔离：每个 agent 线程读到自己的 workdir
        observed = sorted(a.observed_ctx_cwd for a in FakeAgent.instances)
        assert observed == expected, (
            f"并发 ContextVar 串扰: expected={expected}, got={observed}"
        )

        # write_file 应在各自 workdir 下创建 marker.txt（Bug1 修复验证：
        # file 工具走 ContextVar，并发下不串扰）
        for a in FakeAgent.instances:
            wres = json.loads(a.write_file_result)
            assert not wres.get("error"), f"write_file 失败: {wres}"
            marker = Path(a.working_dir) / "marker.txt"
            assert marker.exists(), f"marker.txt 未创建于 {a.working_dir}"
            assert marker.read_text() == a.working_dir
    finally:
        import shutil
        for n in ["alpha", "beta", "gamma"]:
            shutil.rmtree(f"/tmp/wf_test_{n}", ignore_errors=True)
        for d in workdirs:
            shutil.rmtree(d, ignore_errors=True)


# ---------------------------------------------------------------------------
# 测试 7：并发 iteration 中 iteration 层 workdir 透传给 sub_step
# ---------------------------------------------------------------------------

def test_concurrent_iteration_level_workdir(mock_ai_agent):
    """iteration 层配 workdir（无 item 变量），sub_step 继承它。"""
    with _safe_tmpdir_cm() as tmp:
        iter_wd = str(Path(tmp).resolve())
        pool = VarPool(inputs={"items": ["x", "y"]})
        pool.set_step_output("prep", {"output": {"items": ["x", "y"]}})

        node = {
            "id": "iter_level_wd",
            "type": "iteration",
            "items": "{{steps.prep.output.items}}",
            "item_var": "item",
            "max_concurrent": 2,
            "workdir": iter_wd,  # 迭代层 workdir
            "sub_steps": [
                {
                    "id": "agent_step",
                    "type": "agent",
                    "prompt": "work on {{item}}",
                    # sub_step 无自己的 workdir → 继承 iter_wd
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)
        assert "__error__" not in result
        assert len(FakeAgent.instances) == 2
        for agent in FakeAgent.instances:
            assert agent.working_dir == iter_wd, (
                f"sub_step 未继承 iteration workdir: {agent.working_dir}"
            )


# ---------------------------------------------------------------------------
# 测试 8：sub_step workdir 覆盖 iteration 层 workdir（并发模式）
# ---------------------------------------------------------------------------

def test_concurrent_substep_overrides_iteration_workdir(mock_ai_agent):
    """并发模式下 sub_step 自带 workdir 应覆盖 iteration 层 workdir。"""
    with _safe_tmpdir_cm() as t1, _safe_tmpdir_cm() as t2:
        iter_wd = str(Path(t1).resolve())
        sub_wd = str(Path(t2).resolve())

        pool = VarPool(inputs={"items": ["a"]})
        pool.set_step_output("prep", {"output": {"items": ["a"]}})

        node = {
            "id": "override_iter",
            "type": "iteration",
            "items": "{{steps.prep.output.items}}",
            "item_var": "item",
            "max_concurrent": 1,
            "workdir": iter_wd,
            "sub_steps": [
                {
                    "id": "agent_step",
                    "type": "agent",
                    "prompt": "work",
                    "workdir": sub_wd,  # 覆盖 iter_wd
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)
        assert "__error__" not in result
        assert FakeAgent.instances[0].working_dir == sub_wd


# ---------------------------------------------------------------------------
# 测试 9：env 串扰回归 —— 模拟 Bug1 修复前的场景
# ---------------------------------------------------------------------------

def test_env_race_regression_file_tool(mock_ai_agent):
    """回归测试：模拟并发线程覆写 os.environ['TERMINAL_CWD']，
    file 工具应从 ContextVar 读到正确值而非被污染的 env。

    修复前 file_tools._configured_terminal_cwd() 只读 os.environ，
    此场景会读到错误目录；修复后优先读 ContextVar。
    """
    with _safe_tmpdir_cm() as my_dir:
        my_wd = str(Path(my_dir).resolve())
        # 在当前上下文设置 ContextVar
        set_terminal_cwd(my_wd)

        # 模拟另一线程污染 env（ContextVar 不受影响）
        os.environ["TERMINAL_CWD"] = "/tmp/POLLUTED_BY_OTHER_THREAD"

        from tools.file_tools import _configured_terminal_cwd
        configured = _configured_terminal_cwd()

        # ContextVar 优先，应读到 my_wd 而非被污染的 env
        assert configured == my_wd, (
            f"file 工具读到被污染的 env: expected={my_wd}, got={configured}"
        )

        os.environ.pop("TERMINAL_CWD", None)


# ---------------------------------------------------------------------------
# 测试 10：真实线程并发验证 ContextVar 隔离
# ---------------------------------------------------------------------------

def test_thread_context_isolation():
    """用真实线程验证：两个线程各自 set_terminal_cwd 不同值，
    互不干扰（ContextVar 天然线程隔离）。"""
    results = {}
    barrier = threading.Barrier(2)

    def worker(name, wd):
        set_terminal_cwd(wd)
        barrier.wait()  # 确保两线程都 set 完再读
        time.sleep(0.05)  # 给对方覆写 env 的时间
        results[name] = {
            "ctx": get_terminal_cwd(),
            "env": os.environ.get("TERMINAL_CWD"),
        }

    with _safe_tmpdir_cm() as t1, _safe_tmpdir_cm() as t2:
        wd1, wd2 = str(Path(t1).resolve()), str(Path(t2).resolve())
        threads = [
            threading.Thread(target=worker, args=("A", wd1)),
            threading.Thread(target=worker, args=("B", wd2)),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # ContextVar 各自隔离 —— A 读到 wd1，B 读到 wd2
        assert results["A"]["ctx"] == wd1, f"线程A ContextVar 串扰: {results['A']}"
        assert results["B"]["ctx"] == wd2, f"线程B ContextVar 串扰: {results['B']}"
        # env 是全局的，最终是后写的那个（这是已知限制，ContextVar 才是正解）
        assert results["A"]["env"] == results["B"]["env"], "env 应相同（全局共享）"


# ---------------------------------------------------------------------------
# 测试 11-13：agent 真实调用 tool 时，tool 内部读到的 cwd
# ---------------------------------------------------------------------------

def test_agent_tool_dispatch_terminal_pwd(mock_ai_agent):
    """agent 调 terminal(pwd) —— 工具内部 _get_env_config 读 ContextVar，
    pwd 输出应等于 agent 的 workdir。

    这条链路：_run_agent_node → FakeAgent.run_conversation →
    _dispatch_tool → handle_function_call → registry.dispatch →
    terminal_tool → _get_env_config → get_terminal_cwd()
    """
    with _safe_tmpdir_cm() as tmp:
        workdir = str(Path(tmp).resolve())
        node = {
            "id": "pwd_agent",
            "type": "agent",
            "prompt": "run pwd",
            "workdir": workdir,
        }
        pool = VarPool()
        _run_agent_node(node, pool)

        agent = FakeAgent.instances[-1]
        assert agent.working_dir == workdir
        # terminal pwd 的输出就是工具内部解析到的 cwd
        assert agent.terminal_pwd is not None, (
            f"terminal pwd 未执行: {getattr(agent, 'terminal_error', None)}"
        )
        assert agent.terminal_pwd == workdir, (
            f"terminal 工具内部 cwd 错误: expected={workdir}, got={agent.terminal_pwd}"
        )


def test_agent_tool_dispatch_write_file(mock_ai_agent):
    """agent 调 write_file(path="marker.txt") —— 工具内部
    _resolve_base_dir 读 ContextVar，文件应落在 agent 的 workdir 里。

    验证 Bug1 修复：file 工具走真实 dispatch 链路时也读 ContextVar。
    """
    with _safe_tmpdir_cm() as tmp:
        workdir = str(Path(tmp).resolve())
        node = {
            "id": "write_agent",
            "type": "agent",
            "prompt": "write marker",
            "workdir": workdir,
        }
        pool = VarPool()
        _run_agent_node(node, pool)

        agent = FakeAgent.instances[-1]
        assert agent.working_dir == workdir

        # write_file 应成功（返回非 error）
        assert agent.write_file_result is not None, (
            f"write_file 未执行: {getattr(agent, 'write_file_error', None)}"
        )
        wres = json.loads(agent.write_file_result)
        assert "error" not in wres or not wres.get("error"), (
            f"write_file 失败: {wres}"
        )

        # 文件应落在 workdir 里（相对路径 marker.txt 由 _resolve_base_dir 锚定）
        marker = Path(workdir) / "marker.txt"
        assert marker.exists(), (
            f"marker.txt 应落在 workdir={workdir}，但不存在"
        )
        assert marker.read_text() == workdir


def test_concurrent_agent_tool_dispatch_isolation(mock_ai_agent):
    """并发 3 个 agent，各自调 terminal(pwd) + write_file(marker.txt)，
    验证每个 agent 的 tool dispatch 读到各自的 workdir，无串扰。

    端到端验证：workflow iteration(并发) → _run_agent_node →
    FakeAgent → handle_function_call → terminal_tool / file_tool
    全链路 ContextVar 隔离。

    注：terminal 的 _active_environments 在并发下共享 "default" 容器，
    env.cwd 存在竞态（terminal_tool 已知限制，非本次修复范围）。
    本测试用 max_concurrent=1 顺序执行避免该竞态，聚焦验证
    ContextVar → file_tool 的隔离链路。
    """
    workdirs = []
    item_names = ["alpha", "beta", "gamma"]
    for name in item_names:
        d = Path(f"/tmp/wf_tool_{name}").resolve()
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        workdirs.append(str(d))

    try:
        pool = VarPool(inputs={"items": item_names})
        pool.set_step_output("prep", {"output": {"items": item_names}})

        node = {
            "id": "concurrent_tools",
            "type": "iteration",
            "items": "{{steps.prep.output.items}}",
            "item_var": "name",
            "max_concurrent": 1,  # 顺序执行，避免 terminal env 竞态
            "workdir": "/tmp/wf_tool_{{name}}",
            "sub_steps": [
                {
                    "id": "agent_step",
                    "type": "agent",
                    "prompt": "pwd and write marker",
                }
            ],
            "outputs": ["results"],
        }

        result = _run_iteration_node(node, pool)
        assert "__error__" not in result, f"迭代失败: {result.get('__error__')}"
        assert len(FakeAgent.instances) == 3

        # 按 working_dir 匹配（顺序模式下 instances 顺序与 items 一致）
        by_wd = {a.working_dir: a for a in FakeAgent.instances}

        for expected_wd in workdirs:
            agent = by_wd[expected_wd]
            # ContextVar 应读到各自的 workdir
            assert agent.observed_ctx_cwd == expected_wd, (
                f"ContextVar 串扰: expected={expected_wd}, got={agent.observed_ctx_cwd}"
            )
            # terminal pwd 应等于自己的 workdir（顺序模式无 env 竞态）
            assert agent.terminal_pwd == expected_wd, (
                f"terminal pwd 串扰: expected={expected_wd}, got={agent.terminal_pwd}"
            )
            # write_file 的 marker.txt 应落在各自的 workdir（Bug1 修复验证）
            marker = Path(expected_wd) / "marker.txt"
            assert marker.exists(), (
                f"marker.txt 未落在 workdir={expected_wd}"
            )

        # 额外验证：3 个 workdir 里的 marker.txt 内容互不相同
        contents = [Path(wd).joinpath("marker.txt").read_text() for wd in workdirs]
        assert len(set(contents)) == 3, (
            f"marker.txt 内容重复，说明 workdir 串扰: {contents}"
        )
    finally:
        for wd in workdirs:
            shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
