"""
Hermes Workflow Engine — step-based execution engine.

Supports node types:
  - script:    run Python script, stdin JSON → stdout JSON
  - shell:     run shell command, capture stdout/stderr/returncode
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

import contextvars
import copy
import json
import logging
import os
import subprocess
import sys
import threading
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


def _ensure_workdir(value: Optional[str], node_id: str = "?",
                    source: str = "workdir") -> Optional[str]:
    """Resolve, auto-create, and return a workdir path.

    渲染后的 workdir 模板（例如 ``/tmp/runs/{{run_id}}/{{item.name}}``）
    经常指向尚未存在的目录。如果不预先创建：
      - script/shell 节点：``subprocess.run(cwd=...)`` 直接抛
        ``FileNotFoundError: [Errno 2] No such file or directory``。
      - agent 节点：``working_dir`` 传给 AIAgent，后续 terminal/file 工具
        在该目录下操作时才报错，问题更隐蔽。

    本 helper 把所有 workdir 落地点统一处理：
      1. ``expanduser`` + ``resolve`` 得到绝对路径；
      2. ``mkdir(parents=True, exist_ok=True)`` 自动创建（含父目录）；
      3. 创建失败只记 warning，仍返回解析后的路径，让真实子进程 / 工具去
         报具体的权限错误（避免本 helper 吞掉错误）。

    传入空值（None / 空字符串）原样返回 None。
    """
    if not value:
        return None
    try:
        resolved = Path(str(value)).expanduser().resolve()
    except Exception as exc:
        logger.warning("Node[%s]: failed to resolve %s=%r: %s",
                       node_id, source, value, exc)
        return str(value)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except FileExistsError:
        # 路径已存在但是文件而非目录 —— 让下游报具体错
        logger.warning("Node[%s]: %s=%s exists but is not a directory",
                       node_id, source, resolved)
    except PermissionError as exc:
        logger.warning("Node[%s]: cannot create %s=%s: %s",
                       node_id, source, resolved, exc)
    except Exception as exc:
        logger.warning("Node[%s]: unexpected error creating %s=%s: %s",
                       node_id, source, resolved, exc)
    return str(resolved)


# ---------------------------------------------------------------------------
# Script node executor
# ---------------------------------------------------------------------------

def _run_script_node(node: Dict[str, Any], pool: VarPool,
                     default_workdir: Optional[str] = None) -> Dict[str, Any]:
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
        logger.error("Script[%s]: path escapes scripts directory — %s", node.get("id", "?"), script_path)
        return {"__error__": f"Script path escapes scripts directory: {script_path}"}

    if not resolved.exists():
        logger.error("Script[%s]: file not found — %s", node.get("id", "?"), resolved)
        return {"__error__": f"Script not found: {resolved}"}

    # Render input variables
    input_data = render(node.get("input", {}), pool)

    # Determine working directory for the script.
    # _ensure_workdir 会自动 expanduser+resolve+mkdir，避免模板渲染出尚未
    # 存在的目录时 subprocess 抛 FileNotFoundError。
    script_workdir = node.get("workdir")
    if script_workdir:
        script_workdir = _ensure_workdir(
            render(str(script_workdir), pool),
            node.get("id", "?"), "script.workdir")
    elif default_workdir:
        script_workdir = _ensure_workdir(
            default_workdir, node.get("id", "?"), "script.default_workdir")
    else:
        script_workdir = str(resolved.parent)  # Default: script's directory

    timeout = node.get("timeout", 120)

    logger.info("Script[%s]: resolved=%s, workdir=%s, timeout=%ds",
                node.get("id", "?"), resolved, script_workdir, timeout)

    try:
        # Build child env: inherit from parent but override TERMINAL_CWD with
        # this script's resolved working dir, so scripts that read the env var
        # still see the right worktree even when concurrent peers have
        # overwritten the parent process's env.
        child_env = os.environ.copy()
        child_env["TERMINAL_CWD"] = script_workdir

        result = subprocess.run(
            [sys.executable, str(resolved)],
            input=json.dumps(input_data, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=script_workdir,
            env=child_env,
        )

        # Log stderr if present
        if result.stderr:
            for line in result.stderr.strip().split("\n"):
                logger.info("Script[%s] stderr: %s", node.get("id", "?"), line)

        if result.returncode != 0:
            logger.error("Script[%s]: exited with code %d", node.get("id", "?"), result.returncode)
            return {"__error__": f"Script exited with code {result.returncode}: {result.stderr.strip()}"}

        # Parse stdout as JSON
        outputs = _parse_script_output(result.stdout)
        logger.info("Script[%s]: completed OK, output keys=%s",
                    node.get("id", "?"), list(outputs.keys()) if isinstance(outputs, dict) else type(outputs).__name__)

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
        logger.error("Script[%s]: timed out after %ds", node.get("id", "?"), timeout)
        return {"__error__": f"Script timed out after {timeout}s"}
    except Exception as e:
        logger.error("Script[%s]: execution failed — %s", node.get("id", "?"), e)
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
# Shell node executor
# ---------------------------------------------------------------------------

# 默认开启 shell 节点；用户可通过设置环境变量 HERMES_WF_ALLOW_SHELL=0 / false
# 在部署侧统一禁用，避免误用 YAML 执行任意命令。
_SHELL_DISABLE_VALUES = {"0", "false", "no", "off", "disable", "disabled"}


def _shell_nodes_enabled() -> bool:
    val = os.environ.get("HERMES_WF_ALLOW_SHELL")
    if val is None:
        return True
    return val.strip().lower() not in _SHELL_DISABLE_VALUES


def _run_shell_node(node: Dict[str, Any], pool: VarPool,
                    default_workdir: Optional[str] = None) -> Dict[str, Any]:
    """Execute an arbitrary shell command.

    YAML fields:
      cmd            — str (passed to /bin/sh) or list[str] (argv, no shell)
      workdir        — optional working dir (supports variable rendering)
      timeout        — seconds, default 60
      env            — optional dict of extra env vars (merged onto parent env)
      shell          — bool, default True when cmd is str, False when list
      allow_nonzero  — bool, default False; when True, non-zero exit is NOT
                       treated as error (stdout/stderr/returncode still returned)
      outputs        — optional list[str] to filter returned keys

    Returns:
      {
        "stdout": str,
        "stderr": str,
        "returncode": int,
        ["__error__": str]   # on timeout / nonzero (unless allow_nonzero)
      }
    """
    node_id = node.get("id", "?")

    if not _shell_nodes_enabled():
        logger.error("Shell[%s]: shell nodes are disabled via HERMES_WF_ALLOW_SHELL", node_id)
        return {"__error__": "Shell nodes are disabled (HERMES_WF_ALLOW_SHELL=0)"}

    raw_cmd = node.get("cmd")
    if raw_cmd is None or raw_cmd == "":
        logger.error("Shell[%s]: missing 'cmd' field", node_id)
        return {"__error__": "Shell node missing 'cmd' field"}

    # Render cmd (supports templating). String stays string; list elements
    # get rendered one by one so argv mode still works with variables.
    if isinstance(raw_cmd, list):
        cmd = [render(str(part), pool) for part in raw_cmd]
        use_shell_default = False
        cmd_display = " ".join(cmd)
    else:
        cmd = render(str(raw_cmd), pool)
        use_shell_default = True
        cmd_display = cmd

    use_shell = bool(node.get("shell", use_shell_default))

    # Working directory — same precedence as script/agent nodes.
    # _ensure_workdir 会创建不存在的目录，避免 subprocess 抛
    # FileNotFoundError(cwd)。
    shell_workdir = node.get("workdir")
    if shell_workdir:
        shell_workdir = _ensure_workdir(
            render(str(shell_workdir), pool), node_id, "shell.workdir")
    elif default_workdir:
        shell_workdir = _ensure_workdir(
            default_workdir, node_id, "shell.default_workdir")
    else:
        shell_workdir = None  # inherit from current process

    timeout = node.get("timeout", 60)
    allow_nonzero = bool(node.get("allow_nonzero", False))

    # Merge extra env
    child_env = os.environ.copy()
    extra_env = node.get("env") or {}
    if extra_env:
        for k, v in extra_env.items():
            child_env[str(k)] = render(str(v), pool)
    if shell_workdir:
        child_env["TERMINAL_CWD"] = shell_workdir

    logger.info("Shell[%s]: cmd=%r, workdir=%s, timeout=%ss, shell=%s, allow_nonzero=%s",
                node_id, cmd_display, shell_workdir, timeout, use_shell, allow_nonzero)

    try:
        result = subprocess.run(
            cmd,
            shell=use_shell,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=shell_workdir,
            env=child_env,
        )
    except subprocess.TimeoutExpired:
        logger.error("Shell[%s]: timed out after %ss", node_id, timeout)
        return {"__error__": f"Shell command timed out after {timeout}s"}
    except FileNotFoundError as e:
        logger.error("Shell[%s]: executable not found — %s", node_id, e)
        return {"__error__": f"Executable not found: {e}"}
    except Exception as e:
        logger.error("Shell[%s]: execution failed — %s", node_id, e)
        return {"__error__": f"Shell execution failed: {e}"}

    stdout = result.stdout or ""
    stderr = result.stderr or ""
    rc = result.returncode

    # Log stderr line-by-line (truncated for very long output)
    if stderr.strip():
        for line in stderr.strip().split("\n")[:20]:
            logger.info("Shell[%s] stderr: %s", node_id, line)

    outputs: Dict[str, Any] = {
        "stdout": stdout,
        "stderr": stderr,
        "returncode": rc,
    }

    if rc != 0 and not allow_nonzero:
        err_preview = stderr.strip() or stdout.strip()
        if len(err_preview) > 200:
            err_preview = err_preview[:200] + "..."
        logger.error("Shell[%s]: exited with code %d — %s", node_id, rc, err_preview)
        outputs["__error__"] = f"Shell exited with code {rc}: {err_preview}"
    else:
        logger.info("Shell[%s]: completed — rc=%d, stdout=%d chars, stderr=%d chars",
                    node_id, rc, len(stdout), len(stderr))

    # Filter by outputs list if specified
    output_keys = node.get("outputs")
    if output_keys and isinstance(output_keys, list):
        filtered: Dict[str, Any] = {}
        for key in output_keys:
            if key in outputs:
                filtered[key] = outputs[key]
        if "__error__" in outputs:
            filtered["__error__"] = outputs["__error__"]
        return filtered

    return outputs


# ---------------------------------------------------------------------------
# Agent node executor
# ---------------------------------------------------------------------------

def _run_agent_node(node: Dict[str, Any], pool: VarPool,
                    session_db=None,
                    default_workdir: Optional[str] = None) -> Dict[str, Any]:
    """Execute an agent node. AIAgent → final_response → variables."""
    node_id = node.get("id", "?")
    try:
        from run_agent import AIAgent
        from dotenv import load_dotenv
    except ImportError:
        logger.error("Agent[%s]: AIAgent not available (run_agent.py not importable)", node_id)
        return {"__error__": "AIAgent not available (run_agent.py not importable)"}

    prompt_template = node.get("prompt", "")
    prompt = render(prompt_template, pool)

    if not prompt.strip():
        logger.error("Agent[%s]: prompt is empty after variable resolution", node_id)
        return {"__error__": "Agent prompt is empty after variable resolution"}

    logger.info("Agent[%s]: prompt rendered, length=%d chars", node_id, len(prompt))

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
        logger.error("Agent[%s]: provider resolution failed — %s", node_id, exc)
        return {"__error__": f"Provider resolution failed: {exc}"}

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

    # Resolve the effective working_dir for this agent node.
    # Priority: node's own workdir > default_workdir (from iteration/parallel) > None.
    # Pass it as a constructor arg so the agent can re-assert TERMINAL_CWD
    # into os.environ before every tool dispatch.  This keeps concurrent
    # agents isolated without needing a process-global lock on the env.
    node_workdir = node.get("workdir")
    if node_workdir:
        node_workdir = render(str(node_workdir), pool)
        # 创建不存在的目录，避免后续 terminal/file 工具在不存在的 workdir
        # 下报错。
        node_workdir = _ensure_workdir(
            node_workdir, node.get("id", "?"), "agent.workdir")
        logger.info("Agent node '%s': using node workdir=%s", node.get("id", "?"), node_workdir)
        effective_working_dir = node_workdir
    elif default_workdir:
        logger.info("Agent node '%s': using default_workdir=%s", node.get("id", "?"), default_workdir)
        effective_working_dir = _ensure_workdir(
            default_workdir, node.get("id", "?"), "agent.default_workdir")
    else:
        logger.info("Agent node '%s': no workdir specified", node.get("id", "?"))
        effective_working_dir = None

    # 显式把 workdir 告诉模型。
    # 仅靠 AIAgent(working_dir=...) 只能把 *子进程 cwd* 设到 workdir；
    # 模型本身看不到这个变量，如果 prompt / skill 里没说清楚保存位置，
    # 模型常会自行写到 /tmp 等无关目录，绕过 cwd 兜底（绝对路径）。
    # 这里在 prompt 顶部追加一行明确的硬约束，让模型把生成文件落到
    # workdir 内。effective_working_dir 为 None（即未配 workdir）时
    # 不注入，保持原有 CLI / 无 workdir 场景行为不变。
    if effective_working_dir:
        workdir_hint = (
            f"[Working directory: {effective_working_dir}]\n"
            f"All generated files MUST be saved under this directory, "
            f"using either absolute paths rooted here or paths relative to it. "
            f"Do NOT write to /tmp, the current process cwd, or other "
            f"locations unless the task explicitly requires it.\n\n---\n\n"
        )
        prompt = workdir_hint + prompt
        logger.info("Agent[%s]: injected workdir hint into prompt (workdir=%s)",
                    node_id, effective_working_dir)

    # Silence child-agent's KawaiiSpinner.  In workflow mode, many agent
    # nodes run concurrently and their spinner animations corrupt each
    # other's stdout (the "ruminating... / mulling..." zombie lines seen
    # during concurrent iteration).  The workflow CLI renders its own
    # heartbeat line driven by AIAgent._last_activity_ts, so child spinners
    # are redundant and actively harmful here.
    _noop_thinking_cb = lambda _msg: None  # noqa: E731

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
        thinking_callback=_noop_thinking_cb,
        working_dir=effective_working_dir
    )
    logger.info("Agent[%s]: AIAgent created — model=%s, max_iterations=%d, workdir=%s",
                node_id, turn_route.get("model", model), max_iterations, effective_working_dir)

    # Register this agent as "alive" so the CLI heartbeat can read its
    # real _last_activity_ts and tell whether it's still making progress
    # or actually stuck (HTTP hang, deadlock, etc.).  Label follows the
    # iteration index when inside an iteration so users can correlate
    # heartbeat entries with the tree view above.
    try:
        from workflow.agent_registry import register_agent, unregister_agent
        node_id = node.get("id", "agent")
        iter_ctx = getattr(pool, "iter_context", None)
        if iter_ctx and "index" in iter_ctx and "total" in iter_ctx:
            label = f"{node_id}[{int(iter_ctx['index']) + 1}/{iter_ctx['total']}]"
        else:
            label = node_id
        registry_token = register_agent(label=label, agent=agent)
        logger.info("Agent[%s]: registered as alive — label=%s, token=%s",
                    node_id, label, registry_token[:8])
    except Exception:
        register_agent = None  # type: ignore[assignment]
        unregister_agent = None  # type: ignore[assignment]
        registry_token = None

    try:
        # Agent 节点整体超时保护。
        # 与 script 节点不同，agent.run_conversation() 没有内置 timeout，
        # 如果 provider 无响应 + stale detector 未触发（如本地 provider 的
        # _stream_stale_timeout=inf），agent 可能无限期挂起，阻塞整个 workflow。
        agent_timeout = node.get("timeout", 600)
        _agent_result = [None]
        _agent_exc = [None]
        logger.info("Agent[%s]: starting run_conversation, timeout=%ss",
                    node_id, agent_timeout)

        def _run_agent():
            try:
                _agent_result[0] = agent.run_conversation(prompt, system_message=system_prompt)
            except Exception as _e:
                _agent_exc[0] = _e

        if agent_timeout is not None:
            agent_timeout = float(agent_timeout)
            t = threading.Thread(target=_run_agent, daemon=True)
            t.start()
            t.join(timeout=agent_timeout)
            if t.is_alive():
                # Agent 超时 —— 尝试通过 _interrupt_requested 中断
                logger.warning(
                    "Agent node '%s' timed out after %.0fs, requesting interrupt",
                    node.get("id", "?"), agent_timeout,
                )
                try:
                    agent._interrupt_requested = True
                except Exception:
                    pass
                # 再给 5s 让 agent 优雅退出
                t.join(timeout=5.0)
                if t.is_alive():
                    logger.error(
                        "Agent node '%s' did not respond to interrupt after %.0fs timeout",
                        node.get("id", "?"), agent_timeout,
                    )
                return {"__error__": f"Agent timed out after {int(agent_timeout)}s"}
            if _agent_exc[0] is not None:
                raise _agent_exc[0]
            result = _agent_result[0]
        else:
            result = agent.run_conversation(prompt, system_message=system_prompt)

        final_response = result.get("final_response", "") or ""

        output_keys = node.get("outputs", ["text"])
        logger.info("Agent[%s]: run_conversation completed — response length=%d chars, output_keys=%s",
                    node_id, len(final_response), output_keys)
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
        logger.error("Agent[%s]: execution failed — %s", node_id, e, exc_info=True)
        return {"__error__": f"Agent execution failed: {e}"}
    finally:
        # Unregister this agent from the registry so the heartbeat
        # snapshot stops reporting it and the agent object can be GC'd.
        if registry_token is not None:
            try:
                unregister_agent(registry_token)
                logger.info("Agent[%s]: unregistered from heartbeat (token=%s)",
                            node_id, registry_token[:8])
            except Exception:
                pass

        # Prevent session_id from leaking to the next agent that happens
        # to land on this reused ThreadPoolExecutor worker thread.
        # ``set_session_context`` is backed by ``threading.local`` in
        # ``hermes_logging``; without an explicit clear, any log record
        # emitted between this worker being reassigned and the next
        # agent's ``run_conversation`` setting its own session_id would
        # inherit the previous agent's tag.
        try:
            from hermes_logging import clear_session_context
            clear_session_context()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Template node executor
# ---------------------------------------------------------------------------

def _run_template_node(node: Dict[str, Any], pool: VarPool) -> Dict[str, Any]:
    """Render a string template with variable substitution."""
    template_str = node.get("template", "")
    rendered = render(template_str, pool)
    node_id = node.get("id", "?")
    logger.info("Template[%s]: rendered, output length=%d chars", node_id, len(str(rendered)))

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
    node_id = node.get("id", "?")

    if result:
        logger.info("If[%s]: condition evaluated to TRUE → branch 'then' → '%s'",
                    node_id, node.get("then", ""))
        return {"__branch__": "then", "__target__": node.get("then", "")}
    else:
        logger.info("If[%s]: condition evaluated to FALSE → branch 'else' → '%s'",
                    node_id, node.get("else", ""))
        return {"__branch__": "else", "__target__": node.get("else", "")}


# ---------------------------------------------------------------------------
# Iteration node executor
# ---------------------------------------------------------------------------

def _run_iteration_node(node: Dict[str, Any], pool: VarPool,
                        session_db=None,
                        progress_callback=None) -> Dict[str, Any]:
    """Loop over a list and execute sub_steps for each item.

    When max_concurrent > 1, sub_steps are executed concurrently using
    ThreadPoolExecutor. In concurrent mode, workdir is set once at the
    iteration level (no item variables), and sub_steps must not have
    their own workdir (validated at creation time).
    """
    node_id = node.get("id", "?")

    # Resolve items list
    items_expr = node.get("items", "")
    items = render(items_expr, pool)

    # If items is a string, try parsing as JSON
    if isinstance(items, str):
        try:
            items = json.loads(items)
        except json.JSONDecodeError:
            logger.error("Iteration[%s]: cannot resolve items — %s", node_id, items_expr)
            return {"__error__": f"Cannot resolve iteration items: {items_expr}"}

    if not isinstance(items, list):
        items = [items]

    item_var = node.get("item_var", "item")
    sub_steps = node.get("sub_steps", [])
    max_concurrent = node.get("max_concurrent", 1)
    stop_on_error = node.get("stop_on_error", False)
    output_key = (node.get("outputs") or ["results"])[0]

    logger.info("Iteration[%s]: %d items, item_var=%s, max_concurrent=%d, sub_steps=%d, stop_on_error=%s",
                node_id, len(items), item_var, max_concurrent, len(sub_steps), stop_on_error)

    # Iteration-level workdir template
    iter_workdir_template = node.get("workdir")

    if max_concurrent > 1:
        return _run_iteration_concurrent(
            node, pool, items, item_var, sub_steps, max_concurrent,
            stop_on_error, iter_workdir_template, output_key,
            session_db=session_db, progress_callback=progress_callback,
        )
    else:
        return _run_iteration_sequential(
            node, pool, items, item_var, sub_steps,
            stop_on_error, iter_workdir_template, output_key,
            session_db=session_db, progress_callback=progress_callback,
        )


def _run_iteration_sequential(node, pool, items, item_var, sub_steps,
                              stop_on_error, iter_workdir_template, output_key,
                              session_db=None, progress_callback=None):
    """顺序执行迭代（max_concurrent <= 1），sub_step 可配置自己的 workdir。"""
    results = []
    errors = []

    for i, item in enumerate(items):
        logger.info("Iteration[%d/%d]: processing item", i + 1, len(items))

        # Notify: iteration item starting
        if progress_callback:
            progress_callback(f"{node.get('id', 'iter')}[{i+1}/{len(items)}]", 
                             "iteration_item", "start", None, 0, depth=0)

        # Push iteration context BEFORE rendering workdir,
        # so {{cluster}} etc. can be resolved.
        pool.push_iter_context(item_var, item, i, len(items))

        # Render iteration-level workdir with current item context.
        # _ensure_workdir 会创建不存在的目录，使 /tmp/runs/{{item.name}}
        # 这种动态路径在 sub_step 拿不到不存在的 cwd。
        iter_workdir = None
        if iter_workdir_template:
            iter_workdir = _ensure_workdir(
                render(str(iter_workdir_template), pool),
                node.get("id", "?"), "iteration.workdir")
        logger.info("Iteration item %d/%d (%s=%r): iter_workdir=%s",
                     i + 1, len(items), item_var, item, iter_workdir)

        # Execute sub_steps sequentially
        iter_output = {}
        iter_error = None

        for sub_step in sub_steps:
            sub_type = sub_step.get("type", "agent")
            sub_id = sub_step.get("id", f"sub_{i}")

            # Notify: sub_step starting (depth=1 for nested sub_step)
            if progress_callback:
                progress_callback(sub_id, sub_type, "start", None, 0, depth=1)

            sub_start = time.time()

            # Per-sub-step workdir (overrides iteration-level).
            sub_workdir = sub_step.get("workdir")
            if sub_workdir:
                sub_workdir = _ensure_workdir(
                    render(str(sub_workdir), pool),
                    sub_id, "sub_step.workdir")
            effective_workdir = sub_workdir or iter_workdir
            logger.info("Sub-step '%s' workdir: sub=%s, iter=%s, effective=%s",
                         sub_id, sub_workdir, iter_workdir, effective_workdir)

            if sub_type == "script":
                sub_result = _run_script_node(sub_step, pool,
                                              default_workdir=effective_workdir)
            elif sub_type == "shell":
                sub_result = _run_shell_node(sub_step, pool,
                                             default_workdir=effective_workdir)
            elif sub_type == "agent":
                sub_result = _run_agent_node(sub_step, pool, session_db=session_db,
                                             default_workdir=effective_workdir)
            elif sub_type == "template":
                sub_result = _run_template_node(sub_step, pool)
            else:
                logger.error("Iteration[%d/%d] sequential: unknown sub-step type '%s' for '%s'",
                             i + 1, len(items), sub_type, sub_id)
                sub_result = {"__error__": f"Unknown sub-step type: {sub_type}"}

            sub_duration = time.time() - sub_start
            if "__error__" in sub_result:
                logger.warning("Iteration[%d/%d] sequential: sub-step '%s' FAILED — %s (took %.2fs)",
                              i + 1, len(items), sub_id, sub_result["__error__"], sub_duration)
            else:
                logger.info("Iteration[%d/%d] sequential: sub-step '%s' OK (took %.2fs)",
                           i + 1, len(items), sub_id, sub_duration)
            iter_output[sub_id] = sub_result
            pool.set_step_output(sub_id, sub_result)

            # Notify: sub_step completed
            if progress_callback:
                has_error = "__error__" in sub_result
                progress_callback(sub_id, sub_type, "ok" if not has_error else "error", 
                                sub_result, round(sub_duration, 2), depth=1)

            if "__error__" in sub_result:
                iter_error = sub_result["__error__"]
                if stop_on_error:
                    break

        # Pop iteration context
        pool.pop_iter_context()

        if iter_error:
            errors.append({"index": i, "item": item, "error": iter_error})
            logger.warning("Iteration[%s] sequential: item %d/%d FAILED — %s",
                          node.get('id', '?'), i + 1, len(items), iter_error)
            if progress_callback:
                progress_callback(f"{node.get('id', 'iter')}[{i+1}/{len(items)}]",
                                "iteration_item", "error", {"__error__": iter_error}, 0, depth=0)
            if stop_on_error:
                break
        else:
            results.append({
                "index": i,
                "item": item,
                "output": iter_output,
            })

    # Iteration complete — workdir passed via arg, nothing to clean up at env level
    output = {output_key: results}
    if errors:
        output["__errors__"] = errors

    logger.info("Iteration[%s] sequential: completed — %d/%d items succeeded, %d errors",
                node.get('id', '?'), len(results), len(items), len(errors))
    return output


def _run_iteration_concurrent(node, pool, items, item_var, sub_steps,
                              max_concurrent, stop_on_error,
                              iter_workdir_template, output_key,
                              session_db=None, progress_callback=None):
    """并发执行迭代（max_concurrent > 1）。

    约束：
      - 迭代层 workdir 模板在每个线程内独立渲染，支持 {{item}}/{{index}} 等变量，
        每个 item 可拥有独立的工作目录
      - sub_step 可配置自己的 workdir（覆盖迭代层），行为与顺序模式对齐
      - 每个线程使用独立的 VarPool 副本，避免变量竞争
      - workdir 通过构造参数传递给 AIAgent/subprocess，不依赖进程级环境变量，
        天然线程安全
      - 所有 item 执行完毕后才继续外层节点
    """
    # 空列表直接返回
    if not items:
        return {output_key: []}

    results = [None] * len(items)  # 预分配，保证顺序
    errors_list = []
    cancel_event = threading.Event()

    def _process_item(i, item):
        """单个 item 的处理逻辑，在线程中执行。"""
        if cancel_event.is_set():
            return

        _thread_name = threading.current_thread().name
        logger.info("Iteration[%d/%d] [thread=%s]: processing item (concurrent)", i + 1, len(items), _thread_name)

        # Notify: iteration item starting
        if progress_callback:
            progress_callback(f"{node.get('id', 'iter')}[{i+1}/{len(items)}]",
                             "iteration_item", "start", None, 0, depth=0)

        # 每个线程使用独立的 VarPool 副本
        item_pool = copy.deepcopy(pool)
        item_pool.push_iter_context(item_var, item, i, len(items))

        # 线程内渲染迭代层 workdir（支持 {{item}}/{{index}} 等变量）
        # _ensure_workdir 会 expanduser + resolve + mkdir，避免动态生成的
        # workdir 不存在时 sub_step 的 subprocess/AIAgent 报错。
        iter_workdir = None
        if iter_workdir_template:
            iter_workdir = _ensure_workdir(
                render(str(iter_workdir_template), item_pool),
                node.get("id", "?"), "iteration.workdir")
        logger.info("Iteration[%d/%d] [thread=%s]: iter_workdir=%s",
                    i + 1, len(items), _thread_name, iter_workdir)

        iter_output = {}
        iter_error = None

        for sub_step in sub_steps:
            if cancel_event.is_set():
                break

            sub_type = sub_step.get("type", "agent")
            sub_id = sub_step.get("id", f"sub_{i}")

            # Notify: sub_step starting
            if progress_callback:
                progress_callback(sub_id, sub_type, "start", None, 0, depth=1)

            sub_start = time.time()

            # sub_step 可配置自己的 workdir（覆盖迭代层），用当前线程的 item_pool 渲染
            sub_workdir = sub_step.get("workdir")
            if sub_workdir:
                sub_workdir = _ensure_workdir(render(str(sub_workdir), item_pool), sub_id, "sub_step.workdir")
            effective_workdir = sub_workdir or iter_workdir
            logger.info("Iteration[%d/%d] [thread=%s]: sub-step '%s' workdir: sub=%s, iter=%s, effective=%s",
                        i + 1, len(items), _thread_name, sub_id,
                        sub_workdir, iter_workdir, effective_workdir)

            if sub_type == "script":
                sub_result = _run_script_node(sub_step, item_pool,
                                              default_workdir=effective_workdir)
            elif sub_type == "shell":
                sub_result = _run_shell_node(sub_step, item_pool,
                                             default_workdir=effective_workdir)
            elif sub_type == "agent":
                sub_result = _run_agent_node(sub_step, item_pool, session_db=session_db,
                                             default_workdir=effective_workdir)
            elif sub_type == "template":
                sub_result = _run_template_node(sub_step, item_pool)
            else:
                logger.error("Iteration[%d/%d] [thread=%s]: unknown sub-step type '%s' for '%s'",
                             i + 1, len(items), _thread_name, sub_type, sub_id)
                sub_result = {"__error__": f"Unknown sub-step type: {sub_type}"}

            sub_duration = time.time() - sub_start
            if "__error__" in sub_result:
                logger.warning("Iteration[%d/%d] [thread=%s]: sub-step '%s' FAILED — %s (took %.2fs)",
                              i + 1, len(items), _thread_name, sub_id, sub_result["__error__"], sub_duration)
            else:
                logger.info("Iteration[%d/%d] [thread=%s]: sub-step '%s' OK (took %.2fs)",
                           i + 1, len(items), _thread_name, sub_id, sub_duration)
            iter_output[sub_id] = sub_result
            item_pool.set_step_output(sub_id, sub_result)

            # Notify: sub_step completed
            if progress_callback:
                has_error = "__error__" in sub_result
                progress_callback(sub_id, sub_type, "ok" if not has_error else "error",
                                sub_result, round(sub_duration, 2), depth=1)

            if "__error__" in sub_result:
                iter_error = sub_result["__error__"]
                if stop_on_error:
                    cancel_event.set()
                    break

        item_pool.pop_iter_context()

        if iter_error:
            errors_list.append({"index": i, "item": item, "error": iter_error})
            results[i] = {
                "index": i,
                "item": item,
                "output": iter_output,
                "__error__": iter_error,
            }
            if progress_callback:
                progress_callback(f"{node.get('id', 'iter')}[{i+1}/{len(items)}]",
                                "iteration_item", "error", {"__error__": iter_error}, 0, depth=0)
        else:
            results[i] = {
                "index": i,
                "item": item,
                "output": iter_output,
            }
            if progress_callback:
                progress_callback(f"{node.get('id', 'iter')}[{i+1}/{len(items)}]",
                                "iteration_item", "ok", None, 0, depth=0)

    # 使用 ThreadPoolExecutor 并发执行
    workers = min(max_concurrent, len(items))
    logger.info("Iteration concurrent: launching %d items with %d workers",
                len(items), workers)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {}
        for i, item in enumerate(items):
            # 关键：每个 item 在自己独立的 Context 副本中执行。
            # 否则 ThreadPoolExecutor 的 worker 会被复用，前一个 item 通过
            # AIAgent._apply_working_dir() 写入 ContextVar 的 workdir 会
            # 残留到下一个 item，导致并发场景下 workdir 串扰。
            # 使用 copy_context().run() 让每次提交都拿到主线程当前的 Context
            # 快照副本，item 内的 ContextVar 写入不会污染其他 item / worker。
            ctx = contextvars.copy_context()
            future = executor.submit(ctx.run, _process_item, i, item)
            futures[future] = i

        # 等待所有 item 完成（as_completed 不保证顺序，但 results[i] 保证了顺序）
        for future in as_completed(futures):
            idx = futures[future]
            try:
                future.result()
            except Exception as e:
                logger.error("Iteration item %d raised exception: %s", idx, e)
                results[idx] = {
                    "index": idx,
                    "item": items[idx],
                    "output": {},
                    "__error__": str(e),
                }
                errors_list.append({"index": idx, "item": items[idx], "error": str(e)})
                if stop_on_error:
                    cancel_event.set()

    # 并发迭代完成 — workdir 通过构造参数透传给每个 AIAgent，
    # 不再依赖进程级 TERMINAL_CWD，所以无需在此清理 env。

    # 合并结果到主 pool（按顺序，只合并非错误的结果）
    valid_results = [r for r in results if r is not None and "__error__" not in r]
    for r in valid_results:
        for sub_id, sub_result in r["output"].items():
            pool.set_step_output(sub_id, sub_result)

    output = {output_key: valid_results}
    if errors_list:
        output["__errors__"] = errors_list

    logger.info("Iteration[%s] concurrent: completed — %d/%d items succeeded, %d errors",
                node.get('id', '?'), len(valid_results), len(items), len(errors_list))
    return output


# ---------------------------------------------------------------------------
# Parallel node executor
# ---------------------------------------------------------------------------

def _run_parallel_node(node: Dict[str, Any], pool: VarPool,
                       session_db=None,
                       progress_callback=None) -> Dict[str, Any]:
    """Run multiple branches concurrently."""
    node_id = node.get("id", "?")
    branches = node.get("branches", [])
    if not branches:
        logger.info("Parallel[%s]: no branches defined, returning empty", node_id)
        return {}

    logger.info("Parallel[%s]: executing %d branch(es)", node_id, len(branches))
    results = {}

    if len(branches) == 1:
        # No need for threading with a single branch
        branch = branches[0]
        branch_id = branch.get("id", "branch_0")
        logger.info("Parallel[%s]: single branch '%s', running synchronously", node_id, branch_id)
        result = _execute_single_node(branch, pool, session_db=session_db,
                                      progress_callback=progress_callback)
        results[branch_id] = result
    else:
        with ThreadPoolExecutor(max_workers=len(branches)) as executor:
            futures = {}
            for idx, branch in enumerate(branches):
                branch_id = branch.get("id", f"branch_{idx}")
                # Each branch gets a copy of the pool
                branch_pool = copy.deepcopy(pool)
                # 每个 branch 在自己独立的 Context 副本中执行，
                # 避免 worker 复用导致 ContextVar（如 workdir）在分支间串扰。
                ctx = contextvars.copy_context()
                future = executor.submit(
                    ctx.run,
                    _execute_single_node, branch, branch_pool, session_db,
                    progress_callback
                )
                futures[future] = branch_id

            for future in as_completed(futures):
                branch_id = futures[future]
                try:
                    results[branch_id] = future.result()
                except Exception as e:
                    logger.error("Parallel[%s]: branch '%s' raised exception — %s",
                                node_id, branch_id, e)
                    results[branch_id] = {"__error__": str(e)}

    # 汇总各分支执行结果
    branch_errors = [bid for bid, r in results.items() if isinstance(r, dict) and "__error__" in r]
    if branch_errors:
        logger.warning("Parallel[%s]: completed — %d/%d branches have errors: %s",
                       node_id, len(branch_errors), len(branches), branch_errors)
    else:
        logger.info("Parallel[%s]: completed — all %d branches succeeded", node_id, len(branches))

    return results


# ---------------------------------------------------------------------------
# Node dispatch
# ---------------------------------------------------------------------------

def _execute_single_node(node: Dict[str, Any], pool: VarPool,
                         session_db=None,
                         progress_callback=None) -> Dict[str, Any]:
    """Dispatch to the right executor based on node type."""
    node_type = node.get("type", "agent")
    node_id = node.get("id", "?")
    logger.info("Node dispatch: '%s' (type=%s)", node_id, node_type)

    if node_type == "script":
        return _run_script_node(node, pool)
    elif node_type == "shell":
        return _run_shell_node(node, pool)
    elif node_type == "agent":
        return _run_agent_node(node, pool, session_db=session_db)
    elif node_type == "iteration":
        return _run_iteration_node(node, pool, session_db=session_db,
                                   progress_callback=progress_callback)
    elif node_type == "if":
        return _run_if_node(node, pool)
    elif node_type == "template":
        return _run_template_node(node, pool)
    elif node_type == "parallel":
        return _run_parallel_node(node, pool, session_db=session_db,
                                  progress_callback=progress_callback)
    else:
        return {"__error__": f"Unknown node type: {node_type}"}


# ---------------------------------------------------------------------------
# Workflow execution engine
# ---------------------------------------------------------------------------

def run_workflow(workflow: Dict[str, Any], inputs: Optional[Dict[str, Any]] = None,
                 session_db=None, progress_callback=None) -> Dict[str, Any]:
    """Execute a workflow definition.

    The workflow must be a YAML-parsed format with 'nodes' and 'edges'
    (produced by schema.parse_yaml_workflow or load_yaml_file).

    Args:
        workflow: The workflow definition (from schema.parse_yaml_workflow).
        inputs: Optional input variables (override workflow.inputs).
        session_db: Optional SessionDB for agent session persistence.
        progress_callback: Optional callback(node_id, node_type, status, result, duration)
                          called after each node completes. Called with status='start'
                          before execution, 'ok'/'error' after completion.

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
    logger.info("Workflow[%s]: starting — name=%r, id=%r",
                run_id, workflow.get("name", "unnamed"), workflow.get("id", ""))

    nodes = workflow.get("nodes", {})
    edges = workflow.get("edges", [])

    # Merge inputs: workflow defaults + runtime overrides
    wf_inputs = workflow.get("inputs", {})
    merged_inputs = {**wf_inputs, **(inputs or {})}
    logger.info("Workflow[%s]: inputs merged — defaults=%d, runtime=%d, total=%d keys",
                run_id, len(wf_inputs), len(inputs or {}), len(merged_inputs))

    # Validate
    if not nodes:
        logger.error("Workflow[%s]: validation failed — no nodes defined", run_id)
        return {
            "status": "error",
            "outputs": {},
            "errors": ["Workflow has no nodes"],
            "duration_seconds": 0,
            "run_id": run_id,
        }

    # Full structural validation
    try:
        validation_errors = validate_workflow(workflow)
    except Exception as exc:  # defensive — bad workflow shape
        validation_errors = [f"validate_workflow raised: {exc}"]
    if validation_errors:
        logger.error("Workflow[%s]: validation failed — %s", run_id, "; ".join(validation_errors))
        return {
            "status": "error",
            "outputs": {},
            "errors": validation_errors,
            "duration_seconds": 0,
            "run_id": run_id,
        }

    logger.info("Workflow[%s]: validation passed — %d nodes, %d edges", run_id, len(nodes), len(edges))

    # Initialize variable pool
    pool = VarPool(inputs=merged_inputs)
    logger.info("Workflow[%s]: VarPool initialized with %d input keys", run_id, len(merged_inputs))

    # Determine execution order
    # For if-node workflows, we use step-by-step execution
    # (if nodes redirect to different next steps)
    has_if_nodes = any(n.get("type") == "if" for n in nodes.values())

    if has_if_nodes:
        logger.info("Workflow[%s]: detected if-nodes → using branch execution mode", run_id)
        execution_result = _run_workflow_with_branches(
            workflow, nodes, edges, pool, run_id, session_db,
            progress_callback=progress_callback
        )
    else:
        logger.info("Workflow[%s]: no if-nodes → using linear execution mode", run_id)
        execution_result = _run_workflow_linear(
            nodes, edges, pool, run_id, session_db,
            progress_callback=progress_callback
        )

    # Save run output
    duration = round(time.time() - start_time, 2)
    execution_result["duration_seconds"] = duration
    execution_result["run_id"] = run_id

    logger.info("Workflow[%s]: finished — status=%s, duration=%.2fs, errors=%d",
                run_id, execution_result.get("status"), duration,
                len(execution_result.get("errors", [])))

    # Resolve a stable persistence key.  ``parse_yaml_workflow`` returns a
    # dict without an ``id`` field unless the caller went through
    # ``load_yaml_file``; falling back to ``"unknown"`` would collapse every
    # programmatic run into one directory and silently overwrite outputs.
    storage_id = _resolve_storage_id(workflow)

    _save_run_output(storage_id, run_id, {
        "run_id": run_id,
        "workflow_name": workflow.get("name", "unnamed"),
        "workflow_id": workflow.get("id", "") or storage_id,
        "status": execution_result["status"],
        "started_at": datetime.fromtimestamp(start_time).astimezone().isoformat(),
        "duration_seconds": duration,
        "inputs": merged_inputs,
        "outputs": execution_result.get("outputs", {}),
        "errors": execution_result.get("errors", []),
    })

    return execution_result


def _resolve_storage_id(workflow: Dict[str, Any]) -> str:
    """Pick a filesystem-safe directory name for persisting run outputs.

    Precedence: explicit ``id`` > sanitized ``name`` > ``"unknown"``.
    Only word characters, dashes and underscores are kept; everything
    else is collapsed to underscores so the directory stays within what
    every platform can create.
    """
    wf_id = workflow.get("id")
    if wf_id:
        return str(wf_id)
    name = workflow.get("name")
    if name:
        import re as _re
        slug = _re.sub(r"[^\w.-]+", "_", str(name)).strip("_")
        if slug:
            return slug
    return "unknown"


def _run_workflow_linear(nodes: Dict, edges: List, pool: VarPool,
                         run_id: str, session_db=None,
                         progress_callback=None) -> Dict[str, Any]:
    """Execute a workflow with no branching (topological sort, linear execution)."""
    try:
        execution_order = _topological_sort(nodes, edges)
        logger.info("Workflow[%s] linear: topological order — %s", run_id, execution_order)
    except ValueError as e:
        logger.error("Workflow[%s] linear: topological sort failed — %s", run_id, e)
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

        logger.info("Workflow[%s] linear: executing node '%s' (type=%s) [%d/%d]",
                    run_id, node_id, node_type,
                    execution_order.index(node_id) + 1, len(execution_order))
        node_start = time.time()

        # Notify: node starting (depth=0 — top-level node)
        if progress_callback:
            progress_callback(node_id, node_type, "start", None, 0, depth=0)

        result = _execute_single_node(node_def, pool, session_db=session_db,
                                       progress_callback=progress_callback)

        node_duration = time.time() - node_start
        pool.set_step_output(node_id, result)

        all_outputs[node_id] = {
            "result": result,
            "duration_seconds": round(node_duration, 2),
            "type": node_type,
        }

        # Notify: node completed
        has_error = "__error__" in result
        if progress_callback:
            progress_callback(node_id, node_type, "ok" if not has_error else "error",
                              result, round(node_duration, 2), depth=0)

        if has_error:
            error_msg = f"Node '{node_id}' failed: {result['__error__']}"
            errors.append(error_msg)
            logger.error("Workflow[%s] linear: node '%s' FAILED — %s (took %.2fs)",
                        run_id, node_id, error_msg, node_duration)
            break
        else:
            logger.info("Workflow[%s] linear: node '%s' completed OK (took %.2fs)",
                       run_id, node_id, node_duration)

    return {
        "status": "error" if errors else "ok",
        "outputs": all_outputs,
        "errors": errors,
    }


def _run_workflow_with_branches(workflow: Dict, nodes: Dict, edges: List,
                                pool: VarPool, run_id: str,
                                session_db=None,
                                progress_callback=None) -> Dict[str, Any]:
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
        logger.error("Workflow[%s] branches: cannot find entry point", run_id)
        return {
            "status": "error",
            "outputs": {},
            "errors": ["Cannot find entry point in workflow"],
        }

    logger.info("Workflow[%s] branches: entry points — %s", run_id, entry_points)

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
        logger.info("Workflow[%s] branches: running node '%s' (type=%s)", run_id, node_id, node_type)

        node_start = time.time()

        # Notify: node starting (depth=0 — top-level node)
        if progress_callback:
            progress_callback(node_id, node_type, "start", None, 0, depth=0)

        result = _execute_single_node(node_def, pool, session_db=session_db,
                                       progress_callback=progress_callback)
        node_duration = time.time() - node_start

        pool.set_step_output(node_id, result)
        all_outputs[node_id] = {
            "result": result,
            "duration_seconds": round(node_duration, 2),
            "type": node_type,
        }

        # Notify: node completed
        has_error = "__error__" in result
        if progress_callback:
            progress_callback(node_id, node_type, "ok" if not has_error else "error",
                              result, round(node_duration, 2), depth=0)

        if has_error:
            error_msg = f"Node '{node_id}' failed: {result['__error__']}"
            errors.append(error_msg)
            logger.error("Workflow[%s] branches: node '%s' FAILED — %s (took %.2fs)",
                        run_id, node_id, error_msg, node_duration)
            if not workflow.get("continue_on_error", False):
                break
        else:
            logger.info("Workflow[%s] branches: node '%s' completed OK (took %.2fs)",
                       run_id, node_id, node_duration)

        # Determine next nodes
        if node_type == "if" and "__target__" in result:
            # Follow the chosen branch
            target = result["__target__"]
            branch = result.get("__branch__", "?")
            logger.info("Workflow[%s] branches: if-node '%s' chose branch '%s' → '%s'",
                       run_id, node_id, branch, target)
            if target and target in nodes:
                queue.append(target)
            else:
                logger.warning("Workflow[%s] branches: if-node '%s' target '%s' not found in nodes",
                              run_id, node_id, target)
        else:
            # Follow all outgoing edges
            next_ids = adjacency.get(node_id, [])
            if next_ids:
                logger.info("Workflow[%s] branches: node '%s' → next: %s",
                           run_id, node_id, [n for n in next_ids if n not in visited])
            for next_id in next_ids:
                if next_id not in visited:
                    queue.append(next_id)

    logger.info("Workflow[%s] branches: execution complete — visited %d nodes, status=%s, errors=%d",
               run_id, len(visited), "error" if errors else "ok", len(errors))
    return {
        "status": "error" if errors else "ok",
        "outputs": all_outputs,
        "errors": errors,
    }

def _save_run_output(workflow_id: str, run_id: str, data: Dict[str, Any]):
    """Save run output to file."""
    _ensure_dirs()
    run_dir = _get_workflows_dir() / workflow_id / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    output_file = run_dir / "output.json"
    tmp_file = run_dir / ".output.json.tmp"
    logger.info("Workflow[%s]: saving run output to %s", run_id, output_file)

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


# CRUD wrappers moved to workflow/__init__.py — import from there
