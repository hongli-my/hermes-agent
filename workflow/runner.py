"""
workflow/runner.py -- Python runner for Lua API

由 Lua workflow.lua 调用，通过 stdin/stdout 交换 JSON。

调用方式（与 Lua 原来内嵌脚本兼容）:
    echo '{"workflow_id":"xxx","inputs":{}}' | python -m workflow.runner

也支持 yaml_path / yaml_text 三选一参数。
"""

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

# hermes-agent src 入 path
_SRC_DIR = Path(__file__).parent.parent
if _SRC_DIR.exists():
    sys.path.insert(0, str(_SRC_DIR))


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home()
    except Exception:
        return Path.home() / ".hermes"


def _python_bin() -> str:
    """返回 hermes venv 的 python3（不存在则用系统 python3）"""
    venv_python = _hermes_home() / "hermes" / "bin" / "python3"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable or "python3"


def run_workflow(
    workflow_id: Optional[str] = None,
    yaml_path: Optional[str] = None,
    yaml_text: Optional[str] = None,
    inputs: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    执行工作流并返回结果（dict，包含 run_id, status, duration_seconds, outputs_count）。

    三选一参数:
        workflow_id: 从已存储的工作流运行
        yaml_path:   从本地 YAML 文件运行
        yaml_text:   直接传入 YAML 文本运行
    """
    from workflow.engine import run_workflow as _engine_run
    from workflow.schema import load_yaml_file
    from workflow.store import get_workflow

    if yaml_path:
        if not os.path.exists(yaml_path):
            raise ValueError(f"yaml file not found: {yaml_path}")
        wf = load_yaml_file(yaml_path)
        result = _engine_run(wf, inputs=inputs or {})
    elif workflow_id:
        wf = get_workflow(workflow_id)
        if not wf:
            raise ValueError(f"workflow not found: {workflow_id}")
        result = _engine_run(wf, inputs=inputs or {})
    elif yaml_text:
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            f.write(yaml_text)
            tmp_path = f.name
        try:
            wf = load_yaml_file(tmp_path)
            result = _engine_run(wf, inputs=inputs or {})
        finally:
            os.unlink(tmp_path)
    else:
        raise ValueError("one of workflow_id, yaml_path, or yaml_text is required")

    return {
        "ok": True,
        "run_id": result.get("run_id"),
        "status": result.get("status"),
        "duration_seconds": result.get("duration_seconds"),
        "outputs_count": len(result.get("outputs", {})),
    }


def main():
    """stdin JSON → stdout JSON（兼容 Lua io.popen 方式）"""
    raw = sys.stdin.read()
    if not raw:
        print(json.dumps({"ok": False, "error": "empty input"}))
        sys.exit(0)

    try:
        params = json.loads(raw)
    except json.JSONDecodeError as e:
        print(json.dumps({"ok": False, "error": f"invalid json: {e}"}))
        sys.exit(0)

    try:
        result = run_workflow(
            workflow_id=params.get("workflow_id"),
            yaml_path=params.get("yaml_path"),
            yaml_text=params.get("yaml_text"),
            inputs=params.get("inputs"),
        )
        print(json.dumps(result))
    except Exception as e:
        import traceback

        print(
            json.dumps(
                {"ok": False, "error": str(e), "traceback": traceback.format_exc()}
            )
        )


if __name__ == "__main__":
    main()