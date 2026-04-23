import json
from workflow.engine import run_workflow

# 简单工作流：只有 script 节点
wf = {
    "name": "test_script_only",
    "nodes": {
        "fetch": {
            "type": "script",
            "script": "fetch_news.py",
            "outputs": ["articles"]
        }
    },
    "edges": []
}
result = run_workflow(wf)
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
