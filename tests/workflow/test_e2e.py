#!/usr/bin/env python3
"""Test workflow engine with inline YAML definitions (no external files needed)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from workflow.schema import parse_yaml_workflow
from workflow.engine import run_workflow

# Simple workflow: script nodes only
yaml_text = """
name: test_script_only
steps:
  - id: fetch
    type: script
    script: fetch_news.py
    outputs:
      - articles
"""

wf = parse_yaml_workflow(yaml_text)
result = run_workflow(wf)
print(json.dumps(result, indent=2, ensure_ascii=False, default=str))
