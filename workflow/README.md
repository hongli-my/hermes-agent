# Hermes Workflow Engine

用 YAML 声明流程，用 Python 脚本做原子动作，引擎自动编排执行。

```
一个工作流 = 一个 YAML 文件（声明步骤和流转）
一个脚本   = 一个 Python 文件（stdin JSON → stdout JSON）
改流程     = 改 YAML，不动脚本
加功能     = 加一个脚本 + YAML 里加一步
定时执行   = cron job 触发工作流
```

---

## 快速开始

### 1. 写一个脚本

脚本放在 `~/.hermes/scripts/` 下，按领域建子目录：

```python
# ~/.hermes/scripts/stocks/collect.py
"""采集股票行情
stdin:  {"date": "2026-04-24", "stocks": ["AAPL", "GOOGL"]}
stdout: {"stocks": [{"symbol": "AAPL", "price": 187.5, "change_pct": 1.2}]}
"""
import sys, json

def main():
    params = json.loads(sys.stdin.read())
    # ... 你的采集逻辑 ...
    result = {"stocks": [
        {"symbol": s, "price": 100, "change_pct": 0}
        for s in params.get("stocks", [])
    ]}
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
```

**规则：stdin 读 JSON，stdout 写 JSON，stderr 写日志。单一职责。**

### 2. 写一个工作流 YAML

```yaml
# ~/.hermes/workflows/daily-stock-report.yaml
name: 每日研报
description: 采集行情 → 分析 → 通知

inputs:
  date: "2026-04-24"

steps:
  - id: collect
    type: script
    script: stocks/collect.py
    input:
      date: "{{inputs.date}}"
      stocks: ["AAPL", "GOOGL", "TSLA"]

  - id: analyze
    type: agent
    prompt: |
      分析以下股票数据，给出趋势判断和操作建议：
      {{steps.collect.output}}

  - id: notify
    type: script
    script: notify/telegram.py
    input:
      message: "{{steps.analyze.output.text}}"
```

### 3. 运行

```bash
# 直接运行 YAML 文件（不需要注册）
hermes workflow run-file ~/.hermes/workflows/daily-stock-report.yaml

# 带参数覆盖
hermes workflow run-file ~/.hermes/workflows/daily-stock-report.yaml date=2026-04-25

# 先注册再运行
hermes workflow create ~/.hermes/workflows/daily-stock-report.yaml
hermes workflow run daily-stock-report date=2026-04-25

# 验证 YAML 是否合法
hermes workflow validate ~/.hermes/workflows/daily-stock-report.yaml
```

---

## 目录结构

```
~/.hermes/
├── workflows/                        ← 工作流 YAML 定义
│   ├── daily-stock-report.yaml
│   ├── news-monitor.yaml
│   └── portfolio-rebalance.yaml
│
├── scripts/                          ← 原子脚本（按领域分目录）
│   ├── stocks/
│   │   ├── collect.py                ← 采集行情
│   │   ├── analyze.py                ← 单股分析
│   │   └── filter.py                 ← 筛选股票
│   ├── news/
│   │   ├── fetch.py                  ← 抓取新闻
│   │   └── classify.py               ← 分类新闻
│   ├── notify/
│   │   ├── email.py                  ← 发邮件
│   │   └── telegram.py               ← 发 Telegram
│   └── common/
│       └── format_report.py          ← 公共工具
│
└── workflows/ (运行时存储)            ← 引擎自动生成
    ├── daily-stock-report/
    │   ├── workflow.json             ← 内部解析格式
    │   └── runs/
    │       ├── a1b2c3d4e5f6/
    │       │   └── output.json       ← 执行记录
    │       └── f6e5d4c3b2a1/
    │           └── output.json
    └── news-monitor/
        └── ...
```

---

## 节点类型

### script — 运行 Python 脚本

```yaml
- id: collect
  type: script
  script: stocks/collect.py       # 相对于 ~/.hermes/scripts/
  input:                          # 传给脚本的 stdin JSON
    date: "{{inputs.date}}"
    stocks: ["AAPL", "GOOGL"]
  timeout: 120                    # 超时秒数，默认 120
```

脚本约定：
- `stdin` 接收 `input` 字段的 JSON
- `stdout` 输出 JSON 结果
- `stderr` 输出日志（引擎会记录）
- 退出码非 0 视为失败
- 脚本路径必须在 `~/.hermes/scripts/` 下（安全限制）

### agent — 调用 AIAgent

```yaml
- id: analyze
  type: agent
  prompt: |
    分析以下数据：{{steps.collect.output}}
    给出趋势判断和操作建议。
  model: anthropic/claude-sonnet-4     # 可选，默认用配置文件模型
  provider: openrouter                  # 可选
  system_prompt: "你是一个股票分析师"    # 可选
  skills: [westock-data]               # 可选，加载技能
  max_iterations: 30                    # 可选，工具调用轮次上限
  output_format: json                   # 可选，提示 agent 输出 JSON
  outputs: [text, score]               # 可选，从 agent 响应中提取的 key
```

agent 节点的输出：
- `outputs` 只有一个 key（如 `[text]`）：整个响应存为该 key
- `outputs` 有多个 key：尝试解析响应为 JSON，按 key 提取
- 默认 `outputs: [text]`

### if — 条件分支

```yaml
- id: check
  type: if
  condition: "len({{steps.classify.output.critical}}) > 0"
  then: alert_critical     # 条件为真，跳到这步
  else: check_important    # 条件为假，跳到这步

- id: alert_critical
  type: agent
  prompt: "紧急！分析：{{steps.classify.output.critical}}"

- id: check_important
  type: if
  condition: "len({{steps.classify.output.important}}) > 0"
  then: daily_digest
  else: silent
```

分支后如果两条路径需要汇合到同一个步骤，引擎自动处理合并边。

支持的条件表达式：
```yaml
condition: "len({{steps.X.output.items}}) > 0"       # 列表长度
condition: "{{steps.X.output.count}} >= 5"            # 数值比较
condition: "{{steps.X.output.success}} == true"       # 布尔判断
```

### iteration — 循环迭代

```yaml
- id: analyze_all
  type: iteration
  items: "{{steps.filter.output.stocks}}"    # 要迭代的列表
  item_var: stock                             # 循环变量名，默认 item
  max_concurrent: 1                           # 并发数，默认 1（串行）
  stop_on_error: false                        # 出错是否停止，默认 false
  sub_steps:                                  # 每个元素执行的步骤
    - id: deep_analysis
      type: agent
      prompt: "分析这只股票：{{stock}}"
      skills: [westock-data]
    - id: score
      type: script
      script: stocks/analyze.py
      input:
        data: "{{stock}}"
```

迭代输出：
```json
{
  "results": [
    {"index": 0, "item": {...}, "output": {"deep_analysis": {...}, "score": {...}}},
    {"index": 1, "item": {...}, "output": {"deep_analysis": {...}, "score": {...}}}
  ]
}
```

在 sub_steps 中可用 `{{stock}}`（你的 item_var）访问当前元素，
也可用 `{{__iter__.index}}` 和 `{{__iter__.total}}` 访问迭代元数据。

### parallel — 并行执行

```yaml
- id: parallel_analyze
  type: parallel
  branches:
    - id: tech_analysis
      type: agent
      prompt: "技术面分析：{{steps.portfolio.output}}"
    - id: fundamental_analysis
      type: agent
      prompt: "基本面分析：{{steps.portfolio.output}}"
    - id: risk_analysis
      type: script
      script: stocks/analyze.py
      input:
        portfolio: "{{steps.portfolio.output}}"
        mode: "risk"
```

各分支并行运行，结果汇总到 `steps.parallel_analyze.output.<branch_id>`。

### template — 字符串模板渲染

```yaml
- id: format
  type: template
  template: |
    # 每日研报 {{inputs.date}}
    ## 大盘
    {{steps.clean.output.indices}}
    ## 重点个股
    {{steps.analyze.output.text}}
  outputs: [text]
```

纯字符串拼接，不需要 LLM 或脚本。

---

## 变量系统

所有节点的 `input`、`prompt`、`condition`、`template` 字段都支持变量替换：

| 变量模式 | 说明 | 示例 |
|---------|------|------|
| `{{inputs.key}}` | 工作流输入参数 | `{{inputs.date}}` |
| `{{steps.X.output}}` | 某步完整输出 | `{{steps.collect.output}}` |
| `{{steps.X.output.key}}` | 某步输出的某个字段 | `{{steps.filter.output.stocks}}` |
| `{{env.VAR}}` | 环境变量 | `{{env.TELEGRAM_CHAT_ID}}` |
| `{{item_var}}` | 迭代中的当前元素 | `{{stock}}` |
| `{{__iter__.index}}` | 迭代索引 | `0, 1, 2...` |
| `{{__iter__.total}}` | 迭代总数 | `5` |

**自动类型转换**：如果整个值是 `{{steps.X.output.stocks}}`，变量解析为列表/字典后直接传入（不会变成字符串）。

---

## 完整示例

### 每日研报（线性流程）

```yaml
name: 每日研报
description: 采集行情 → 清洗 → 筛选 → 逐股分析 → 汇总 → 通知

inputs:
  date: "2026-04-24"

steps:
  - id: collect
    type: script
    script: stocks/collect.py
    input:
      date: "{{inputs.date}}"
      stocks: ["AAPL", "GOOGL", "TSLA"]

  - id: clean
    type: agent
    prompt: |
      清洗以下股票数据，去除停牌和异常值，补全缺失字段：
      {{steps.collect.output}}
    output_format: json

  - id: filter
    type: script
    script: stocks/filter.py
    input:
      stocks: "{{steps.clean.output.stocks}}"
      rules:
        min_volume: 1000000
        price_change_pct_gt: 3

  - id: analyze
    type: iteration
    items: "{{steps.filter.output.stocks}}"
    item_var: stock
    sub_steps:
      - id: deep_analysis
        type: agent
        prompt: "深度分析：{{stock}}"
        skills: [westock-data]

  - id: report
    type: agent
    prompt: |
      根据分析结果生成今日研报：
      {{steps.analyze.output}}
    model: anthropic/claude-sonnet-4

  - id: format
    type: script
    script: common/format_report.py
    input:
      title: "每日研报 {{inputs.date}}"
      content: "{{steps.report.output.text}}"

  - id: notify
    type: script
    script: notify/telegram.py
    input:
      message: "{{steps.format.output}}"
```

```
collect → clean → filter → analyze[deep_analysis × N] → report → format → notify
 脚本    agent    脚本       迭代(agent)             agent    脚本     脚本
```

### 新闻监控（条件分支）

```yaml
name: 新闻监控
description: 抓新闻 → 分类 → 有重要新闻才通知

steps:
  - id: fetch
    type: script
    script: news/fetch.py
    input:
      sources: ["reuters", "sina"]
      keywords: ["AI", "芯片"]
      hours: 4

  - id: classify
    type: agent
    prompt: |
      分类新闻：critical / important / normal
      输出 JSON：{"critical": [...], "important": [...], "normal": [...]}
      新闻：{{steps.fetch.output}}
    output_format: json

  - id: check_critical
    type: if
    condition: "len({{steps.classify.output.critical}}) > 0"
    then: alert_critical
    else: check_important

  - id: alert_critical
    type: agent
    prompt: "紧急分析：{{steps.classify.output.critical}}"

  - id: send_alert
    type: script
    script: notify/telegram.py
    input:
      message: "🚨 {{steps.alert_critical.output.text}}"

  - id: check_important
    type: if
    condition: "len({{steps.classify.output.important}}) > 0"
    then: daily_digest
    else: silent

  - id: daily_digest
    type: script
    script: notify/email.py
    input:
      subject: "新闻速递"
      body: "{{steps.classify.output.important}}"

  - id: silent
    type: template
    template: "无重要新闻"
```

```
fetch → classify → check_critical
                      ├─ YES → alert_critical → send_alert
                      └─ NO  → check_important
                                 ├─ YES → daily_digest
                                 └─ NO  → silent
```

### 仓位调仓（并行 + 聚合）

```yaml
name: 仓位调仓建议
description: 并行分析 → 聚合 → 调仓方案

steps:
  - id: portfolio
    type: script
    script: stocks/collect.py
    input:
      mode: "portfolio"

  - id: parallel_analyze
    type: parallel
    branches:
      - id: tech
        type: agent
        prompt: "技术面分析：{{steps.portfolio.output}}"
      - id: fundamental
        type: agent
        prompt: "基本面分析：{{steps.portfolio.output}}"
      - id: risk
        type: script
        script: stocks/analyze.py
        input:
          portfolio: "{{steps.portfolio.output}}"
          mode: "risk"

  - id: rebalance
    type: agent
    prompt: |
      基于三方分析，给出调仓建议：
      技术面：{{steps.parallel_analyze.output.tech}}
      基本面：{{steps.parallel_analyze.output.fundamental}}
      风险：{{steps.parallel_analyze.output.risk}}

  - id: notify
    type: script
    script: notify/email.py
    input:
      subject: "调仓建议"
      body: "{{steps.rebalance.output.text}}"
```

```
portfolio → parallel_analyze ──┬─ tech (agent)
                               ├─ fundamental (agent)  ──→ rebalance → notify
                               └─ risk (script)
```

---

## CLI 命令

```bash
# 列出所有工作流
hermes workflow list

# 查看工作流定义
hermes workflow show <workflow_id>

# 从 YAML 创建（注册到存储）
hermes workflow create <file.yaml>

# 删除工作流（含所有执行记录）
hermes workflow delete <workflow_id>

# 运行已注册的工作流
hermes workflow run <workflow_id>
hermes workflow run <workflow_id> date=2026-04-25 mode=fast

# 直接运行 YAML 文件（不需要注册）
hermes workflow run-file <yaml_path>
hermes workflow run-file <yaml_path> date=2026-04-25

# 查看执行记录
hermes workflow runs <workflow_id>

# 验证 YAML 语法
hermes workflow validate <file.yaml>
```

---

## 定时执行（配合 Cron）

工作流可以和 Hermes Cron 系统结合，定时执行：

**方式一：脚本触发**

```bash
# 创建一个触发脚本 ~/.hermes/scripts/run_workflow.py
import sys, json, subprocess
params = json.loads(sys.stdin.read())
workflow_id = params["workflow_id"]
inputs = params.get("inputs", {})

# 调用 hermes CLI
cmd = ["hermes", "workflow", "run", workflow_id]
for k, v in inputs.items():
    cmd.append(f"{k}={v}")
result = subprocess.run(cmd, capture_output=True, text=True)
print(json.dumps({"ok": result.returncode == 0, "output": result.stdout}))
```

然后创建 cron job：
```bash
hermes cron create \
  --name "每日研报" \
  --schedule "0 9 * * *" \
  --prompt "运行脚本 run_workflow.py，参数：{\"workflow_id\": \"daily-stock-report\", \"inputs\": {\"date\": \"today\"}}"
```

**方式二：直接用 agent 触发**

Cron job 的 prompt 直接写"执行工作流 daily-stock-report"，agent 会调用 workflow 工具。

---

## 架构设计

```
workflow/
├── __init__.py      ← 统一导出
├── schema.py        ← YAML → 内部格式（nodes + edges DAG）
├── renderer.py      ← 变量渲染引擎（{{}} 替换 + 条件求值）
├── engine.py        ← 执行引擎（6种节点执行器 + 拓扑排序 + 分支执行）
├── store.py         ← CRUD + 运行记录持久化
├── cli.py           ← hermes workflow 子命令
└── README.md        ← 本文档
```

核心流程：

```
YAML 文件 → schema.py 解析 → nodes{} + edges[] DAG
                              ↓
                     engine.py 执行 → VarPool 变量池
                              ↓          ↑
                   拓扑排序/分支追踪   renderer.py 变量渲染
                              ↓
                    逐节点执行 → 结果存 VarPool → 下游节点可引用
                              ↓
                    store.py 保存执行记录 → ~/.hermes/workflows/{id}/runs/
```

两种执行模式：
- **线性模式**：无 if 节点时，拓扑排序后顺序执行
- **分支模式**：有 if 节点时，逐步执行，根据 if 结果选择下一步

---

## 脚本编写规范

```python
#!/usr/bin/env python3
"""一句话描述这个脚本做什么

stdin:  {"key1": "value1", "key2": "value2"}
stdout: {"result_key": "result_value"}
"""
import sys
import json


def main():
    # 1. 读 stdin
    params = json.loads(sys.stdin.read())

    # 2. 取参数
    key1 = params.get("key1")

    # 3. 做事
    result = do_something(key1)

    # 4. 写 stdout（必须是合法 JSON）
    print(json.dumps({"result_key": result}, ensure_ascii=False))


def do_something(key1):
    # 实际逻辑
    return "done"


if __name__ == "__main__":
    main()
```

注意事项：
- **只用标准库**：脚本应只依赖 Python 标准库，避免额外安装
- **单一职责**：一个脚本只做一件事
- **超时意识**：网络请求等操作设合理超时
- **错误处理**：异常时输出 `{"__error__": "描述"}` 而非崩溃
- **环境变量**：敏感信息（API key）用 `os.environ.get()` 读取，不硬编码
