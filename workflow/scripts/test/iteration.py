#!/usr/bin/env python
# coding=utf-8



"""采集股票行情
stdin:  {"date": "2026-04-24", "stocks": ["AAPL", "GOOGL"]}
stdout: {"stocks": [{"symbol": "AAPL", "price": 187.5, "change_pct": 1.2}]}
"""
import sys, json

def main():
    params = json.loads(sys.stdin.read())
    result = {"output": params}
    print(json.dumps(result, ensure_ascii=False))

if __name__ == "__main__":
    main()
