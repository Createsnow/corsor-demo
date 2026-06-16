#!/usr/bin/env python3
"""校验大模型 tool（函数调用）微调数据结构是否合法。

用法：
    python scripts/validate_tool_data.py <data.jsonl> [<data2.jsonl> ...]

校验内容（对应文档第 6 节 Checklist）：
  1. 每行是合法 JSON，且包含 `messages` 列表；
  2. 消息 role 合法（system/user/assistant/tool）；
  3. assistant 的 tool_calls 结构正确，arguments 是合法 JSON 字符串；
  4. 被调用的 function.name 出现在 tools 定义中（若提供了 tools）；
  5. 每个 tool_call 的 id 都有且仅有一个匹配的 tool 消息（tool_call_id）；
  6. tool 消息出现在对应 assistant(tool_calls) 之后；
  7. 调用参数满足工具 JSON Schema 中 required / enum 的基本约束。

退出码：全部通过返回 0，否则返回 1。
"""

import json
import sys
from typing import Any

VALID_ROLES = {"system", "user", "assistant", "tool"}


class Issue:
    def __init__(self, line_no: int, msg: str, level: str = "ERROR"):
        self.line_no = line_no
        self.msg = msg
        self.level = level

    def __str__(self) -> str:
        return f"  [{self.level}] 第 {self.line_no} 行: {self.msg}"


def collect_tool_schemas(tools: Any, line_no: int, issues: list) -> dict:
    """返回 {function_name: parameters_schema}。"""
    schemas: dict = {}
    if tools is None:
        return schemas
    if not isinstance(tools, list):
        issues.append(Issue(line_no, "`tools` 必须是数组"))
        return schemas
    for i, tool in enumerate(tools):
        if not isinstance(tool, dict) or tool.get("type") != "function":
            issues.append(Issue(line_no, f"tools[{i}] 必须是 type=function 的对象"))
            continue
        fn = tool.get("function")
        if not isinstance(fn, dict) or "name" not in fn:
            issues.append(Issue(line_no, f"tools[{i}].function 缺少 name"))
            continue
        schemas[fn["name"]] = fn.get("parameters", {})
    return schemas


def check_arguments(args_str: Any, schema: dict, line_no: int, fn_name: str, issues: list) -> None:
    if not isinstance(args_str, str):
        issues.append(Issue(line_no, f"{fn_name} 的 arguments 必须是 JSON 字符串，而不是 {type(args_str).__name__}"))
        return
    try:
        args = json.loads(args_str)
    except json.JSONDecodeError as e:
        issues.append(Issue(line_no, f"{fn_name} 的 arguments 不是合法 JSON: {e}"))
        return
    if not isinstance(schema, dict):
        return
    required = schema.get("required", [])
    for key in required:
        if key not in args:
            issues.append(Issue(line_no, f"{fn_name} 缺少必填参数 '{key}'"))
    props = schema.get("properties", {})
    for key, val in args.items():
        if key in props and isinstance(props[key], dict):
            enum = props[key].get("enum")
            if enum is not None and val not in enum:
                issues.append(Issue(line_no, f"{fn_name} 参数 '{key}'={val!r} 不在 enum {enum} 中", "WARN"))


def validate_record(record: Any, line_no: int, issues: list) -> None:
    if not isinstance(record, dict):
        issues.append(Issue(line_no, "样本必须是 JSON 对象"))
        return
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        issues.append(Issue(line_no, "缺少非空的 `messages` 列表"))
        return

    schemas = collect_tool_schemas(record.get("tools"), line_no, issues)
    has_tools = bool(schemas)

    pending_call_ids: dict = {}  # id -> 是否已被 tool 消息消费
    saw_any_tool_call = False

    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            issues.append(Issue(line_no, f"messages[{idx}] 必须是对象"))
            continue
        role = msg.get("role")
        if role not in VALID_ROLES:
            issues.append(Issue(line_no, f"messages[{idx}] role={role!r} 非法"))
            continue

        if role == "assistant":
            tool_calls = msg.get("tool_calls")
            content = msg.get("content")
            if tool_calls is None and (content is None or content == ""):
                issues.append(Issue(line_no, f"messages[{idx}] assistant 既无 content 也无 tool_calls"))
            if tool_calls is not None:
                if not isinstance(tool_calls, list) or not tool_calls:
                    issues.append(Issue(line_no, f"messages[{idx}].tool_calls 必须是非空数组"))
                    continue
                for c_idx, call in enumerate(tool_calls):
                    saw_any_tool_call = True
                    if not isinstance(call, dict):
                        issues.append(Issue(line_no, f"messages[{idx}].tool_calls[{c_idx}] 必须是对象"))
                        continue
                    call_id = call.get("id")
                    fn = call.get("function", {})
                    fn_name = fn.get("name") if isinstance(fn, dict) else None
                    if not call_id:
                        issues.append(Issue(line_no, f"messages[{idx}].tool_calls[{c_idx}] 缺少 id"))
                    else:
                        if call_id in pending_call_ids:
                            issues.append(Issue(line_no, f"tool_call id 重复: {call_id}"))
                        pending_call_ids[call_id] = False
                    if not fn_name:
                        issues.append(Issue(line_no, f"messages[{idx}].tool_calls[{c_idx}] 缺少 function.name"))
                        continue
                    if has_tools and fn_name not in schemas:
                        issues.append(Issue(line_no, f"调用了未在 tools 中定义的函数 '{fn_name}'"))
                    check_arguments(fn.get("arguments"), schemas.get(fn_name, {}), line_no, fn_name, issues)

        elif role == "tool":
            call_id = msg.get("tool_call_id")
            if not call_id:
                issues.append(Issue(line_no, f"messages[{idx}] tool 消息缺少 tool_call_id"))
            elif call_id not in pending_call_ids:
                issues.append(Issue(line_no, f"messages[{idx}] tool_call_id={call_id!r} 没有对应的 assistant tool_call（或顺序错误）"))
            else:
                pending_call_ids[call_id] = True

    for call_id, consumed in pending_call_ids.items():
        if not consumed:
            issues.append(Issue(line_no, f"tool_call id={call_id!r} 没有对应的 tool 返回消息"))

    if has_tools and not saw_any_tool_call:
        issues.append(Issue(line_no, "提供了 tools 但本样本未发生任何工具调用（若为负样本，可忽略此提示）", "INFO"))


def validate_file(path: str) -> bool:
    issues: list = []
    n_records = 0
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                n_records += 1
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as e:
                    issues.append(Issue(line_no, f"非法 JSON: {e}"))
                    continue
                validate_record(record, line_no, issues)
    except FileNotFoundError:
        print(f"✗ 找不到文件: {path}")
        return False

    errors = [i for i in issues if i.level == "ERROR"]
    warns = [i for i in issues if i.level != "ERROR"]

    print(f"\n=== {path} （{n_records} 条样本）===")
    for i in errors + warns:
        print(i)
    if errors:
        print(f"✗ 失败：{len(errors)} 个错误，{len(warns)} 个提示")
        return False
    print(f"✓ 通过：{n_records} 条样本结构合法" + (f"（{len(warns)} 个提示）" if warns else ""))
    return True


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__)
        return 2
    all_ok = True
    for path in argv[1:]:
        if not validate_file(path):
            all_ok = False
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
