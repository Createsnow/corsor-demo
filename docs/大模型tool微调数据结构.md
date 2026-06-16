# 大模型 Tool（工具/函数调用）微调数据结构

本文档系统说明在对大语言模型（LLM）进行 **工具调用 / 函数调用（Function Calling / Tool Use）** 能力微调时，训练数据应当如何组织。涵盖核心字段定义、单轮 / 多轮 / 并行调用样本、主流框架的格式差异，以及数据质量与校验要点。

> 配套文件：
> - `examples/` 目录下提供了可直接使用的 JSONL 示例数据
> - `scripts/validate_tool_data.py` 提供了数据结构校验脚本

---

## 1. 为什么需要专门的数据结构

普通对话微调只有 `user` 与 `assistant` 两类消息。而工具调用引入了两个新概念：

1. **工具定义（tools / schema）**：模型可以调用哪些函数，每个函数的名称、用途、参数（通常用 JSON Schema 描述）。
2. **工具调用与返回**：助手不直接回答用户，而是先"决定调用某个函数并给出参数"，外部执行后把结果回传给模型，模型再基于结果生成最终回答。

因此训练样本必须能够表达一条完整的链路：

```
系统提示 + 工具定义
  → 用户提问
    → 助手发起工具调用（tool_calls）
      → 工具执行结果（role=tool）
        → 助手给出最终自然语言回答
```

---

## 2. 核心数据结构（推荐基准格式）

我们采用与 OpenAI Chat Completions 兼容的结构作为基准格式，它被绝大多数训练框架（LLaMA-Factory、ms-swift、axolotl、trl 等）支持或可无损转换。

一条训练样本是一个 JSON 对象，主要包含两个顶层字段：

| 字段       | 类型   | 说明                                             |
| ---------- | ------ | ------------------------------------------------ |
| `messages` | array  | 多轮对话消息列表（必填）                         |
| `tools`    | array  | 本轮对话可用的工具定义列表（可选，但工具样本必填）|

### 2.1 `tools`：工具定义

每个工具是一个 `type=function` 的对象，`function.parameters` 使用标准 **JSON Schema** 描述入参。

```json
{
  "type": "function",
  "function": {
    "name": "get_current_weather",
    "description": "查询指定城市的实时天气。当用户询问天气、温度、是否下雨等信息时调用。",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称，例如：北京、上海"
        },
        "unit": {
          "type": "string",
          "enum": ["celsius", "fahrenheit"],
          "description": "温度单位，默认摄氏度"
        }
      },
      "required": ["location"]
    }
  }
}
```

**字段要点：**

- `name`：函数名，建议 `snake_case`，全局唯一，模型据此输出调用目标。
- `description`：**至关重要**。它是模型判断"何时调用 / 不调用"的主要依据，应写清用途和触发条件。
- `parameters`：JSON Schema。常用关键字：`type`、`properties`、`required`、`enum`、`items`（数组元素类型）、`description`。
- 没有参数的函数也要给出 `"parameters": {"type": "object", "properties": {}}`。

### 2.2 `messages`：消息序列

消息按对话顺序排列，每条包含 `role` 与对应内容。工具场景下共有 4 种角色：

| role        | 含义           | 关键字段                                  |
| ----------- | -------------- | ----------------------------------------- |
| `system`    | 系统/人设指令  | `content`                                 |
| `user`      | 用户输入       | `content`                                 |
| `assistant` | 模型输出       | `content` 和/或 `tool_calls`              |
| `tool`      | 工具执行结果   | `content`、`tool_call_id`                 |

#### assistant 发起工具调用

当助手决定调用工具时，`content` 通常为 `null`（或空），并填充 `tool_calls` 数组：

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_current_weather",
        "arguments": "{\"location\": \"北京\", \"unit\": \"celsius\"}"
      }
    }
  ]
}
```

> ⚠️ 注意：`function.arguments` 是一个 **JSON 字符串**（被转义的字符串），而不是 JSON 对象。这是为了与推理时的流式生成对齐——模型逐 token 生成的就是这段字符串。

#### tool 返回结果

每个工具结果必须通过 `tool_call_id` 与上面的某个 `tool_calls[].id` 一一对应：

```json
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "{\"temperature\": 26, \"unit\": \"celsius\", \"condition\": \"晴\"}"
}
```

#### assistant 最终回答

```json
{
  "role": "assistant",
  "content": "北京现在 26°C，天气晴朗，适合外出。"
}
```

---

## 3. 三类典型样本

### 3.1 单轮单次调用

最常见：用户提问 → 调用一个工具 → 返回 → 回答。见 `examples/single_turn.jsonl`。

### 3.2 多轮 + 多次调用

对话跨多轮，且可能在不同轮次多次调用工具（包括同一工具的连续调用）。见 `examples/multi_turn.jsonl`。

### 3.3 并行工具调用（Parallel Tool Calls）

单条 `assistant` 消息中一次性发起 **多个** `tool_calls`，随后跟随 **多条** `tool` 消息分别回传。见 `examples/parallel_calls.jsonl`。

```json
{
  "role": "assistant",
  "content": null,
  "tool_calls": [
    {"id": "call_1", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"北京\"}"}},
    {"id": "call_2", "type": "function", "function": {"name": "get_current_weather", "arguments": "{\"location\": \"上海\"}"}}
  ]
}
```

### 3.4 负样本：不应调用工具

为防止模型"逢问必调用"（over-calling），数据集中必须包含 **即使提供了工具、但正确行为是直接回答** 的样本。例如用户只是打招呼或问常识。见 `examples/no_call.jsonl`。

---

## 4. 主流框架格式对照

不同训练栈对同一逻辑结构有不同写法，下面给出映射关系。

### 4.1 OpenAI / 通用对话格式（本文档基准）

如第 2 节所示，`{"messages": [...], "tools": [...]}`。LLaMA-Factory 的 `sharegpt`/`openai` 风格、ms-swift 的 messages 格式均可直接或近似使用。

### 4.2 ChatML（Qwen 系列）渲染后形态

训练框架会把上面的结构渲染成带特殊标记的纯文本。Qwen 的工具调用渲染大致如下：

```text
<|im_start|>system
你是一个有用的助手。

# Tools
你可以调用以下函数...
<tools>
{"type":"function","function":{"name":"get_current_weather", ...}}
</tools><|im_end|>
<|im_start|>user
北京天气怎么样？<|im_end|>
<|im_start|>assistant
<tool_call>
{"name": "get_current_weather", "arguments": {"location": "北京"}}
</tool_call><|im_end|>
<|im_start|>user
<tool_response>
{"temperature": 26, "condition": "晴"}
</tool_response><|im_end|>
<|im_start|>assistant
北京现在 26°C，天气晴朗。<|im_end|>
```

要点：工具结果在 ChatML 里常以 `user` 轮内的 `<tool_response>` 承载，调用以 `<tool_call>` 承载。**一般无需手写这层文本**，交给框架的 chat template 渲染即可。

### 4.3 Llama 3.1 格式

Llama 3.1 使用 `<|start_header_id|>...<|end_header_id|>` 标记角色，工具结果使用 `ipython` 角色，调用以 JSON 形式输出。同样推荐用官方 chat template 渲染，训练数据保持基准 `messages` 结构。

### 4.4 Hermes / Glaive 等开源数据集

- **Glaive Function Calling**：字段名可能为 `system` / `chat` 文本流，需转换为 `messages`。
- **Hermes**：使用 `<tool_call>` / `<tool_response>` 标签的 ChatML 变体。

**结论**：以"逻辑结构（基准 `messages` + `tools`）"为单一数据源（Source of Truth），再按目标框架渲染，是最稳健的工程实践。

---

## 5. 训练时的标签与损失（Loss Mask）

工具微调的关键不只是格式，还有 **哪些 token 计算损失**：

- ✅ 需要计算 loss：`assistant` 的 `content` 与 `tool_calls`（即模型应当学会"何时调用、调用什么、参数是什么、最终如何回答"）。
- ❌ 不计算 loss：`system`、`user`、`tool` 消息（这些是输入/外部观测，模型不应"生成"它们）。

大多数框架通过 chat template + `train_on_inputs=false` / 自动 mask 处理。务必确认：**工具返回结果（role=tool）被 mask 掉**，否则模型会去"幻想"工具输出。

---

## 6. 数据质量清单（Checklist）

- [ ] 每个 `tool_calls[].id` 都有且仅有一个对应的 `tool` 消息（`tool_call_id` 匹配）。
- [ ] `tool` 消息一定出现在对应 `assistant(tool_calls)` 之后，最终回答之前。
- [ ] `function.arguments` 是合法 JSON 字符串，且其中字段满足工具的 JSON Schema（类型、`required`、`enum`）。
- [ ] 实际被调用的 `function.name` 出现在 `tools` 定义中。
- [ ] 调用参数能从用户输入中合理推断（避免"凭空捏造"参数的脏数据）。
- [ ] 含有负样本（不调用工具直接回答）防止过度调用。
- [ ] 覆盖多样性：不同工具、缺参/追问、并行调用、调用失败/报错处理。
- [ ] 一致性：`system` 中工具说明与 `tools` 字段不矛盾。
- [ ] 数据可被框架的 chat template 正确渲染（先小批量试渲染检查）。

---

## 7. 快速开始

```bash
# 校验示例数据结构是否合法
python scripts/validate_tool_data.py examples/single_turn.jsonl
python scripts/validate_tool_data.py examples/multi_turn.jsonl
python scripts/validate_tool_data.py examples/parallel_calls.jsonl
python scripts/validate_tool_data.py examples/no_call.jsonl

# 校验自己的数据
python scripts/validate_tool_data.py path/to/your_data.jsonl
```

---

## 8. 参考

- OpenAI Function Calling / Chat Completions API 文档
- Qwen-Agent / Qwen Chat Template（ChatML + `<tool_call>`）
- Llama 3.1 Tool Use 文档
- LLaMA-Factory、ms-swift、axolotl、trl 微调框架文档
- JSON Schema 规范（参数定义）
