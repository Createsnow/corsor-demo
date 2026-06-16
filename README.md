# corsor-demo

## 大模型 Tool（函数调用）微调数据结构

本仓库整理了对大语言模型进行 **工具调用 / 函数调用（Function Calling）** 能力微调时的数据结构规范、示例数据与校验脚本。

- 📖 文档：[`docs/大模型tool微调数据结构.md`](docs/%E5%A4%A7%E6%A8%A1%E5%9E%8Btool%E5%BE%AE%E8%B0%83%E6%95%B0%E6%8D%AE%E7%BB%93%E6%9E%84.md)
- 📂 示例数据：`examples/`
  - `single_turn.jsonl` — 单轮单次调用
  - `multi_turn.jsonl` — 多轮 + 多次调用
  - `parallel_calls.jsonl` — 并行工具调用
  - `no_call.jsonl` — 负样本（不调用工具直接回答 / 追问）
- 🔧 校验脚本：`scripts/validate_tool_data.py`

### 快速开始

```bash
# 校验示例数据结构是否合法
python scripts/validate_tool_data.py examples/*.jsonl

# 校验自己的数据
python scripts/validate_tool_data.py path/to/your_data.jsonl
```
