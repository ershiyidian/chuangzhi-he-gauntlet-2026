# Chuangzhi HE Gauntlet 2026

[English](README.md)

创智学院 2026 夏令营 Harness Engineering 考核数据集与自建压力测试集。

本仓库面向限制输入窗口的 LLM 文本分类 Harness 评测：训练阶段通过 `update(text, label)` 流式接收带标签样本，预测阶段通过 `predict(text)` 返回精确标签；模型权重不可更新，所有学习都必须发生在 Harness 的外部状态、样本检索、上下文压缩、标签约束和输出解析中。

## 数据定位

仓库包含两部分：

- `task0_official_dev`: 考核包官方提供的 DEV 集，客服意图分类，77 个标签。
- `task1_finance` 至 `task8_prompt_injection`: 自建本地 Benchmark，用于模拟正式评测中的多领域 OOD、自然语言选择题和 Prompt Injection 压力场景。

这套数据用于训练 Harness Engineering 的关键能力：少样本标签语义归纳、跨领域迁移、2048 token 预算管理、候选标签约束、选择题推理触发和用户文本指令隔离。

## 数据格式

所有样本均为 JSONL，每行一条 JSON：

```json
{"text": "I no longer have my phone.", "label": "lost_or_stolen_phone"}
```

字段说明：

- `text`: 待分类的自然语言输入。它可能包含噪声、口语表达、伪系统指令或多行选择题。
- `label`: 精确标签字符串。预测输出必须与该字段完全一致。

每个任务均保证测试集中出现的标签已在对应训练集中出现。公开数据的测试集包含 `label` 方便本地评测；正式考核中测试标签不可见。

## 文件清单

| 任务目录 | 内容 | Train | Test | 标签数 |
| --- | --- | ---: | ---: | ---: |
| `task0_official_dev` | 官方 DEV 客服意图分类 | 231 | 539 | 77 |
| `task1_finance` | 金融客服意图分类 | 90 | 250 | 25 |
| `task2_ecommerce` | 电商客服意图分类 | 90 | 250 | 25 |
| `task3_tech_support` | 技术支持意图分类 | 89 | 250 | 25 |
| `task4_medical_triage` | 医疗咨询与分诊分类 | 90 | 250 | 25 |
| `task5_news_topic` | 新闻主题分类 | 90 | 250 | 25 |
| `task6_tone` | 情绪、语气与态度分类 | 90 | 250 | 25 |
| `task7_mcq` | 自然语言选择题，标签为 `A/B/C/D` | 32 | 169 | 4 |
| `task8_prompt_injection` | Prompt Injection 专项鲁棒性测试 | 49 | 150 | 15 |

合计：`851` 条训练样本，`2358` 条测试样本，`3209` 条样本。

## 目录结构

```text
.
├── task0_official_dev/
│   ├── train.jsonl
│   └── test.jsonl
├── task1_finance/
│   ├── train.jsonl
│   └── test.jsonl
├── ...
├── task8_prompt_injection/
│   ├── train.jsonl
│   └── test.jsonl
├── docs/
│   └── benchmark.md
└── scripts/
    └── validate_dataset.py
```

`docs/benchmark.md` 是更长的 Benchmark 设计说明，包含 8 个自建任务的标签体系、样本构造思路和诊断价值。

## 快速校验

```bash
python scripts/validate_dataset.py
```

校验内容包括：

- 所有 `train.jsonl` / `test.jsonl` 可按 UTF-8 逐行解析为 JSON。
- 每条样本均包含且仅依赖 `text`、`label` 字段。
- 每个任务的测试标签集合是训练标签集合的子集。
- 输出每个任务的样本数和标签数。

## 推荐使用方式

一个基础评测循环通常如下：

```python
import json
from pathlib import Path

def load_jsonl(path):
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

harness = MyHarness(...)

for sample in load_jsonl("task1_finance/train.jsonl"):
    harness.update(sample["text"], sample["label"])

correct = 0
total = 0
for sample in load_jsonl("task1_finance/test.jsonl"):
    pred = harness.predict(sample["text"])
    correct += pred == sample["label"]
    total += 1

print(correct / total)
```

按任务分别统计准确率（而非只看总平均）：不同任务暴露的问题不同，金融高而 OOD 低通常意味着领域过拟合，选择题低通常意味着缺少推理分支，注入任务低通常意味着系统提示与用户文本隔离不足。

## Harness 设计提示

- 不要把全部训练样本无差别塞入上下文。考核限制单轮输入 token 小于 2048，需要主动做样本选择与上下文预算控制。
- 不要只针对官方 DEV 集写死标签或关键词。正式评测包含不同领域、不同标签数量和选择题任务。
- 对 `text` 中的 “ignore previous instructions”、“SYSTEM says”、“Developer message” 等内容应视为待分类文本，而不是真实指令。
- 预测结果应约束在当前任务训练集中出现过的候选标签内，并做必要的输出清洗。
- 对 `task7_mcq` 这类选择题，应允许 LLM 根据题干和选项推理，而不是完全依赖最近邻样本。

## 来源与说明

`task0_official_dev` 来自创智学院 2026 夏令营 Harness Engineering 考核包中的官方 DEV 集。其余任务为本地自建 Benchmark，用于考核准备和 Harness 迭代诊断。

本仓库仅用于学习、研究与本地评测。正式提交请遵守考核规则：不读取测试集标签，不硬编码公开数据答案。
