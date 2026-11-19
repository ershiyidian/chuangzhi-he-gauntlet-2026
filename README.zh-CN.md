# Chuangzhi HE Gauntlet 2026

[English](README.md) · 简体中文

![Tasks](https://img.shields.io/badge/Tasks-9%20Suites-0969da.svg)
![Samples](https://img.shields.io/badge/Samples-3%2C209%20JSONL-0f766e.svg)
![Budget](https://img.shields.io/badge/Input%20Budget-%3C2048%20Tokens-475569.svg)

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

## Harness 设计要点

- **上下文预算管理**：单轮输入受限于 2048 token 窗口，建议结合代表性样本检索与紧凑标签摘要控制上下文长度。
- **跨领域动态归纳**：从 `update(text, label)` 流式样本中动态建立标签语义索引，适配不同领域、标签规模与题型分布。
- **指令与数据隔离**：将 `text` 严格封装为待分类数据字段，使其中包含的 `"ignore previous instructions"`、`"SYSTEM says"` 等对抗文本不触发指令越权。
- **候选标签集约束**：将输出空间限制在当前任务训练集已观测到的标签集合内，并配合确定性输出解析。
- **选择题显式推理**：针对 `task7_mcq` 等题干加选项任务启用选项对比推理分支，弥补纯近邻检索的局限。

## 来源与说明

`task0_official_dev` 来自创智学院 2026 夏令营 Harness Engineering 考核包官方 DEV 集；`task1_finance` 至 `task8_prompt_injection` 为面向跨领域泛化与注入鲁棒性诊断的自建评测集。本地 `test.jsonl` 保留 `label` 字段用于离线准确率统计，评测循环调用 `predict(text)` 时仅传入 `sample["text"]`。
