# Chuangzhi HE Gauntlet 2026

English · [简体中文](README.zh-CN.md)

![Tasks](https://img.shields.io/badge/Tasks-9%20Suites-0969da.svg)
![Samples](https://img.shields.io/badge/Samples-3%2C209%20JSONL-0f766e.svg)
![Budget](https://img.shields.io/badge/Input%20Budget-%3C2048%20Tokens-475569.svg)

The Chuangzhi Academy 2026 summer-camp Harness Engineering evaluation dataset, plus a self-built stress-test suite.

This repository targets LLM text-classification harness evaluation under a constrained input window: during training, `update(text, label)` streams labeled samples; during prediction, `predict(text)` returns the exact label. Model weights are frozen — all learning must happen in the harness's external state, sample retrieval, context compression, label constraints, and output parsing.

## Data

Two parts:

- `task0_official_dev`: the official DEV set from the evaluation kit — customer-intent classification, 77 labels.
- `task1_finance` through `task8_prompt_injection`: a self-built local benchmark simulating multi-domain OOD, natural-language multiple-choice, and prompt-injection stress scenarios from the real evaluation.

This data trains the core Harness Engineering skills: few-shot label-semantics induction, cross-domain transfer, 2048-token budget management, candidate-label constraints, multiple-choice reasoning triggers, and isolating user-text instructions.

## Format

All samples are JSONL, one JSON object per line:

```json
{"text": "I no longer have my phone.", "label": "lost_or_stolen_phone"}
```

Fields:

- `text`: the natural-language input to classify. May contain noise, colloquialisms, fake system instructions, or multi-line multiple-choice questions.
- `label`: the exact label string. Predictions must match this field exactly.

Every task guarantees that labels appearing in the test set also appear in the corresponding training set. The public test sets include `label` for convenient local evaluation; in the real evaluation, test labels are hidden.

## File Manifest

| Task dir | Content | Train | Test | Labels |
| --- | --- | ---: | ---: | ---: |
| `task0_official_dev` | Official DEV: customer-intent classification | 231 | 539 | 77 |
| `task1_finance` | Finance customer-intent classification | 90 | 250 | 25 |
| `task2_ecommerce` | E-commerce customer-intent classification | 90 | 250 | 25 |
| `task3_tech_support` | Tech-support classification | 89 | 250 | 25 |
| `task4_medical_triage` | Medical consultation & triage classification | 90 | 250 | 25 |
| `task5_news_topic` | News topic classification | 90 | 250 | 25 |
| `task6_tone` | Emotion, tone & attitude classification | 90 | 250 | 25 |
| `task7_mcq` | Natural-language multiple choice, labels `A/B/C/D` | 32 | 169 | 4 |
| `task8_prompt_injection` | Prompt-injection robustness tests | 49 | 150 | 15 |

Total: `851` training samples, `2358` test samples, `3209` samples.

## Directory Layout

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

`docs/benchmark.md` is the longer benchmark design note, covering the label taxonomies, sample-construction rationale, and diagnostic value of the 8 self-built tasks.

## Quick Validation

```bash
python scripts/validate_dataset.py
```

Checks:

- Every `train.jsonl` / `test.jsonl` parses as UTF-8 JSON, one object per line.
- Every sample carries exactly the `text` and `label` fields.
- Each task's test label set is a subset of its training label set.
- Prints per-task sample and label counts.

## Recommended Usage

A basic evaluation loop:

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

Score accuracy per task rather than looking at the overall average alone: different tasks expose different problems — high finance accuracy with low OOD usually means domain overfitting, low multiple-choice usually means a missing reasoning branch, low injection-task scores usually mean insufficient isolation between system prompts and user text.

## Harness Design Principles

- **Token-Budgeted Retrieval**: Single-turn prompts operate within a 2,048-token ceiling—combine selective few-shot retrieval with compact label summaries.
- **Task-Agnostic Label Induction**: Induce label semantics dynamically from streamed `update(text, label)` examples across varying label cardinalities and domains.
- **Instruction-Data Isolation**: Encapsulate `text` strictly as untrusted input data so embedded strings such as `"ignore previous instructions"` or `"SYSTEM says"` cannot override classification directives.
- **Constrained Output Decoding**: Restrict final predictions to the active task's observed training label set with deterministic output normalization.
- **Adaptive MCQ Reasoning**: Route structured stem-and-option inputs (such as `task7_mcq`) through explicit option-comparison reasoning alongside retrieved exemplars.

## Provenance

`task0_official_dev` originates from the official DEV split of the Chuangzhi Academy 2026 Harness Engineering evaluation kit; `task1_finance` through `task8_prompt_injection` form an extended diagnostic benchmark for out-of-distribution generalization and prompt-injection stress testing. Public `test.jsonl` splits retain ground-truth `label` fields for offline scoring—evaluation loops should pass only `sample["text"]` into `predict(text)`.
