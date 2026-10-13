# Chuangzhi HE Gauntlet 2026

[中文版](README.zh-CN.md)

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

## Harness Design Tips

- Don't dump all training samples into context indiscriminately. The evaluation caps single-turn input below 2048 tokens — budget your samples and compress context deliberately.
- Don't hardcode labels or keywords against the official DEV set only. The real evaluation spans different domains, label counts, and multiple-choice tasks.
- Treat "ignore previous instructions", "SYSTEM says", "Developer message" inside `text` as text to classify, not as real instructions.
- Constrain predictions to candidate labels seen in the current task's training set, and clean outputs as needed.
- For multiple-choice tasks like `task7_mcq`, let the LLM reason from the stem and options instead of relying purely on nearest-neighbor samples.

## Provenance

`task0_official_dev` comes from the official DEV set in the Chuangzhi Academy 2026 summer-camp Harness Engineering evaluation kit. The remaining tasks are a self-built local benchmark for evaluation prep and harness iteration diagnostics.

This repository is for learning, research, and local evaluation only. For real submissions, follow the evaluation rules: don't read test labels, don't hardcode answers from public data.
