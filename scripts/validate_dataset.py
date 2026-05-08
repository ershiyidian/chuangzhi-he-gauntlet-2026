#!/usr/bin/env python3
"""Validate Chuangzhi HE Gauntlet JSONL task files."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TASK_DIRS = [
    "task0_official_dev",
    "task1_finance",
    "task2_ecommerce",
    "task3_tech_support",
    "task4_medical_triage",
    "task5_news_topic",
    "task6_tone",
    "task7_mcq",
    "task8_prompt_injection",
]


def load_jsonl(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc

            missing = {"text", "label"} - row.keys()
            if missing:
                raise ValueError(f"{path}:{line_no}: missing fields {sorted(missing)}")
            if not isinstance(row["text"], str) or not isinstance(row["label"], str):
                raise ValueError(f"{path}:{line_no}: text and label must be strings")

            rows.append(row)
    return rows


def main() -> None:
    total_train = 0
    total_test = 0

    for task_dir in TASK_DIRS:
        train_path = ROOT / task_dir / "train.jsonl"
        test_path = ROOT / task_dir / "test.jsonl"
        if not train_path.exists() or not test_path.exists():
            raise FileNotFoundError(f"{task_dir}: expected train.jsonl and test.jsonl")

        train = load_jsonl(train_path)
        test = load_jsonl(test_path)
        train_labels = {row["label"] for row in train}
        test_labels = {row["label"] for row in test}
        unseen = sorted(test_labels - train_labels)
        if unseen:
            raise ValueError(f"{task_dir}: test labels missing from train: {unseen}")

        total_train += len(train)
        total_test += len(test)
        print(
            f"{task_dir}: train={len(train):3d} "
            f"test={len(test):3d} labels={len(train_labels):2d}"
        )

    print(f"TOTAL: train={total_train} test={total_test} all={total_train + total_test}")


if __name__ == "__main__":
    main()
