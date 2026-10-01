#!/usr/bin/env python3
"""Prepare Qwen3-32B M2 answer-space rewrites for visible-reasoning judgment."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "results"

INTERFACE_BY_DATASET = {
    "c3_core_mcq_base": "MCQ",
    "c3_core_bool_verify": "Bool",
    "c3_core_open_end": "Open-ended",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="External records.jsonl input.")
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/eir")
    return parser.parse_args()


def normalize_space(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    corpus_path = args.output_dir / "m2_corpus.jsonl"

    counts: Counter[str] = Counter()
    with args.input.open("r", encoding="utf-8") as source, corpus_path.open(
        "w", encoding="utf-8"
    ) as output:
        for line in source:
            if not line.strip() or "\x00" in line:
                continue
            row = json.loads(line)
            dataset = str(row.get("dataset", ""))
            if row.get("mode") != "mode2" or dataset not in INTERFACE_BY_DATASET:
                continue
            if str(row.get("error", "") or "").strip():
                continue

            interface = INTERFACE_BY_DATASET[dataset]
            record_id = f"C3-{counts['total'] + 1:05d}"
            question = normalize_space(row.get("question_text_used_for_similarity"))
            pre_answer_text = normalize_space(row.get("extracted_t"))
            prepared = {
                "record_id": record_id,
                "model": "Qwen3-32B",
                "mode": "mode2",
                "dataset": dataset,
                "interface": interface,
                "sample_id": str(row.get("id", "")),
                "question": question,
                "pre_answer_text": pre_answer_text,
                "t_is_empty": int(not pre_answer_text),
            }
            output.write(json.dumps(prepared, ensure_ascii=False) + "\n")
            counts["total"] += 1
            counts[f"{interface}_total"] += 1
            counts[f"{interface}_empty"] += int(not pre_answer_text)

    if counts["total"] != 8400:
        raise RuntimeError(f"expected 8,400 M2 outputs, found {counts['total']:,}")

    summary = {
        "source": str(args.input),
        "selection": "Qwen3-32B, paper M2, matched C3 answer-space rewrites",
        "records": counts["total"],
        "interfaces": {
            interface: {
                "records": counts[f"{interface}_total"],
                "empty_t": counts[f"{interface}_empty"],
            }
            for interface in ("MCQ", "Bool", "Open-ended")
        },
        "definition": "Empty T is retained and has no visible explicit reasoning.",
    }
    (args.output_dir / "corpus_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
