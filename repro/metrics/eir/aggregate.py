#!/usr/bin/env python3
"""Aggregate answer-space GPT-5.5 labels into all-output EIR values."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


RESULTS_DIR = Path(__file__).resolve().parents[3] / "outputs/eir"
INTERFACE_ORDER = ("MCQ", "Bool", "Open-ended")
CATEGORIES = (
    "empty",
    "generic_or_off_topic",
    "paraphrase_only",
    "relevant_noninferential",
    "explicit_reasoning",
    "unclear",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=RESULTS_DIR / "m2_corpus.jsonl")
    parser.add_argument("--labels", type=Path, default=RESULTS_DIR / "gpt55_labels.jsonl")
    parser.add_argument(
        "--fallback-labels",
        type=Path,
        default=RESULTS_DIR / "manual_content_filter_labels.jsonl",
    )
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    labels: dict[str, str] = {}
    with args.labels.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            labels[str(row["record_id"])] = str(row["category"])
    primary_label_count = len(labels)

    fallback_count = 0
    if args.fallback_labels.is_file():
        with args.fallback_labels.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                record_id = str(row["record_id"])
                if record_id in labels:
                    continue
                labels[record_id] = str(row["category"])
                fallback_count += 1

    counts: dict[str, Counter[str]] = defaultdict(Counter)
    missing: list[str] = []
    empty_mismatches: list[str] = []
    with args.corpus.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = row["record_id"]
            category = labels.get(record_id)
            if category is None:
                missing.append(record_id)
                continue
            if category not in CATEGORIES:
                raise RuntimeError(f"invalid category for {record_id}: {category}")
            if bool(row["t_is_empty"]) != (category == "empty"):
                empty_mismatches.append(record_id)
            counts[row["interface"]]["n"] += 1
            counts[row["interface"]][category] += 1

    if missing or empty_mismatches:
        raise RuntimeError(
            f"missing={len(missing)}, empty_mismatches={len(empty_mismatches)}"
        )

    rows: list[dict[str, object]] = []
    for interface in INTERFACE_ORDER:
        group = counts[interface]
        n = group["n"]
        row: dict[str, object] = {
            "interface": interface,
            "n": n,
            "explicit_reasoning_n": group["explicit_reasoning"],
            "eir_percent_all": 100.0 * group["explicit_reasoning"] / n,
        }
        for category in CATEGORIES:
            row[f"{category}_n"] = group[category]
            row[f"{category}_percent_all"] = 100.0 * group[category] / n
        rows.append(row)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "interface_eir.csv"
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    audit = {
        "corpus_records": sum(row["n"] for row in rows),
        "unique_labels": len(labels),
        "gpt55_labels": primary_label_count,
        "manual_content_filter_fallback_labels": fallback_count,
        "missing_labels": len(missing),
        "empty_label_mismatches": len(empty_mismatches),
        "eir_denominator": "all outputs within each interface",
        "rows": rows,
    }
    (args.output_dir / "audit.json").write_text(
        json.dumps(audit, indent=2), encoding="utf-8"
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
