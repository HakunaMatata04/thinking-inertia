#!/usr/bin/env python3
"""Aggregate QRel and EIR by the official MMLU subject areas."""

from __future__ import annotations

import csv
import json
import argparse
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]

MODELS = ("Qwen3-4B", "Qwen3-32B", "Qwen3-235B", "DeepSeek-V4")
MODES = ("mode1", "mode2", "mode5", "mode6")

MMLU_AREAS = {
    "Humanities": {
        "formal_logic",
        "high_school_european_history",
        "high_school_us_history",
        "high_school_world_history",
        "international_law",
        "jurisprudence",
        "logical_fallacies",
        "moral_disputes",
        "moral_scenarios",
        "philosophy",
        "prehistory",
        "professional_law",
        "world_religions",
    },
    "Social Sciences": {
        "econometrics",
        "high_school_geography",
        "high_school_government_and_politics",
        "high_school_macroeconomics",
        "high_school_microeconomics",
        "high_school_psychology",
        "human_sexuality",
        "professional_psychology",
        "public_relations",
        "security_studies",
        "sociology",
        "us_foreign_policy",
    },
    "STEM": {
        "abstract_algebra",
        "anatomy",
        "astronomy",
        "college_biology",
        "college_chemistry",
        "college_computer_science",
        "college_mathematics",
        "college_physics",
        "computer_security",
        "conceptual_physics",
        "electrical_engineering",
        "elementary_mathematics",
        "high_school_biology",
        "high_school_chemistry",
        "high_school_computer_science",
        "high_school_mathematics",
        "high_school_physics",
        "high_school_statistics",
        "machine_learning",
    },
    "Other": {
        "business_ethics",
        "clinical_knowledge",
        "college_medicine",
        "global_facts",
        "human_aging",
        "management",
        "marketing",
        "medical_genetics",
        "miscellaneous",
        "nutrition",
        "professional_accounting",
        "professional_medicine",
        "virology",
    },
}


def load_subjects(mmlu_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    subject_by_id: dict[str, str] = {}
    with mmlu_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            subject_by_id[str(row["id"])] = str(row["meta"]["subject"])

    area_by_subject = {
        subject: area
        for area, subjects in MMLU_AREAS.items()
        for subject in subjects
    }
    observed = set(subject_by_id.values())
    if observed != set(area_by_subject):
        missing = sorted(observed - set(area_by_subject))
        extra = sorted(set(area_by_subject) - observed)
        raise RuntimeError(f"MMLU area mapping mismatch: missing={missing}, extra={extra}")
    return subject_by_id, area_by_subject


def load_explicit_labels(labels_path: Path) -> dict[str, bool]:
    labels: dict[str, bool] = {}
    with labels_path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row["record_id"])
            if record_id in labels:
                raise RuntimeError(f"duplicate judge label: {record_id}")
            labels[record_id] = row["category"] == "explicit_reasoning"
    return labels


def update(bucket: dict[str, float], qrel: float, explicit: bool) -> None:
    bucket["n"] += 1
    bucket["qrel_sum"] += qrel
    bucket["explicit_n"] += int(explicit)


def finalize(grouped: dict[tuple[str, ...], dict[str, float]], fields: list[str]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for key, values in sorted(grouped.items()):
        n = int(values["n"])
        rows.append(
            {
                **dict(zip(fields, key)),
                "n": n,
                "qrel": values["qrel_sum"] / n,
                "explicit_n": int(values["explicit_n"]),
                "eir": 100.0 * values["explicit_n"] / n,
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mmlu", type=Path, default=REPO_ROOT / "data/processed/mmlu.jsonl")
    parser.add_argument("--judge-labels", type=Path, required=True)
    parser.add_argument("--embedding-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=REPO_ROOT / "outputs/domain")
    args = parser.parse_args()
    subject_by_id, area_by_subject = load_subjects(args.mmlu)
    explicit_by_id = load_explicit_labels(args.judge_labels)

    area_metrics: dict[tuple[str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "qrel_sum": 0.0, "explicit_n": 0}
    )
    subject_metrics: dict[tuple[str, str, str, str], dict[str, float]] = defaultdict(
        lambda: {"n": 0, "qrel_sum": 0.0, "explicit_n": 0}
    )
    selected = 0
    missing_labels = 0

    for path in sorted(args.embedding_dir.glob("item_scores_shard_*.csv")):
        with path.open(encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                if row["dataset"] != "mmlu" or row["model"] not in MODELS or row["mode"] not in MODES:
                    continue
                record_id = row["record_id"]
                explicit = explicit_by_id.get(record_id)
                if explicit is None:
                    missing_labels += 1
                    continue
                subject = subject_by_id[row["sample_id"]]
                area = area_by_subject[subject]
                qrel = float(row["qwen3_embedding_4b_reasoning"])
                update(area_metrics[(row["model"], row["mode"], area)], qrel, explicit)
                update(subject_metrics[(row["model"], row["mode"], area, subject)], qrel, explicit)
                selected += 1

    if missing_labels:
        raise RuntimeError(f"missing {missing_labels} judge labels")

    area_rows = finalize(area_metrics, ["model", "mode", "area"])
    subject_rows = finalize(subject_metrics, ["model", "mode", "area", "subject"])

    full_area_sizes = {
        "Humanities": 4705,
        "Social Sciences": 3077,
        "STEM": 3153,
        "Other": 3107,
    }
    expected_by_model_area: dict[tuple[str, str], int] = {}
    for row in area_rows:
        cell = (str(row["model"]), str(row["area"]))
        observed = int(row["n"])
        expected = expected_by_model_area.setdefault(cell, observed)
        if observed != expected:
            raise RuntimeError(f"area size changes across modes: {row}, expected n={expected}")
        if row["model"] in {"Qwen3-4B", "Qwen3-32B"} and observed != full_area_sizes[str(row["area"])]:
            raise RuntimeError(f"incomplete local-model MMLU cell: {row}")

    totals_by_model_mode: dict[tuple[str, str], int] = defaultdict(int)
    for row in area_rows:
        totals_by_model_mode[(str(row["model"]), str(row["mode"]))] += int(row["n"])
    expected_totals = {
        "Qwen3-4B": 14042,
        "Qwen3-32B": 14042,
        "Qwen3-235B": 1000,
        "DeepSeek-V4": 1000,
    }
    for (model, mode), observed in totals_by_model_mode.items():
        if observed != expected_totals[model]:
            raise RuntimeError(f"unexpected total for {model} {mode}: {observed}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "area_mode_metrics.csv", area_rows)
    write_csv(args.output_dir / "subject_mode_metrics.csv", subject_rows)
    audit = {
        "selected_records": selected,
        "models": MODELS,
        "modes": MODES,
        "full_mmlu_area_sizes": full_area_sizes,
        "actual_area_sizes_by_model": {
            f"{model}/{area}": n
            for (model, area), n in sorted(expected_by_model_area.items())
        },
        "qrel_field": "qwen3_embedding_4b_reasoning",
        "eir_definition": "100 * explicit_reasoning / all outputs",
        "judge_labels": str(args.judge_labels),
    }
    (args.output_dir / "audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
