#!/usr/bin/env python3
"""Collect GEDRanker-SEABED result JSON files into a compact table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


METRICS = ["mse", "mae", "acc", "fea", "rho", "tau", "pk1", "pk5", "pk10", "pk15", "pk20", "time", "run_time"]


def load_result(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    config = payload.get("config", {})
    row = {
        "file": path.name,
        "dataset": config.get("dataset", infer_dataset(path.name)),
        "ged_column": config.get("ged_column", infer_token(path.name, "gedcol")),
        "cost_mode": config.get("cost_mode", infer_cost_mode(path.name)),
        "model_name": config.get("model_name", ""),
        "test_k": config.get("test_k", ""),
        "max_train_pairs": config.get("max_train_pairs", ""),
        "max_test_pairs": config.get("max_test_pairs", ""),
    }
    for metric in METRICS:
        row[metric] = payload.get(metric, "")
    return row


def infer_dataset(file_name: str) -> str:
    for dataset in ["LUBM", "SWDF", "YAGO", "WIKIDATA", "IMDB"]:
        if f"_{dataset}_" in file_name:
            return dataset
    return ""


def infer_token(file_name: str, prefix: str) -> str:
    for token in file_name.replace(".json", "").split("_"):
        if token.startswith(prefix):
            return token[len(prefix) :]
    return ""


def infer_cost_mode(file_name: str) -> str:
    for mode in ["containment", "unit"]:
        if f"_{mode}_" in file_name or file_name.endswith(f"_{mode}.json"):
            return mode
    return ""


def markdown_table(rows: list[dict]) -> str:
    columns = ["dataset", "ged_column", "cost_mode", "mae", "acc", "fea", "rho", "tau", "pk10", "pk20", "test_k", "max_test_pairs", "file"]
    out = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, default=Path("result"))
    parser.add_argument("--pattern", default="result_SEABED_*_*.json")
    parser.add_argument("--output", type=Path, default=None, help="Optional CSV output path.")
    args = parser.parse_args()

    files = sorted(args.result_dir.glob(args.pattern))
    rows = [load_result(path) for path in files]
    print(markdown_table(rows))

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [])
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    main()
