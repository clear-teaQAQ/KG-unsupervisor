#!/usr/bin/env python3
"""Audit simple GED cost baselines on SEABED-format graph pairs.

This script does not train or load GEDRanker. It checks whether cheap matching
or graph-size signals already agree with SEABED GED labels.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable


DATASETS = ["LUBM", "SWDF", "YAGO", "WIKIDATA"]


def read_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload.get("0", payload)


def graph_paths(dataset_root: Path) -> dict[str, Path]:
    paths = {}
    for split in ["train", "val", "test"]:
        for path in (dataset_root / split).glob("*.json"):
            if path.name in paths:
                raise ValueError(f"Duplicate graph filename under {dataset_root}: {path.name}")
            paths[path.name] = path
    return paths


def node_ids(graph: dict) -> list[str]:
    ids = []
    for idx, node in enumerate(graph.get("node_features", [])):
        value = node.get("id") if isinstance(node, dict) else None
        ids.append(str(value) if value is not None else str(idx))
    return ids


def node_embeddings(graph: dict) -> list[list[float]]:
    values = []
    for node in graph.get("node_features", []):
        emb = node.get("embedding", []) if isinstance(node, dict) else []
        if emb and isinstance(emb[0], list):
            emb = emb[0]
        values.append([float(x) for x in emb])
    return values


def edge_label(graph: dict, edge_idx: int) -> str:
    edge_features = graph.get("edge_features", [])
    if edge_idx < len(edge_features) and isinstance(edge_features[edge_idx], dict):
        value = edge_features[edge_idx].get("id")
        if value is not None:
            return str(value)
    kg = graph.get("KG", [])
    if edge_idx < len(kg) and len(kg[edge_idx]) >= 3:
        return str(kg[edge_idx][1])
    return ""


def labeled_adjacency(graph: dict) -> list[list[str]]:
    n = len(graph.get("node_features", []))
    adj = [["" for _ in range(n)] for _ in range(n)]
    for edge_idx, uv in enumerate(graph.get("edge_indices", [])):
        if len(uv) < 2:
            continue
        u, v = int(uv[0]), int(uv[1])
        if 0 <= u < n and 0 <= v < n:
            label = edge_label(graph, edge_idx)
            adj[u][v] = label
            adj[v][u] = label
    return adj


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0 or nb == 0:
        return -1.0
    return dot / (na * nb)


def complete_mapping(partial: dict[int, int], n1: int, n2: int) -> list[int]:
    used = set(partial.values())
    remaining = [idx for idx in range(n2) if idx not in used]
    mapping = []
    next_remaining = 0
    for i in range(n1):
        if i in partial and partial[i] not in used - {partial[i]}:
            mapping.append(partial[i])
        elif i in partial:
            mapping.append(partial[i])
        else:
            mapping.append(remaining[next_remaining])
            next_remaining += 1
    return mapping


def random_mapping(n1: int, n2: int, rng: random.Random) -> list[int]:
    cols = list(range(n2))
    rng.shuffle(cols)
    return cols[:n1]


def entity_mapping(g1: dict, g2: dict) -> list[int]:
    ids1 = node_ids(g1)
    ids2 = node_ids(g2)
    id_to_col = {}
    for col, entity in enumerate(ids2):
        id_to_col.setdefault(entity, col)
    partial = {}
    used = set()
    for row, entity in enumerate(ids1):
        col = id_to_col.get(entity)
        if col is not None and col not in used:
            partial[row] = col
            used.add(col)
    return complete_mapping(partial, len(ids1), len(ids2))


def feature_greedy_mapping(g1: dict, g2: dict) -> list[int]:
    emb1 = node_embeddings(g1)
    emb2 = node_embeddings(g2)
    scores = []
    for i, left in enumerate(emb1):
        for j, right in enumerate(emb2):
            scores.append((cosine(left, right), i, j))
    scores.sort(reverse=True)
    partial = {}
    used_rows = set()
    used_cols = set()
    for _, row, col in scores:
        if row not in used_rows and col not in used_cols:
            partial[row] = col
            used_rows.add(row)
            used_cols.add(col)
        if len(partial) == len(emb1):
            break
    return complete_mapping(partial, len(emb1), len(emb2))


def unit_cost(g1: dict, g2: dict, mapping: list[int]) -> float:
    n1 = len(g1.get("node_features", []))
    n2 = len(g2.get("node_features", []))
    adj1 = labeled_adjacency(g1)
    adj2 = labeled_adjacency(g2)
    unmatched_cols = [col for col in range(n2) if col not in mapping]
    permutation = mapping + unmatched_cols
    edge_cost = 0
    for i in range(n2):
        for j in range(i + 1, n2):
            left = adj1[i][j] if i < n1 and j < n1 else ""
            right = adj2[permutation[i]][permutation[j]]
            if left != right:
                edge_cost += 1
    return float((n2 - n1) + edge_cost)


def size_delta_cost(g1: dict, g2: dict) -> float:
    n1 = len(g1.get("node_features", []))
    n2 = len(g2.get("node_features", []))
    m1 = len(g1.get("edge_indices", []))
    m2 = len(g2.get("edge_indices", []))
    return float(abs(n2 - n1) + abs(m2 - m1))


def metric_rows(predictions: dict[str, list[tuple[float, float]]]) -> list[dict]:
    rows = []
    for name, values in predictions.items():
        if not values:
            continue
        errors = [abs(pred - gt) for pred, gt in values]
        rows.append(
            {
                "method": name,
                "pairs": len(values),
                "mae": round(sum(errors) / len(errors), 4),
                "acc": round(sum(int(round(pred) == gt) for pred, gt in values) / len(values), 4),
                "fea": round(sum(int(round(pred) >= gt) for pred, gt in values) / len(values), 4),
            }
        )
    return rows


def table_lines(rows: Iterable[dict]) -> list[str]:
    columns = ["method", "pairs", "mae", "acc", "fea"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return lines


def print_table(rows: Iterable[dict]) -> None:
    print("\n".join(table_lines(rows)))


def load_pairs(dataset_root: Path, split: str, ged_column: int) -> list[list]:
    pair_file = dataset_root / f"{split}_GEDINFO.json"
    pairs = json.loads(pair_file.read_text(encoding="utf-8"))["pairs_info"]
    ged_index = ged_column - 1
    for pair in pairs[:1]:
        if ged_index >= len(pair):
            raise ValueError(f"{pair_file} pair has {len(pair)} columns; cannot read --ged-column {ged_column}: {pair}")
    return pairs


def audit_dataset(
    dataset_root: Path,
    split: str,
    ged_column: int,
    max_pairs: int,
    seed: int,
    progress_interval: int,
) -> list[dict]:
    paths = graph_paths(dataset_root)
    pairs = load_pairs(dataset_root, split, ged_column)
    if max_pairs > 0:
        pairs = pairs[:max_pairs]

    rng = random.Random(seed)
    predictions = defaultdict(list)
    ged_index = ged_column - 1
    graph_cache = {}

    def get_graph(file_name: str) -> dict:
        graph = graph_cache.get(file_name)
        if graph is None:
            graph = read_json(paths[file_name])
            graph_cache[file_name] = graph
        return graph

    for pair_idx, pair in enumerate(pairs, start=1):
        if progress_interval > 0 and pair_idx % progress_interval == 0:
            print(f"[{dataset_root.name}] audited {pair_idx}/{len(pairs)} pairs; cached_graphs={len(graph_cache)}", file=sys.stderr)
        g1 = get_graph(pair[0])
        g2 = get_graph(pair[1])
        gt = float(pair[ged_index])
        if len(g1.get("node_features", [])) > len(g2.get("node_features", [])):
            g1, g2 = g2, g1

        n1 = len(g1.get("node_features", []))
        n2 = len(g2.get("node_features", []))
        predictions["size_delta"].append((size_delta_cost(g1, g2), gt))
        predictions["random_unit"].append((unit_cost(g1, g2, random_mapping(n1, n2, rng)), gt))
        predictions["entity_id_unit"].append((unit_cost(g1, g2, entity_mapping(g1, g2)), gt))
        predictions["feature_greedy_unit"].append((unit_cost(g1, g2, feature_greedy_mapping(g1, g2)), gt))

    rows = metric_rows(predictions)
    for row in rows:
        row["dataset"] = dataset_root.name
        row["split"] = split
        row["ged_column"] = ged_column
        row["checked_pairs"] = len(pairs)
    return rows


def report_lines(all_rows: list[dict]) -> list[str]:
    lines = []
    by_dataset = defaultdict(list)
    for row in all_rows:
        by_dataset[row["dataset"]].append(row)
    for dataset, rows in by_dataset.items():
        lines.append(f"\n# {dataset} split={rows[0]['split']} ged_column={rows[0]['ged_column']} checked_pairs={rows[0]['checked_pairs']}")
        lines.extend(table_lines(rows))
    return lines


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=Path("/data/projects/SEABED-main/data"))
    parser.add_argument("--dataset", choices=DATASETS + ["all"], default="all")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--ged-column", type=int, default=3)
    parser.add_argument("--max-pairs", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--progress-interval", type=int, default=1000, help="Print progress every N pairs. 0 disables progress.")
    parser.add_argument("--output", type=Path, default=None, help="Optional Markdown report path.")
    args = parser.parse_args()

    datasets = DATASETS if args.dataset == "all" else [args.dataset]
    all_rows = []
    for dataset in datasets:
        all_rows.extend(
            audit_dataset(
                args.data_root / dataset,
                args.split,
                args.ged_column,
                args.max_pairs,
                args.seed,
                args.progress_interval,
            )
        )

    lines = report_lines(all_rows)
    print("\n".join(lines))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text("\n".join(lines).lstrip() + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
