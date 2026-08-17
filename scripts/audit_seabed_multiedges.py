#!/usr/bin/env python3
"""Read-only audit of parallel relation edges in SEABED datasets.

The official V16 graph view treats endpoint pairs as undirected and keeps the
last raw edge for each pair. This script measures exactly what that projection
collapses. It never changes graph JSON, GED labels, or model checkpoints.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Iterable


DEFAULT_DATA_ROOT = Path("/data/projects/SEABED-main/data")
DEFAULT_DATASETS = ("LUBM", "SWDF", "YAGO", "WIKIDATA")
DEFAULT_SPLITS = ("train", "val", "test")


def canonical_endpoint(edge: list[int]) -> tuple[int, int]:
    if len(edge) != 2:
        raise ValueError(f"Expected a two-node edge, got {edge!r}")
    source, target = int(edge[0]), int(edge[1])
    return (source, target) if source <= target else (target, source)


def load_payload(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return payload["0"] if "0" in payload else payload


def audit_graph(task: tuple[str, str, str]) -> dict:
    dataset, split, raw_path = task
    path = Path(raw_path)
    graph = load_payload(path)
    edges = graph.get("edge_indices", [])
    edge_features = graph.get("edge_features", [])
    if len(edges) != len(edge_features):
        raise ValueError(
            f"{path}: edge_indices has {len(edges)} entries but edge_features "
            f"has {len(edge_features)}"
        )

    endpoint_relations: dict[tuple[int, int], list[str]] = defaultdict(list)
    directed_counts = Counter()
    for edge, feature in zip(edges, edge_features):
        source, target = int(edge[0]), int(edge[1])
        endpoint_relations[canonical_endpoint(edge)].append(str(feature.get("id", "")))
        directed_counts[(source, target)] += 1

    multiplicities = [len(relations) for relations in endpoint_relations.values()]
    parallel_groups = [
        relations for relations in endpoint_relations.values() if len(relations) > 1
    ]
    duplicate_occurrences = len(edges) - len(endpoint_relations)
    return {
        "dataset": dataset,
        "split": split,
        "name": path.name,
        "path": str(path),
        "nodes": len(graph.get("node_features", [])),
        "raw_edges": len(edges),
        "undirected_endpoint_pairs": len(endpoint_relations),
        "duplicate_occurrences": duplicate_occurrences,
        "parallel_endpoint_groups": len(parallel_groups),
        "max_multiplicity": max(multiplicities, default=0),
        "same_label_parallel_groups": sum(
            len(set(relations)) == 1 for relations in parallel_groups
        ),
        "mixed_label_parallel_groups": sum(
            len(set(relations)) > 1 for relations in parallel_groups
        ),
        "same_directed_duplicate_groups": sum(
            count > 1 for count in directed_counts.values()
        ),
        "multiplicity_histogram": dict(Counter(multiplicities)),
    }


def graph_tasks(
    data_root: Path, datasets: Iterable[str], splits: Iterable[str]
) -> list[tuple[str, str, str]]:
    tasks = []
    for dataset in datasets:
        dataset_root = data_root / dataset
        if not dataset_root.is_dir():
            raise FileNotFoundError(f"Dataset directory does not exist: {dataset_root}")
        for split in splits:
            split_root = dataset_root / split
            if not split_root.is_dir():
                raise FileNotFoundError(f"Split directory does not exist: {split_root}")
            tasks.extend(
                (dataset, split, str(path))
                for path in sorted(split_root.glob("*.json"))
            )
    return tasks


def empty_summary() -> dict:
    return {
        "graphs": 0,
        "graphs_with_parallel_edges": 0,
        "nodes": 0,
        "raw_edges": 0,
        "undirected_endpoint_pairs": 0,
        "duplicate_occurrences": 0,
        "parallel_endpoint_groups": 0,
        "same_label_parallel_groups": 0,
        "mixed_label_parallel_groups": 0,
        "same_directed_duplicate_groups": 0,
        "max_multiplicity": 0,
        "multiplicity_histogram": Counter(),
        "top_affected_graphs": [],
    }


def add_graph(summary: dict, graph: dict, example_limit: int) -> None:
    summary["graphs"] += 1
    summary["graphs_with_parallel_edges"] += int(graph["duplicate_occurrences"] > 0)
    for key in (
        "nodes",
        "raw_edges",
        "undirected_endpoint_pairs",
        "duplicate_occurrences",
        "parallel_endpoint_groups",
        "same_label_parallel_groups",
        "mixed_label_parallel_groups",
        "same_directed_duplicate_groups",
    ):
        summary[key] += graph[key]
    summary["max_multiplicity"] = max(
        summary["max_multiplicity"], graph["max_multiplicity"]
    )
    summary["multiplicity_histogram"].update(
        {int(key): value for key, value in graph["multiplicity_histogram"].items()}
    )

    if graph["duplicate_occurrences"]:
        summary["top_affected_graphs"].append(
            {
                "split": graph["split"],
                "graph": graph["name"],
                "raw_edges": graph["raw_edges"],
                "duplicate_occurrences": graph["duplicate_occurrences"],
                "parallel_endpoint_groups": graph["parallel_endpoint_groups"],
                "max_multiplicity": graph["max_multiplicity"],
            }
        )
        summary["top_affected_graphs"].sort(
            key=lambda item: (
                -item["duplicate_occurrences"],
                -item["max_multiplicity"],
                item["split"],
                item["graph"],
            )
        )
        del summary["top_affected_graphs"][example_limit:]


def finalize_summary(summary: dict) -> dict:
    graphs = summary["graphs"]
    raw_edges = summary["raw_edges"]
    groups = summary["parallel_endpoint_groups"]
    summary["graph_parallel_rate"] = (
        summary["graphs_with_parallel_edges"] / graphs if graphs else 0.0
    )
    summary["duplicate_occurrence_rate"] = (
        summary["duplicate_occurrences"] / raw_edges if raw_edges else 0.0
    )
    summary["mixed_label_parallel_group_rate"] = (
        summary["mixed_label_parallel_groups"] / groups if groups else 0.0
    )
    summary["multiplicity_histogram"] = {
        str(key): value
        for key, value in sorted(summary["multiplicity_histogram"].items())
    }
    return summary


def audit_pair_exposure(dataset_root: Path, split: str, graph_lookup: dict) -> dict:
    pair_path = dataset_root / f"{split}_GEDINFO.json"
    if not pair_path.is_file():
        return {"available": False, "reason": f"Missing {pair_path}"}
    with pair_path.open("r", encoding="utf-8") as handle:
        pairs = json.load(handle).get("pairs_info", [])

    exposed = 0
    both = 0
    known_pairs = 0
    unknown_graphs = set()
    for entry in pairs:
        left_name, right_name = entry[:2]
        left = graph_lookup.get(left_name)
        right = graph_lookup.get(right_name)
        if left is None:
            unknown_graphs.add(left_name)
        if right is None:
            unknown_graphs.add(right_name)
        if left is None or right is None:
            continue
        known_pairs += 1
        left_affected = left["duplicate_occurrences"] > 0
        right_affected = right["duplicate_occurrences"] > 0
        exposed += int(left_affected or right_affected)
        both += int(left_affected and right_affected)

    return {
        "available": True,
        "pairs": len(pairs),
        "known_pairs": known_pairs,
        "unknown_graph_files": sorted(unknown_graphs),
        "pairs_with_at_least_one_parallel_graph": exposed,
        "exposure_rate": exposed / known_pairs if known_pairs else 0.0,
        "pairs_with_both_parallel_graphs": both,
    }


def markdown_report(payload: dict) -> str:
    lines = [
        "# SEABED 重边审计",
        "",
        "> 只读统计；未修改图、GED 真实值、单位代价、偏好标签或模型。",
        "",
        "重边定义：将端点 `(u, v)` 与 `(v, u)` 视为同一无向端点对；同一端点对出现多次即为重边。",
        "",
        "额外重边数：同一无向端点对保留一条边后，其余会被覆盖的边数。",
        "",
        "| 数据集 | 额外重边数 | 受影响图 | 影响比例 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for dataset in payload["datasets"]:
        summary = payload["datasets"][dataset]["all"]
        lines.append(
            f"| {dataset} | {summary['duplicate_occurrences']} | "
            f"{summary['graphs_with_parallel_edges']} / {summary['graphs']} | "
            f"{summary['graph_parallel_rate']:.2%} |"
        )
    return "\n".join(lines)


def print_terminal_summary(payload: dict) -> None:
    headers = (
        "Dataset",
        "Extra edges",
        "Affected graphs",
        "Affected rate",
    )
    rows = []
    for dataset, dataset_payload in payload["datasets"].items():
        summary = dataset_payload["all"]
        rows.append(
            (
                dataset,
                str(summary["duplicate_occurrences"]),
                (
                    f"{summary['graphs_with_parallel_edges']} / "
                    f"{summary['graphs']}"
                ),
                f"{summary['graph_parallel_rate']:.4%}",
            )
        )

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows))
        for index in range(len(headers))
    ]

    def render(row: tuple[str, ...], header: bool = False) -> str:
        cells = []
        for index, value in enumerate(row):
            if index == 0 or header:
                cells.append(value.ljust(widths[index]))
            else:
                cells.append(value.rjust(widths[index]))
        return "  ".join(cells)

    print("\n" + render(headers, header=True))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print(render(row))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(DEFAULT_DATASETS),
        help="Dataset directory names, or 'all'.",
    )
    parser.add_argument("--splits", nargs="+", default=list(DEFAULT_SPLITS))
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, (os.cpu_count() or 2) // 2)),
    )
    parser.add_argument("--progress-interval", type=int, default=5000)
    parser.add_argument("--example-limit", type=int, default=10)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("experiments/audits/seabed_multiedges.json"),
    )
    parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("experiments/audits/seabed_multiedges.md"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    datasets = list(DEFAULT_DATASETS) if args.datasets == ["all"] else args.datasets
    unknown_splits = set(args.splits) - set(DEFAULT_SPLITS)
    if unknown_splits:
        raise ValueError(f"Unknown splits: {sorted(unknown_splits)}")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    tasks = graph_tasks(args.data_root, datasets, args.splits)
    print(
        f"Auditing {len(tasks)} graph files from {len(datasets)} datasets "
        f"with {args.workers} workers...",
        flush=True,
    )
    if args.workers == 1:
        results = map(audit_graph, tasks)
    else:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        results = executor.map(audit_graph, tasks, chunksize=128)

    all_summaries = {dataset: empty_summary() for dataset in datasets}
    split_summaries = {
        dataset: {split: empty_summary() for split in args.splits}
        for dataset in datasets
    }
    graph_lookups = {dataset: {} for dataset in datasets}
    try:
        for index, graph in enumerate(results, 1):
            add_graph(all_summaries[graph["dataset"]], graph, args.example_limit)
            add_graph(
                split_summaries[graph["dataset"]][graph["split"]],
                graph,
                args.example_limit,
            )
            dataset_lookup = graph_lookups[graph["dataset"]]
            if graph["name"] in dataset_lookup:
                raise ValueError(
                    f"Duplicate graph filename across splits in {graph['dataset']}: "
                    f"{graph['name']}"
                )
            dataset_lookup[graph["name"]] = graph
            if args.progress_interval and index % args.progress_interval == 0:
                print(f"Audited {index}/{len(tasks)} graph files...", flush=True)
    finally:
        if args.workers != 1:
            executor.shutdown()

    payload = {
        "audit": "seabed_undirected_parallel_edge_audit_v1",
        "definition": (
            "(u,v) and (v,u) share one undirected endpoint key; repeated raw "
            "occurrences are parallel edges, matching V16 last-write projection"
        ),
        "read_only": True,
        "ground_truth_changed": False,
        "unit_cost_changed": False,
        "preference_labels_changed": False,
        "data_root": str(args.data_root.resolve()),
        "splits": args.splits,
        "datasets": {},
    }
    for dataset in datasets:
        payload["datasets"][dataset] = {
            "all": finalize_summary(all_summaries[dataset]),
            "by_split": {
                split: finalize_summary(split_summaries[dataset][split])
                for split in args.splits
            },
            "pair_exposure": {
                split: audit_pair_exposure(
                    args.data_root / dataset, split, graph_lookups[dataset]
                )
                for split in args.splits
            },
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")

    print_terminal_summary(payload)
    print(f"\nSaved JSON: {args.output}")
    print(f"Saved Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
