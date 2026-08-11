#!/usr/bin/env python3
"""Audit whether SWDF GED labels account for parallel relation edges."""

import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path

import networkx as nx


@dataclass(frozen=True)
class GraphRecord:
    path: Path
    node_count: int
    raw_edges: tuple
    simple_edge_count: int

    @property
    def raw_edge_count(self):
        return len(self.raw_edges)

    @property
    def collapsed_edge_count(self):
        return self.raw_edge_count - self.simple_edge_count


def canonical_endpoint(source, target):
    return (source, target) if source <= target else (target, source)


def load_graph(path):
    with path.open(encoding="utf-8") as handle:
        graph = json.load(handle)["0"]

    raw_edges = tuple(
        (int(endpoint[0]), int(endpoint[1]), str(feature["id"]))
        for endpoint, feature in zip(graph["edge_indices"], graph["edge_features"])
    )
    endpoints = {canonical_endpoint(source, target) for source, target, _ in raw_edges}
    return GraphRecord(
        path=path,
        node_count=len(graph["node_features"]),
        raw_edges=raw_edges,
        simple_edge_count=len(endpoints),
    )


def build_graph(record, multigraph):
    graph = nx.MultiGraph() if multigraph else nx.Graph()
    graph.add_nodes_from(range(record.node_count))
    for source, target, relation in record.raw_edges:
        graph.add_edge(source, target, label=relation)
    return graph


def labeled_ged(left, right, timeout):
    return nx.graph_edit_distance(
        left,
        right,
        edge_match=lambda edge_1, edge_2: edge_1["label"] == edge_2["label"],
        timeout=timeout,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("/data/projects/SEABED-main/data/SWDF"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--selection",
        choices=("certificate", "affected-random"),
        default="certificate",
        help="Prefer strict counterexamples, or sample all pairs involving parallel edges.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="Per NetworkX GED call. A timed result is an upper bound, not an optimality proof.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pair_path = args.dataset_root / f"{args.split}_GEDINFO.json"
    graph_dir = args.dataset_root / args.split
    with pair_path.open(encoding="utf-8") as handle:
        pairs = json.load(handle)["pairs_info"]

    cache = {}

    def graph_record(name):
        if name not in cache:
            cache[name] = load_graph(graph_dir / name)
        return cache[name]

    affected = []
    certificates = []
    for file_1, file_2, ged_column_3, *_ in pairs:
        graph_1 = graph_record(file_1)
        graph_2 = graph_record(file_2)
        if graph_1.collapsed_edge_count + graph_2.collapsed_edge_count == 0:
            continue

        # Any unit-cost multigraph edit path must at least reconcile both the
        # node-count and raw-edge-count differences.
        multigraph_lower_bound = abs(graph_1.node_count - graph_2.node_count) + abs(
            graph_1.raw_edge_count - graph_2.raw_edge_count
        )
        item = (file_1, file_2, float(ged_column_3), multigraph_lower_bound)
        affected.append(item)
        if multigraph_lower_bound > float(ged_column_3):
            certificates.append(item)

    rng = random.Random(args.seed)
    rng.shuffle(certificates)
    if args.selection == "certificate":
        certificate_keys = {(item[0], item[1]) for item in certificates}
        remaining = [item for item in affected if (item[0], item[1]) not in certificate_keys]
        rng.shuffle(remaining)
        selected = (certificates + remaining)[: args.samples]
    else:
        rng.shuffle(affected)
        selected = affected[: args.samples]

    print(
        f"split={args.split} selection={args.selection} total_pairs={len(pairs)} "
        f"pairs_with_parallel_edges={len(affected)} "
        f"lower_bound_certificates={len(certificates)}"
    )
    print(
        "A certificate means: column-3 GED is smaller than the unavoidable "
        "node/edge-count cost of a full multigraph."
    )

    simple_matches = 0
    sampled_certificates = 0
    multigraph_results_above_official = 0
    for index, (file_1, file_2, official_ged, lower_bound) in enumerate(selected, 1):
        graph_1 = graph_record(file_1)
        graph_2 = graph_record(file_2)
        simple_ged = labeled_ged(
            build_graph(graph_1, multigraph=False),
            build_graph(graph_2, multigraph=False),
            args.timeout,
        )
        multi_ged = labeled_ged(
            build_graph(graph_1, multigraph=True),
            build_graph(graph_2, multigraph=True),
            args.timeout,
        )
        simple_matches += simple_ged == official_ged
        sampled_certificates += official_ged < lower_bound
        multigraph_results_above_official += multi_ged is not None and multi_ged > official_ged
        print(f"\n[{index}] {file_1} -> {file_2}")
        print(
            f"  G1 nodes/raw/simple/lost = {graph_1.node_count}/"
            f"{graph_1.raw_edge_count}/{graph_1.simple_edge_count}/"
            f"{graph_1.collapsed_edge_count}"
        )
        print(
            f"  G2 nodes/raw/simple/lost = {graph_2.node_count}/"
            f"{graph_2.raw_edge_count}/{graph_2.simple_edge_count}/"
            f"{graph_2.collapsed_edge_count}"
        )
        print(f"  official column-3 GED         = {official_ged:g}")
        print(f"  nx.Graph GED (timeout result) = {simple_ged}")
        print(f"  full-multigraph lower bound   = {lower_bound}")
        print(f"  nx.MultiGraph timeout result  = {multi_ged}")
        print(f"  certified_not_full_multigraph = {official_ged < lower_bound}")

    print("\nSample summary")
    print(f"  official_equals_nx.Graph      = {simple_matches}/{len(selected)}")
    print(f"  certified_not_full_multigraph = {sampled_certificates}/{len(selected)}")
    print(
        "  MultiGraph result > official  = "
        f"{multigraph_results_above_official}/{len(selected)}"
    )


if __name__ == "__main__":
    main()
