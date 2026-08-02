"""Audit semantic tie space on saved executable paths without model inference."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
for path in (PROJECT_ROOT, V2_DIR, V3_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_evaluator import (
    build_multirelation_edit_path,
    build_multirelation_graph,
    build_simple_edit_path,
    build_simple_graph,
)
from semantic_tie import (
    cosine_similarity_matrix,
    matrix_mapping_score,
    optimize_equal_cost_mapping,
    shared_entity_alignment,
)
from topology_reindex import derive_topology_feature_order, reorder_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure dual-cost-preserving semantic repair space on saved paths."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--path-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
    parser.add_argument("--max-iterations", type=int, default=20)
    return parser.parse_args()


def load_graph(path):
    with open(path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    graph = payload["0"] if "0" in payload else payload
    reindex = derive_topology_feature_order(graph)
    raw_features = [
        np.asarray(node["embedding"], dtype=float).reshape(-1).tolist()
        for node in graph["node_features"]
    ]
    features = reorder_features(raw_features, reindex.permutation)
    edge_labels = [str(edge["id"]) for edge in graph["edge_features"]]
    return {
        "ids": reindex.node_ids,
        "features": features,
        "simple": build_simple_graph(len(reindex.node_ids), graph["edge_indices"], edge_labels),
        "multi": build_multirelation_graph(len(reindex.node_ids), graph["edge_indices"], edge_labels),
    }


def analyze_pair(record, dataset_root, graph_cache, max_iterations):
    def cached_graph(file_name):
        if file_name not in graph_cache:
            graph_cache[file_name] = load_graph(dataset_root / "test" / file_name)
        return graph_cache[file_name]

    source = cached_graph(record["source_file"])
    target = cached_graph(record["target_file"])
    mapping = tuple(record["simple_graph_path"]["mapping"])
    score_matrix = cosine_similarity_matrix(source["features"], target["features"])
    cost_cache = {}

    def dual_cost(candidate):
        candidate = tuple(candidate)
        if candidate not in cost_cache:
            simple_cost = build_simple_edit_path(
                candidate, source["simple"], target["simple"]
            )["total_cost"]
            multi_cost = build_multirelation_edit_path(
                candidate, source["multi"], target["multi"]
            )["total_cost"]
            cost_cache[candidate] = (simple_cost, multi_cost)
        return cost_cache[candidate]

    initial_cost = dual_cost(mapping)
    recorded_cost = (
        record["simple_graph_path"]["total_cost"],
        record["multirelation_path"]["total_cost"],
    )
    if initial_cost != recorded_cost:
        raise RuntimeError(
            f"Saved-path cost mismatch for pair {record['pair_index']}: "
            f"computed={initial_cost}, recorded={recorded_cost}."
        )

    def identity_score(candidate):
        return shared_entity_alignment(
            candidate, source["ids"], target["ids"]
        )[0]

    def embedding_score(candidate):
        return matrix_mapping_score(candidate, score_matrix)

    oracle = optimize_equal_cost_mapping(
        mapping,
        target["simple"].num_nodes,
        dual_cost,
        identity_score,
        max_iterations=max_iterations,
    )
    embedding = optimize_equal_cost_mapping(
        mapping,
        target["simple"].num_nodes,
        dual_cost,
        embedding_score,
        max_iterations=max_iterations,
    )
    initial_aligned, shared = shared_entity_alignment(
        mapping, source["ids"], target["ids"]
    )
    oracle_aligned, _ = shared_entity_alignment(
        oracle.mapping, source["ids"], target["ids"]
    )
    embedding_aligned, _ = shared_entity_alignment(
        embedding.mapping, source["ids"], target["ids"]
    )
    if dual_cost(oracle.mapping) != initial_cost or dual_cost(embedding.mapping) != initial_cost:
        raise RuntimeError("A semantic repair changed simple or multirelation cost.")

    return {
        "pair_index": record["pair_index"],
        "source_file": record["source_file"],
        "target_file": record["target_file"],
        "simple_cost": initial_cost[0],
        "multirelation_cost": initial_cost[1],
        "shared_entities": shared,
        "initial_aligned": initial_aligned,
        "id_oracle_aligned": oracle_aligned,
        "embedding_aligned": embedding_aligned,
        "id_oracle": oracle.__dict__,
        "embedding": embedding.__dict__,
    }


def aggregate(dataset, path_file, records):
    shared = sum(record["shared_entities"] for record in records)
    initial = sum(record["initial_aligned"] for record in records)
    oracle = sum(record["id_oracle_aligned"] for record in records)
    embedding = sum(record["embedding_aligned"] for record in records)
    rate = lambda value: round(value / shared if shared else 0.0, 4)
    return {
        "version": "v5_tie_space_audit",
        "scope": "saved-path offline diagnostic; no model inference",
        "dataset": dataset,
        "path_file": str(path_file),
        "pairs": len(records),
        "shared_entities": shared,
        "alignment": {
            "initial": initial,
            "initial_rate": rate(initial),
            "id_local_oracle": oracle,
            "id_local_oracle_rate": rate(oracle),
            "embedding_tie_break": embedding,
            "embedding_tie_break_rate": rate(embedding),
        },
        "tie_space": {
            "pairs_with_equal_cost_neighbor": sum(
                record["id_oracle"]["initial_equal_cost_neighbors"] > 0
                for record in records
            ),
            "pairs_with_id_improving_neighbor": sum(
                record["id_oracle"]["initial_improving_neighbors"] > 0
                for record in records
            ),
            "pairs_improved_by_id_oracle": sum(
                record["id_oracle_aligned"] > record["initial_aligned"]
                for record in records
            ),
            "pairs_improved_by_embedding": sum(
                record["embedding_aligned"] > record["initial_aligned"]
                for record in records
            ),
            "pairs_harmed_by_embedding": sum(
                record["embedding_aligned"] < record["initial_aligned"]
                for record in records
            ),
        },
        "cost_invariants": {
            "simple_unchanged_rate": 1.0,
            "multirelation_unchanged_rate": 1.0,
        },
        "per_pair": records,
    }


def main():
    args = parse_args()
    if args.max_pairs < 0 or args.max_iterations < 0:
        raise ValueError("--max-pairs and --max-iterations must be non-negative.")
    dataset_root = Path(args.dataset_root).resolve()
    path_file = Path(args.path_file).resolve()
    output = Path(args.output).resolve()
    graph_cache = {}
    records = []
    with open(path_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if args.max_pairs and len(records) >= args.max_pairs:
                break
            records.append(
                analyze_pair(
                    json.loads(line), dataset_root, graph_cache, args.max_iterations
                )
            )
    if not records:
        raise RuntimeError("No saved paths were available for analysis.")
    result = aggregate(args.dataset, path_file, records)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "per_pair"}, indent=2))
    print("Saved tie-space audit:", output)


if __name__ == "__main__":
    main()
