"""Audit whether SEABED node embeddings behave like entity identifiers."""

import argparse
import json
from pathlib import Path
import sys

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
for path in (PROJECT_ROOT, V3_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embedding_evidence import analyze_pair_embeddings, count_rate, distribution
from topology_reindex import derive_topology_feature_order, reorder_features


def parse_args():
    parser = argparse.ArgumentParser(
        description="Measure entity identity and cosine-neighbor evidence in saved paths."
    )
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--path-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-pairs", type=int, default=100)
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
    return {
        "ids": reindex.node_ids,
        "features": reorder_features(raw_features, reindex.permutation),
    }


def merge_evidence(all_evidence):
    list_keys = (
        "shared_entity_ids",
        "shared_exact",
        "correct_cosines",
        "correct_top1",
        "correct_unique_top1",
        "correct_strict_top1",
        "correct_reciprocal_top1",
        "correct_ranks",
        "correct_margins",
        "max_incorrect_cosines",
        "nonshared_max_cosines",
    )
    merged = {key: [] for key in list_keys}
    merged["incorrect_exact_collisions"] = 0
    for evidence in all_evidence:
        for key in list_keys:
            merged[key].extend(evidence[key])
        merged["incorrect_exact_collisions"] += evidence["incorrect_exact_collisions"]
    return merged


def aggregate(args, records, evidence):
    shared_count = len(evidence["shared_entity_ids"])
    ranks = evidence["correct_ranks"]
    unique_ids = set(evidence["shared_entity_ids"])
    exact_by_id = {entity_id: True for entity_id in unique_ids}
    for entity_id, exact in zip(evidence["shared_entity_ids"], evidence["shared_exact"]):
        exact_by_id[entity_id] = exact_by_id[entity_id] and bool(exact)

    return {
        "version": "v7_embedding_evidence_audit",
        "scope": "offline audit of V6 saved smoke paths; no checkpoint or inference",
        "dataset": args.dataset,
        "path_file": str(Path(args.path_file).resolve()),
        "coverage": {
            "pairs": len(records),
            "unique_source_graphs": len({record["source_file"] for record in records}),
            "unique_target_graphs": len({record["target_file"] for record in records}),
        },
        "entity_overlap": {
            "shared_entity_observations": shared_count,
            "unique_shared_entity_ids": len(unique_ids),
            "nonshared_source_observations": len(evidence["nonshared_max_cosines"]),
        },
        "identity_evidence": {
            "exact_shared_vectors": count_rate(evidence["shared_exact"]),
            "unique_ids_exact_in_every_observation": {
                "count": sum(exact_by_id.values()),
                "total": len(exact_by_id),
                "rate": round(
                    sum(exact_by_id.values()) / len(exact_by_id) if exact_by_id else 0.0,
                    6,
                ),
            },
            "correct_entity_is_cosine_top1": count_rate(evidence["correct_top1"]),
            "correct_entity_is_unique_top1": count_rate(
                evidence["correct_unique_top1"]
            ),
            "correct_entity_beats_every_incorrect": count_rate(
                evidence["correct_strict_top1"]
            ),
            "correct_entity_is_reciprocal_top1": count_rate(
                evidence["correct_reciprocal_top1"]
            ),
            "mean_reciprocal_rank": round(
                sum(1.0 / rank for rank in ranks) / len(ranks) if ranks else 0.0,
                6,
            ),
            "incorrect_exact_vector_collisions": evidence[
                "incorrect_exact_collisions"
            ],
        },
        "cosine_distributions": {
            "correct_entity": distribution(evidence["correct_cosines"]),
            "best_incorrect_entity": distribution(
                evidence["max_incorrect_cosines"]
            ),
            "correct_minus_best_incorrect_margin": distribution(
                evidence["correct_margins"]
            ),
            "nonshared_source_best_target": distribution(
                evidence["nonshared_max_cosines"]
            ),
            "correct_entity_rank": distribution(ranks),
        },
        "interpretation": {
            "entity_ids_used_by_v6_selection": False,
            "entity_ids_used_by_this_audit": True,
            "low_overlap_warning": shared_count < 100,
            "smoke_prefix_is_not_representative": True,
        },
    }


def main():
    args = parse_args()
    if args.max_pairs < 0:
        raise ValueError("--max-pairs must be non-negative.")
    dataset_root = Path(args.dataset_root).resolve()
    graph_cache = {}
    records = []
    evidence = []

    def cached_graph(file_name):
        if file_name not in graph_cache:
            graph_cache[file_name] = load_graph(dataset_root / "test" / file_name)
        return graph_cache[file_name]

    with open(args.path_file, "r", encoding="utf-8") as handle:
        for line in handle:
            if args.max_pairs and len(records) >= args.max_pairs:
                break
            record = json.loads(line)
            source = cached_graph(record["source_file"])
            target = cached_graph(record["target_file"])
            records.append(record)
            evidence.append(
                analyze_pair_embeddings(
                    source["ids"],
                    source["features"],
                    target["ids"],
                    target["features"],
                )
            )
    if not records:
        raise RuntimeError("No saved path records were available for audit.")

    result = aggregate(args, records, merge_evidence(evidence))
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    print(json.dumps(result, indent=2))
    print("Saved embedding evidence audit:", output)


if __name__ == "__main__":
    main()

