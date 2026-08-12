#!/usr/bin/env python3
"""CPU-only structural and aggregate-error audit for the frozen V15 SWDF run.

This script never loads a model or checkpoint. It reads the benchmark JSON files
and already-written V15 result summaries, leaving GED labels and evaluators
unchanged.
"""

import argparse
from collections import Counter
import json
import math
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("/data/projects/SEABED-main/data/SWDF")
DEFAULT_V15_ROOT = Path(
    "experiments/seabed_versions/v15_benchmark_projected_relation_input"
)
DEFAULT_RAW_RESULT = DEFAULT_V15_ROOT / (
    "training_results/"
    "result_SEABED_SWDF_test_BPR_gedcol3_unit_20260811_214247.json"
)
DEFAULT_PROJECTED_RESULT = DEFAULT_V15_ROOT / (
    "training_results/"
    "result_SEABED_SWDF_test_BPR_gedcol3_unit_20260812_011747.json"
)
DEFAULT_OUTPUT = DEFAULT_V15_ROOT / "cpu_audit_results/swdf_v15_cpu_audit.json"
DEFAULT_MARKDOWN = DEFAULT_V15_ROOT / "cpu_audit_results/RESULTS.md"


def canonical_endpoint(source, target):
    source = int(source)
    target = int(target)
    return (source, target) if source <= target else (target, source)


def load_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def load_graph(path):
    payload = load_json(path)
    graph = payload["0"] if "0" in payload else payload
    raw_edges = graph.get("edge_indices", [])
    endpoint_counts = Counter(canonical_endpoint(*edge) for edge in raw_edges)
    simple_edges = len(endpoint_counts)
    raw_edge_count = len(raw_edges)
    dropped_edges = raw_edge_count - simple_edges
    return {
        "path": str(path),
        "nodes": len(graph.get("node_features", [])),
        "raw_edges": raw_edge_count,
        "simple_edges": simple_edges,
        "dropped_edges": dropped_edges,
        "dropped_fraction": (
            dropped_edges / raw_edge_count if raw_edge_count else 0.0
        ),
        "parallel_endpoint_pairs": sum(
            multiplicity > 1 for multiplicity in endpoint_counts.values()
        ),
        "max_multiplicity": max(endpoint_counts.values(), default=0),
    }


def index_graphs(dataset_root):
    graph_paths = {}
    graph_splits = {}
    for split in ("train", "val", "test"):
        for path in (dataset_root / split).glob("*.json"):
            if path.name in graph_paths:
                raise ValueError(f"Duplicate graph filename: {path.name}")
            graph_paths[path.name] = path
            graph_splits[path.name] = split
    return graph_paths, graph_splits


def integer_bucket(value, boundaries):
    for upper, label in boundaries:
        if value <= upper:
            return label
    return boundaries[-1][1]


def fraction_bucket(value):
    if value == 0:
        return "0"
    if value <= 0.10:
        return "(0, 0.10]"
    if value <= 0.25:
        return "(0.10, 0.25]"
    return "> 0.25"


def rounded_count(rate, total):
    return int(round(float(rate) * total))


def metric_interval(value, decimals, total):
    half_step = 0.5 * (10 ** -decimals)
    lower = max(0.0, (float(value) - half_step) * total)
    upper = (float(value) + half_step) * total
    return [math.ceil(lower - 1e-12), math.floor(upper - 1e-12)]


def infer_aggregate_error_constraints(result, total_pairs):
    exact = rounded_count(result["acc"], total_pairs)
    feasible = rounded_count(result["fea"], total_pairs)
    wrong = total_pairs - exact
    below_gt = total_pairs - feasible
    above_gt = feasible - exact
    mae_sum_interval = metric_interval(result["mae"], 3, total_pairs)
    squared_sum_interval = metric_interval(result["mse"], 3, total_pairs)

    # The following is an illustrative reconstruction from the rounded central
    # values, not a claim that per-pair predictions were saved.
    absolute_sum = int(round(float(result["mae"]) * total_pairs))
    squared_sum = int(round(float(result["mse"]) * total_pairs))
    excess_absolute = absolute_sum - wrong
    moment_remainder = squared_sum - absolute_sum - 2 * excess_absolute
    illustrative = None
    if wrong >= 0 and excess_absolute >= 0 and moment_remainder >= 0:
        # When no error exceeds 3, n3 = remainder / 2 and
        # n2 = excess_absolute - 2*n3.
        if moment_remainder % 2 == 0:
            magnitude_3 = moment_remainder // 2
            magnitude_2 = excess_absolute - 2 * magnitude_3
            magnitude_1 = wrong - magnitude_2 - magnitude_3
            if min(magnitude_1, magnitude_2, magnitude_3) >= 0:
                illustrative = {
                    "assumption": "central rounded metrics and no |error| above 3",
                    "abs_error_1_pairs": magnitude_1,
                    "abs_error_2_pairs": magnitude_2,
                    "abs_error_3_pairs": magnitude_3,
                }

    return {
        "pairs": total_pairs,
        "exact_pairs_from_rounded_acc": exact,
        "non_exact_pairs_from_rounded_acc": wrong,
        "below_gt_pairs_from_rounded_fea": below_gt,
        "above_gt_pairs_from_rounded_acc_and_fea": above_gt,
        "absolute_error_sum_interval_from_rounded_mae": mae_sum_interval,
        "squared_error_sum_interval_from_rounded_mse": squared_sum_interval,
        "minimum_absolute_error_sum_if_every_wrong_pair_is_off_by_one": wrong,
        "central_absolute_error_beyond_off_by_one_minimum": excess_absolute,
        "illustrative_error_magnitude_reconstruction": illustrative,
        "warning": (
            "Counts are inferred from three-decimal aggregate metrics. Exact "
            "per-pair buckets require rerunning frozen-checkpoint inference "
            "while saving mappings."
        ),
    }


def audit_dataset(dataset_root, split):
    graph_paths, graph_splits = index_graphs(dataset_root)
    pair_path = dataset_root / f"{split}_GEDINFO.json"
    pairs = load_json(pair_path)["pairs_info"]
    graph_cache = {}

    def graph_record(name):
        if name not in graph_paths:
            raise KeyError(f"Pair references missing graph: {name}")
        if name not in graph_cache:
            graph_cache[name] = load_graph(graph_paths[name])
        return graph_cache[name]

    gt_buckets = Counter()
    node_gap_buckets = Counter()
    collapse_count_buckets = Counter()
    collapse_fraction_buckets = Counter()
    split_pair_counts = Counter()
    pairs_with_collapse = 0
    pairs_with_both_collapsed = 0
    multigraph_lower_bound_certificates = 0
    simple_lower_bound_violations = 0
    total_dropped_edge_occurrences = 0
    total_raw_edge_occurrences = 0
    ged_values = []
    node_gaps = []
    simple_edge_gaps = []

    ged_boundaries = (
        (0, "0"),
        (4, "1-4"),
        (8, "5-8"),
        (12, "9-12"),
        (math.inf, ">=13"),
    )
    collapse_boundaries = (
        (0, "0"),
        (2, "1-2"),
        (4, "3-4"),
        (math.inf, ">=5"),
    )

    for entry in pairs:
        file_1, file_2 = entry[:2]
        gt = float(entry[2])
        left = graph_record(file_1)
        right = graph_record(file_2)
        dropped = left["dropped_edges"] + right["dropped_edges"]
        raw_edges = left["raw_edges"] + right["raw_edges"]
        dropped_fraction = dropped / raw_edges if raw_edges else 0.0
        node_gap = abs(left["nodes"] - right["nodes"])
        simple_edge_gap = abs(left["simple_edges"] - right["simple_edges"])
        raw_edge_gap = abs(left["raw_edges"] - right["raw_edges"])
        simple_lower_bound = node_gap + simple_edge_gap
        multigraph_lower_bound = node_gap + raw_edge_gap

        ged_values.append(gt)
        node_gaps.append(node_gap)
        simple_edge_gaps.append(simple_edge_gap)
        gt_buckets[integer_bucket(gt, ged_boundaries)] += 1
        node_gap_buckets[integer_bucket(node_gap, ((0, "0"), (2, "1-2"), (4, "3-4"), (math.inf, ">=5")))] += 1
        collapse_count_buckets[integer_bucket(dropped, collapse_boundaries)] += 1
        collapse_fraction_buckets[fraction_bucket(dropped_fraction)] += 1
        split_pair_counts[f"{graph_splits[file_1]}->{graph_splits[file_2]}"] += 1
        pairs_with_collapse += dropped > 0
        pairs_with_both_collapsed += (
            left["dropped_edges"] > 0 and right["dropped_edges"] > 0
        )
        multigraph_lower_bound_certificates += multigraph_lower_bound > gt
        simple_lower_bound_violations += simple_lower_bound > gt
        total_dropped_edge_occurrences += dropped
        total_raw_edge_occurrences += raw_edges

    involved_graphs = list(graph_cache.values())
    return {
        "dataset_root": str(dataset_root),
        "split": split,
        "ged_column": 3,
        "pairs": len(pairs),
        "pair_split_origins": dict(sorted(split_pair_counts.items())),
        "involved_unique_graphs": len(involved_graphs),
        "involved_graph_summary": {
            "graphs_with_collapsed_edges": sum(
                graph["dropped_edges"] > 0 for graph in involved_graphs
            ),
            "raw_edges": sum(graph["raw_edges"] for graph in involved_graphs),
            "simple_last_write_edges": sum(
                graph["simple_edges"] for graph in involved_graphs
            ),
            "collapsed_edges": sum(
                graph["dropped_edges"] for graph in involved_graphs
            ),
            "max_multiplicity": max(
                (graph["max_multiplicity"] for graph in involved_graphs),
                default=0,
            ),
        },
        "pair_collapse_exposure": {
            "pairs_with_at_least_one_collapsed_graph": pairs_with_collapse,
            "rate": pairs_with_collapse / len(pairs),
            "pairs_with_both_graphs_collapsed": pairs_with_both_collapsed,
            "pair_edge_occurrence_drop_fraction": (
                total_dropped_edge_occurrences / total_raw_edge_occurrences
                if total_raw_edge_occurrences
                else 0.0
            ),
            "collapsed_edge_count_buckets": dict(collapse_count_buckets),
            "collapsed_edge_fraction_buckets": dict(collapse_fraction_buckets),
        },
        "ground_truth_and_difficulty": {
            "ged_min": min(ged_values),
            "ged_max": max(ged_values),
            "ged_mean": sum(ged_values) / len(ged_values),
            "ged_buckets": dict(gt_buckets),
            "node_gap_mean": sum(node_gaps) / len(node_gaps),
            "node_gap_buckets": dict(node_gap_buckets),
            "simple_edge_count_gap_mean": sum(simple_edge_gaps) / len(simple_edge_gaps),
        },
        "cost_definition_evidence": {
            "simple_graph_count_lower_bound_violations": simple_lower_bound_violations,
            "full_multigraph_lower_bound_certificates": multigraph_lower_bound_certificates,
            "certificate_meaning": (
                "column-3 GED is below the unavoidable node/raw-edge-count "
                "cost of any full-multigraph edit path"
            ),
        },
    }


def validate_result(result, label):
    config = result.get("config", {})
    expected = {
        "dataset": "SWDF",
        "ged_column": 3,
        "cost_mode": "unit",
        "use_raw_features": 1,
    }
    mismatches = {
        key: {"expected": value, "actual": config.get(key)}
        for key, value in expected.items()
        if config.get(key) != value
    }
    if mismatches:
        raise ValueError(f"{label} result is not the fixed V11 task: {mismatches}")


def metric_comparison(raw, projected):
    return {
        metric: {
            "raw": raw[metric],
            "projected_input": projected[metric],
            "projected_minus_raw": round(projected[metric] - raw[metric], 6),
        }
        for metric in ("mae", "acc", "mse", "fea", "rho", "tau")
    }


def markdown_report(payload):
    dataset = payload["dataset_audit"]
    exposure = dataset["pair_collapse_exposure"]
    evidence = dataset["cost_definition_evidence"]
    comparison = payload["v15_metric_comparison"]
    raw_error = payload["aggregate_error_constraints"]["raw"]
    projected_error = payload["aggregate_error_constraints"]["projected_input"]
    difficulty = dataset["ground_truth_and_difficulty"]
    lines = [
        "# V15 SWDF CPU Audit",
        "",
        "This is a read-only audit. It does not change GED cost, column-3 labels,",
        "preference supervision, model code, checkpoints, or the primary MAE/ACC results.",
        "",
        "## V15 paired result",
        "",
        "| Metric | Raw | Projected input | Change |",
        "|---|---:|---:|---:|",
    ]
    for metric in ("mae", "acc", "mse", "fea", "rho", "tau"):
        item = comparison[metric]
        lines.append(
            f"| {metric.upper()} | {item['raw']:.3f} | "
            f"{item['projected_input']:.3f} | {item['projected_minus_raw']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Test-pair structure",
            "",
            f"- Test pairs: {dataset['pairs']}",
            f"- Unique test graphs involved: {dataset['involved_unique_graphs']}",
            "- Pairs exposed to at least one collapsed graph: "
            f"{exposure['pairs_with_at_least_one_collapsed_graph']} "
            f"({exposure['rate']:.2%})",
            "- Pair-weighted raw edges removed by last-write projection: "
            f"{exposure['pair_edge_occurrence_drop_fraction']:.2%}",
            "- Full-multigraph lower-bound certificates: "
            f"{evidence['full_multigraph_lower_bound_certificates']}",
            "- Simple-graph count lower-bound violations: "
            f"{evidence['simple_graph_count_lower_bound_violations']}",
            f"- Official GED mean/range: {difficulty['ged_mean']:.4f} / "
            f"[{difficulty['ged_min']:.0f}, {difficulty['ged_max']:.0f}]",
            "",
            "## Aggregate error constraints",
            "",
            "The old result files did not save per-pair mappings. The following counts",
            "are therefore inferred from the rounded aggregate metrics, not reconstructed",
            "per-pair predictions.",
            "",
            "| Quantity | Raw | Projected input |",
            "|---|---:|---:|",
            "| Exact pairs | "
            f"{raw_error['exact_pairs_from_rounded_acc']} | "
            f"{projected_error['exact_pairs_from_rounded_acc']} |",
            "| Non-exact pairs | "
            f"{raw_error['non_exact_pairs_from_rounded_acc']} | "
            f"{projected_error['non_exact_pairs_from_rounded_acc']} |",
            "| Below-GT reports | "
            f"{raw_error['below_gt_pairs_from_rounded_fea']} | "
            f"{projected_error['below_gt_pairs_from_rounded_fea']} |",
            "| Above-GT reports | "
            f"{raw_error['above_gt_pairs_from_rounded_acc_and_fea']} | "
            f"{projected_error['above_gt_pairs_from_rounded_acc_and_fea']} |",
            "| Absolute-error mass beyond all-wrong-by-one minimum | "
            f"{raw_error['central_absolute_error_beyond_off_by_one_minimum']} | "
            f"{projected_error['central_absolute_error_beyond_off_by_one_minimum']} |",
            "",
            "## Interpretation",
            "",
            "The projection addresses a real input/target mismatch, but nearly every test",
            "pair is exposed to collapsed edges. Exposure is not the same as an error, so",
            "17.5% fewer unique input edges cannot be translated directly into 17.5 ACC",
            "points. The projected run adds about 160 exact pairs and removes about 50 units",
            "of aggregate absolute-error mass. Most remaining errors are tightly concentrated",
            "near one edit, while the below-GT reports show that a frozen-mapping executable-",
            "path audit is still needed before attributing every miss to the model.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--split", default="test")
    parser.add_argument("--raw-result", type=Path, default=DEFAULT_RAW_RESULT)
    parser.add_argument(
        "--projected-result", type=Path, default=DEFAULT_PROJECTED_RESULT
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MARKDOWN)
    return parser.parse_args()


def main():
    args = parse_args()
    raw_result = load_json(args.raw_result)
    projected_result = load_json(args.projected_result)
    validate_result(raw_result, "raw")
    validate_result(projected_result, "projected")
    dataset_audit = audit_dataset(args.dataset_root, args.split)
    total_pairs = dataset_audit["pairs"]
    payload = {
        "audit": "swdf_v15_cpu_read_only_v1",
        "non_negotiable_objective": {
            "cost_mode": "unit",
            "ged_column": 3,
            "ground_truth_changed": False,
            "preference_labels_changed": False,
            "model_or_checkpoint_loaded": False,
            "gpu_used": False,
            "primary_metrics": ["mae", "acc"],
        },
        "sources": {
            "raw_result": str(args.raw_result.resolve()),
            "projected_result": str(args.projected_result.resolve()),
        },
        "dataset_audit": dataset_audit,
        "v15_metric_comparison": metric_comparison(raw_result, projected_result),
        "aggregate_error_constraints": {
            "raw": infer_aggregate_error_constraints(raw_result, total_pairs),
            "projected_input": infer_aggregate_error_constraints(
                projected_result, total_pairs
            ),
        },
        "next_required_audit": {
            "requires_retraining": False,
            "requires_frozen_checkpoint_inference": True,
            "reason": (
                "Existing V15 summaries do not contain per-pair predicted GED or "
                "node mappings. Save those mappings once, then replay both the "
                "legacy dense evaluator and the official simple last-write "
                "executable-path evaluator on exactly the same correspondence."
            ),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")
    args.markdown.write_text(markdown_report(payload), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print(f"Saved JSON: {args.output}")
    print(f"Saved Markdown: {args.markdown}")


if __name__ == "__main__":
    main()
