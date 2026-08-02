"""V8 inference: recover exact embedding anchors without changing path cost."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
V5_DIR = CURRENT_DIR.parent / "v5_tie_space_audit"
for path in (PROJECT_ROOT, V2_DIR, V3_DIR, V5_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from embedding_anchor import exact_embedding_matrix, mapping_anchor_count
from path_evaluator import build_multirelation_edit_path, build_simple_edit_path
from reindexed_trainer import TopologyFeatureReindexTrainer
from semantic_tie import optimize_equal_cost_mapping, shared_entity_alignment


class ExactEmbeddingAnchorTrainer(TopologyFeatureReindexTrainer):
    version = "v8_exact_embedding_anchor"
    anchor_revision = "dual_cost_exact_embedding_anchor_v1"

    def _reset_anchor_metrics(self):
        self.anchor_metrics = {
            "pairs": 0,
            "shared_entities": 0,
            "aligned_before": 0,
            "aligned_after": 0,
            "anchors_before": 0,
            "anchors_after": 0,
            "pairs_mapping_changed": 0,
            "pairs_anchor_improved": 0,
            "pairs_alignment_improved": 0,
            "pairs_alignment_harmed": 0,
            "iterations": 0,
            "candidates_evaluated": 0,
            "equal_cost_candidates": 0,
            "pairs_with_equal_cost_neighbor": 0,
            "pairs_with_anchor_improving_neighbor": 0,
            "simple_cost_changed_pairs": 0,
            "multirelation_cost_changed_pairs": 0,
            "seconds": 0.0,
        }

    def diffusion_ged_parallel(self, batch, test_k=100):
        final_ged, _, inference_time = super().diffusion_ged_parallel(batch, test_k)
        anchor_start = time.time()
        structural_repair = self.last_repair
        initial_mapping = tuple(structural_repair.mapping.detach().cpu().tolist())
        source_gid = int(batch.i_j[0][0].item())
        target_gid = int(batch.i_j[0][1].item())
        source_simple, source_multi = self._graph_views(source_gid)
        target_simple, target_multi = self._graph_views(target_gid)
        source_ids = self._node_ids(source_gid)
        target_ids = self._node_ids(target_gid)
        anchors = exact_embedding_matrix(
            self.graphs[source_gid]["features"],
            self.graphs[target_gid]["features"],
        )
        cost_cache = {}

        def dual_cost(mapping):
            mapping = tuple(mapping)
            if mapping not in cost_cache:
                cost_cache[mapping] = (
                    build_simple_edit_path(mapping, source_simple, target_simple)[
                        "total_cost"
                    ],
                    build_multirelation_edit_path(mapping, source_multi, target_multi)[
                        "total_cost"
                    ],
                )
            return cost_cache[mapping]

        def anchor_score(mapping):
            return mapping_anchor_count(mapping, anchors)

        initial_cost = dual_cost(initial_mapping)
        initial_anchor_count = anchor_score(initial_mapping)
        repair = optimize_equal_cost_mapping(
            initial_mapping,
            target_simple.num_nodes,
            dual_cost,
            anchor_score,
            max_iterations=self.args.anchor_max_iterations,
        )
        repaired_cost = dual_cost(repair.mapping)
        repaired_anchor_count = anchor_score(repair.mapping)
        if repaired_cost != initial_cost:
            raise RuntimeError(
                f"V8 changed a protected path cost for pair {self.current_pair_index}."
            )
        if repair.mapping != initial_mapping and repaired_anchor_count <= initial_anchor_count:
            raise RuntimeError(
                f"V8 changed mapping without adding an anchor for pair {self.current_pair_index}."
            )

        aligned_before, shared = shared_entity_alignment(
            initial_mapping, source_ids, target_ids
        )
        aligned_after, _ = shared_entity_alignment(
            repair.mapping, source_ids, target_ids
        )
        anchor_seconds = time.time() - anchor_start
        metrics = self.anchor_metrics
        metrics["pairs"] += 1
        metrics["shared_entities"] += shared
        metrics["aligned_before"] += aligned_before
        metrics["aligned_after"] += aligned_after
        metrics["anchors_before"] += initial_anchor_count
        metrics["anchors_after"] += repaired_anchor_count
        metrics["pairs_mapping_changed"] += int(repair.mapping != initial_mapping)
        metrics["pairs_anchor_improved"] += int(
            repaired_anchor_count > initial_anchor_count
        )
        metrics["pairs_alignment_improved"] += int(aligned_after > aligned_before)
        metrics["pairs_alignment_harmed"] += int(aligned_after < aligned_before)
        metrics["iterations"] += repair.iterations
        metrics["candidates_evaluated"] += repair.candidates_evaluated
        metrics["equal_cost_candidates"] += repair.equal_cost_candidates
        metrics["pairs_with_equal_cost_neighbor"] += int(
            repair.initial_equal_cost_neighbors > 0
        )
        metrics["pairs_with_anchor_improving_neighbor"] += int(
            repair.initial_improving_neighbors > 0
        )
        metrics["simple_cost_changed_pairs"] += int(
            repaired_cost[0] != initial_cost[0]
        )
        metrics["multirelation_cost_changed_pairs"] += int(
            repaired_cost[1] != initial_cost[1]
        )
        metrics["seconds"] += anchor_seconds

        repaired_mapping = torch.tensor(
            repair.mapping,
            dtype=structural_repair.mapping.dtype,
            device=structural_repair.mapping.device,
        )
        self.last_repair = replace(structural_repair, mapping=repaired_mapping)
        solution = torch.zeros(
            (len(repair.mapping), target_simple.num_nodes),
            dtype=torch.bool,
            device=self.device,
        )
        solution[
            torch.arange(len(repair.mapping), device=self.device),
            repaired_mapping,
        ] = True
        return final_ged, solution, inference_time + anchor_seconds

    def _anchor_summary(self):
        metrics = self.anchor_metrics
        pairs = metrics["pairs"]
        shared = metrics["shared_entities"]
        simple_unchanged = pairs - metrics["simple_cost_changed_pairs"]
        multi_unchanged = pairs - metrics["multirelation_cost_changed_pairs"]
        return {
            "revision": self.anchor_revision,
            "objective": "number of mapped node pairs with exactly equal embeddings",
            "candidate_moves": "matched 2-swap and unmatched-target replacement",
            "protected_costs": ["simple_graph", "multirelation"],
            "simple_cost_changed_pairs": metrics["simple_cost_changed_pairs"],
            "multirelation_cost_changed_pairs": metrics[
                "multirelation_cost_changed_pairs"
            ],
            "simple_cost_unchanged_rate": round(
                simple_unchanged / pairs if pairs else 0.0, 6
            ),
            "multirelation_cost_unchanged_rate": round(
                multi_unchanged / pairs if pairs else 0.0, 6
            ),
            "shared_entities": shared,
            "aligned_before": metrics["aligned_before"],
            "aligned_after": metrics["aligned_after"],
            "alignment_rate_before": round(
                metrics["aligned_before"] / shared if shared else 0.0, 6
            ),
            "alignment_rate_after": round(
                metrics["aligned_after"] / shared if shared else 0.0, 6
            ),
            "exact_anchors_before": metrics["anchors_before"],
            "exact_anchors_after": metrics["anchors_after"],
            "pairs_mapping_changed": metrics["pairs_mapping_changed"],
            "pairs_anchor_improved": metrics["pairs_anchor_improved"],
            "pairs_alignment_improved": metrics["pairs_alignment_improved"],
            "pairs_alignment_harmed": metrics["pairs_alignment_harmed"],
            "pairs_with_equal_cost_neighbor": metrics[
                "pairs_with_equal_cost_neighbor"
            ],
            "pairs_with_anchor_improving_neighbor": metrics[
                "pairs_with_anchor_improving_neighbor"
            ],
            "average_iterations": round(
                metrics["iterations"] / pairs if pairs else 0.0, 4
            ),
            "average_candidates_evaluated": round(
                metrics["candidates_evaluated"] / pairs if pairs else 0.0, 2
            ),
            "average_equal_cost_candidates": round(
                metrics["equal_cost_candidates"] / pairs if pairs else 0.0, 2
            ),
            "seconds_per_pair": round(
                metrics["seconds"] / pairs if pairs else 0.0, 6
            ),
            "entity_ids_used_for_selection": False,
            "entity_ids_used_for_diagnostic_only": True,
        }

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        self._reset_anchor_metrics()
        result = super().score(testing_graph_set, test_k, top_k_approach)
        result["frozen_inference_source"] = (
            "dataset-appropriate V0/V4 checkpoint + v1 structural repair + "
            + self.anchor_revision
        )
        result["v1_dense_reference"]["scope"] = (
            "pre-anchor structural mapping; executable paths use the "
            "dual-cost-equivalent exact-anchor mapping"
        )
        result["exact_embedding_anchor"] = self._anchor_summary()

        result_stem = (
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}"
            f"_k{test_k}_{self.args.repair_mode}"
        )
        result_path = self._result_file_path(result_stem)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("Recorded V8 exact-anchor metrics:", result_path)
        return result

