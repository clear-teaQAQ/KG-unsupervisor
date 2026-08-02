"""V6 inference: improve cosine semantics while preserving both path costs."""

from dataclasses import replace
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V2_DIR = CURRENT_DIR.parent / "v2_edit_path_audit"
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
V5_DIR = CURRENT_DIR.parent / "v5_tie_space_audit"
for path in (PROJECT_ROOT, V2_DIR, V3_DIR, V5_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from path_evaluator import build_multirelation_edit_path, build_simple_edit_path
from reindexed_trainer import TopologyFeatureReindexTrainer
from semantic_tie import (
    cosine_similarity_matrix,
    matrix_mapping_score,
    optimize_equal_cost_mapping,
    shared_entity_alignment,
)


class SemanticTieBreakTrainer(TopologyFeatureReindexTrainer):
    version = "v6_semantic_tie_break"
    semantic_revision = "dual_cost_cosine_local_v1"

    def _reset_semantic_metrics(self):
        self.semantic_metrics = {
            "pairs": 0,
            "shared_entities": 0,
            "aligned_before": 0,
            "aligned_after": 0,
            "pairs_mapping_changed": 0,
            "pairs_alignment_improved": 0,
            "pairs_alignment_harmed": 0,
            "iterations": 0,
            "candidates_evaluated": 0,
            "equal_cost_candidates": 0,
            "pairs_with_equal_cost_neighbor": 0,
            "pairs_with_cosine_improving_neighbor": 0,
            "cosine_gain": 0.0,
            "seconds": 0.0,
        }

    def diffusion_ged_parallel(self, batch, test_k=100):
        final_ged, _, inference_time = super().diffusion_ged_parallel(batch, test_k)
        semantic_start = time.time()
        structural_repair = self.last_repair
        initial_mapping = tuple(structural_repair.mapping.detach().cpu().tolist())
        source_gid = int(batch.i_j[0][0].item())
        target_gid = int(batch.i_j[0][1].item())
        source_simple, source_multi = self._graph_views(source_gid)
        target_simple, target_multi = self._graph_views(target_gid)
        source_ids = self._node_ids(source_gid)
        target_ids = self._node_ids(target_gid)
        similarity = cosine_similarity_matrix(
            self.graphs[source_gid]["features"],
            self.graphs[target_gid]["features"],
        )
        cost_cache = {}

        def dual_cost(mapping):
            mapping = tuple(mapping)
            if mapping not in cost_cache:
                cost_cache[mapping] = (
                    build_simple_edit_path(
                        mapping, source_simple, target_simple
                    )["total_cost"],
                    build_multirelation_edit_path(
                        mapping, source_multi, target_multi
                    )["total_cost"],
                )
            return cost_cache[mapping]

        def cosine_score(mapping):
            return matrix_mapping_score(mapping, similarity)

        initial_cost = dual_cost(initial_mapping)
        semantic = optimize_equal_cost_mapping(
            initial_mapping,
            target_simple.num_nodes,
            dual_cost,
            cosine_score,
            max_iterations=self.args.semantic_max_iterations,
        )
        if dual_cost(semantic.mapping) != initial_cost:
            raise RuntimeError(
                f"V6 changed a protected path cost for pair {self.current_pair_index}."
            )

        aligned_before, shared = shared_entity_alignment(
            initial_mapping, source_ids, target_ids
        )
        aligned_after, _ = shared_entity_alignment(
            semantic.mapping, source_ids, target_ids
        )
        semantic_seconds = time.time() - semantic_start
        metrics = self.semantic_metrics
        metrics["pairs"] += 1
        metrics["shared_entities"] += shared
        metrics["aligned_before"] += aligned_before
        metrics["aligned_after"] += aligned_after
        metrics["pairs_mapping_changed"] += int(semantic.mapping != initial_mapping)
        metrics["pairs_alignment_improved"] += int(aligned_after > aligned_before)
        metrics["pairs_alignment_harmed"] += int(aligned_after < aligned_before)
        metrics["iterations"] += semantic.iterations
        metrics["candidates_evaluated"] += semantic.candidates_evaluated
        metrics["equal_cost_candidates"] += semantic.equal_cost_candidates
        metrics["pairs_with_equal_cost_neighbor"] += int(
            semantic.initial_equal_cost_neighbors > 0
        )
        metrics["pairs_with_cosine_improving_neighbor"] += int(
            semantic.initial_improving_neighbors > 0
        )
        metrics["cosine_gain"] += semantic.final_score - semantic.initial_score
        metrics["seconds"] += semantic_seconds

        semantic_mapping = torch.tensor(
            semantic.mapping,
            dtype=structural_repair.mapping.dtype,
            device=structural_repair.mapping.device,
        )
        self.last_repair = replace(structural_repair, mapping=semantic_mapping)
        solution = torch.zeros(
            (len(semantic.mapping), target_simple.num_nodes),
            dtype=torch.bool,
            device=self.device,
        )
        solution[
            torch.arange(len(semantic.mapping), device=self.device),
            semantic_mapping,
        ] = True
        return final_ged, solution, inference_time + semantic_seconds

    def _semantic_summary(self):
        metrics = self.semantic_metrics
        pairs = metrics["pairs"]
        shared = metrics["shared_entities"]
        return {
            "revision": self.semantic_revision,
            "objective": "sum of cross-graph node-embedding cosine similarities",
            "candidate_moves": "matched 2-swap and unmatched-target replacement",
            "protected_costs": ["simple_graph", "multirelation"],
            "simple_cost_unchanged_rate": 1.0,
            "multirelation_cost_unchanged_rate": 1.0,
            "shared_entities": shared,
            "aligned_before": metrics["aligned_before"],
            "aligned_after": metrics["aligned_after"],
            "alignment_rate_before": round(
                metrics["aligned_before"] / shared if shared else 0.0, 4
            ),
            "alignment_rate_after": round(
                metrics["aligned_after"] / shared if shared else 0.0, 4
            ),
            "pairs_mapping_changed": metrics["pairs_mapping_changed"],
            "pairs_alignment_improved": metrics["pairs_alignment_improved"],
            "pairs_alignment_harmed": metrics["pairs_alignment_harmed"],
            "pairs_with_equal_cost_neighbor": metrics["pairs_with_equal_cost_neighbor"],
            "pairs_with_cosine_improving_neighbor": metrics[
                "pairs_with_cosine_improving_neighbor"
            ],
            "average_iterations": round(metrics["iterations"] / pairs if pairs else 0.0, 4),
            "average_candidates_evaluated": round(
                metrics["candidates_evaluated"] / pairs if pairs else 0.0, 2
            ),
            "average_equal_cost_candidates": round(
                metrics["equal_cost_candidates"] / pairs if pairs else 0.0, 2
            ),
            "average_cosine_gain": round(
                metrics["cosine_gain"] / pairs if pairs else 0.0, 6
            ),
            "seconds_per_pair": round(metrics["seconds"] / pairs if pairs else 0.0, 6),
            "entity_ids_used_for_selection": False,
            "entity_ids_used_for_diagnostic_only": True,
        }

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        self._reset_semantic_metrics()
        result = super().score(testing_graph_set, test_k, top_k_approach)
        result["frozen_inference_source"] = (
            "dataset-appropriate V0/V4 checkpoint + v1 structural repair + "
            + self.semantic_revision
        )
        result["v1_dense_reference"]["scope"] = (
            "pre-semantic structural mapping; executable paths use the "
            "dual-cost-equivalent semantic mapping"
        )
        result["semantic_tie_break"] = self._semantic_summary()

        result_stem = (
            f"result_SEABED_{self.version}_{self.args.dataset}_{testing_graph_set}"
            f"_k{test_k}_{self.args.repair_mode}"
        )
        result_path = self._result_file_path(result_stem)
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(result, handle, indent=2)
        print("Recorded V6 semantic tie-break metrics:", result_path)
        return result
