"""V16: one official last-write edge view for representation and unit GED."""

import importlib.util
import json
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V15_DIR = CURRENT_DIR.parent / "v15_benchmark_projected_relation_input"
for path in (PROJECT_ROOT, V15_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def _load_v15_module():
    module_path = V15_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v15_relation_trainer_for_v16", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V15 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


V15_MODULE = _load_v15_module()
V15ProjectedRelationTrainer = V15_MODULE.V15ProjectedRelationTrainer


class V16UnifiedOfficialGraphTrainer(V15ProjectedRelationTrainer):
    version = "v16_unified_official_graph"
    unified_revision = "undirected_last_raw_write_all_pipeline_v1"

    def __init__(self, args):
        if getattr(args, "v15_mode", "projected_input") != "projected_input":
            raise ValueError("V16 requires the projected_input model edge view.")
        super().__init__(args)

    def load_data(self):
        super().load_data()
        self.projection_diagnostics.update(
            {
                "revision": self.unified_revision,
                "mode": "unified_official_graph",
                "model_edge_view": "undirected_simple_last_write",
                "unit_ged_edge_view": "undirected_simple_last_write",
                "preference_edge_view": "undirected_simple_last_write",
                "inference_selection_edge_view": "undirected_simple_last_write",
            }
        )
        print(
            "V16 unified projection diagnostics:",
            json.dumps(self.projection_diagnostics, sort_keys=True),
        )

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)
        if self.v15_mode != "projected_input":
            raise RuntimeError("V16 packed a non-projected model view.")

        # This is the single V16 change: every unit-cost consumer uses the
        # exact same projected tensors as the GINE representation.
        data.unit_cost_edge_index = data.edge_index.clone()
        data.unit_cost_edge_labels = data.edge_labels.clone()
        data.m = data.v15_projected_edge_counts.clone()
        n1, n2 = [int(value) for value in data.n[0].tolist()]
        m1, m2 = [int(value) for value in data.m[0].tolist()]
        data.higher_bound = torch.tensor(
            [[max(n1, n2) + max(m1, m2)]], dtype=torch.long
        )
        self._assert_unified_pair(data)
        return data

    @staticmethod
    def _assert_unified_pair(data):
        if not torch.equal(data.edge_index, data.unit_cost_edge_index):
            raise RuntimeError("V16 model and unit-cost edge indices diverged.")
        if not torch.equal(data.edge_labels, data.unit_cost_edge_labels):
            raise RuntimeError("V16 model and unit-cost relation labels diverged.")

        n1, n2 = [int(value) for value in data.n[0].tolist()]
        for graph_index, (offset, node_count) in enumerate(((0, n1), (n1, n2))):
            edge_mask = (
                data.x_indicator[data.edge_index[0]].squeeze(1) == graph_index
            ) & (data.edge_labels > 0)
            edges = data.edge_index[:, edge_mask] - offset
            labels = data.edge_labels[edge_mask]
            directed = {}
            for edge, label in zip(edges.t().tolist(), labels.tolist()):
                source, target = edge
                key = (source, target)
                if key in directed:
                    raise RuntimeError(
                        f"V16 projected graph contains duplicate directed edge {key}."
                    )
                directed[key] = label
            for (source, target), label in directed.items():
                if source == target:
                    continue
                if directed.get((target, source)) != label:
                    raise RuntimeError(
                        "V16 symmetric GINE directions do not share a relation label."
                    )
            if edges.numel() and (edges.min() < 0 or edges.max() >= node_count):
                raise RuntimeError("V16 projected edge is outside its graph.")

    @staticmethod
    def _assert_not_below_ground_truth(costs, batch, source):
        ground_truth = batch.ged.reshape(-1).to(costs.device).float()
        costs = costs.reshape(-1).float()
        if costs.shape != ground_truth.shape:
            raise RuntimeError(
                f"V16 {source} cost/GT shapes differ: "
                f"{tuple(costs.shape)} versus {tuple(ground_truth.shape)}"
            )
        invalid = costs < ground_truth
        if invalid.any():
            indices = torch.nonzero(invalid).reshape(-1).tolist()
            raise RuntimeError(
                "V16 official-compatible candidate cost fell below column-3 "
                f"ground truth in {source}: indices={indices}, "
                f"costs={costs[invalid].tolist()}, gt={ground_truth[invalid].tolist()}"
            )

    def _compute_batch_ged(self, solution_sparse, batch):
        costs = super()._compute_batch_ged(solution_sparse, batch)
        self._assert_not_below_ground_truth(costs, batch, "batch")
        return costs

    def _compute_single_ged_from_dense_solution(self, solution, data):
        cost = super()._compute_single_ged_from_dense_solution(solution, data)
        ground_truth = float(data.ged.item())
        if cost < ground_truth:
            raise RuntimeError(
                "V16 official-compatible inference cost fell below column-3 "
                f"ground truth: cost={cost}, gt={ground_truth}, "
                f"graph_ids={data.i_j[0].tolist()}"
            )
        return cost

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        if payload["fea"] != 1.0:
            raise RuntimeError(f"V16 executable unit-cost FEA must be 1, got {payload['fea']}.")
        payload.update(
            {
                "version": self.version,
                "v16_revision": self.unified_revision,
                "projection_revision": self.unified_revision,
                "pipeline_edge_view": "undirected_simple_last_write",
                "unit_ged_edge_view": "undirected_simple_last_write",
                "preference_edge_view": "undirected_simple_last_write",
                "inference_selection_edge_view": "undirected_simple_last_write",
                "preserves_cost_mode": "unit",
                "preserves_ged_column": 3,
                "ground_truth_changed": False,
                "preference_definition_changed": False,
                "primary_metrics": ["mae", "acc"],
            }
        )
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print("V16 metadata:", json.dumps(self.projection_diagnostics, sort_keys=True))
        return payload
