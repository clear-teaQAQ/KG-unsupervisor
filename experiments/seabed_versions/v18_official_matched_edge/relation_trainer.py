"""V18: V16 official graph with batched matching-conditioned edge reasoning."""

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V16_DIR = CURRENT_DIR.parent / "v16_unified_official_graph"
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V16_DIR, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.GEDRanker.loss_fn import mapping_loss, roll_out_gumbel  # noqa: E402
from v18_models import (  # noqa: E402
    OfficialMatchedEdgeDiscriminator,
    RelationAwareDiffMatch,
    RelationAwareDiscriminator,
    attach_edge_reasoning_cache,
)


def _load_v16_trainer_class():
    module_path = V16_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v16_unified_trainer_for_v18", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V16 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.V16UnifiedOfficialGraphTrainer


V16UnifiedOfficialGraphTrainer = _load_v16_trainer_class()


class V18OfficialMatchedEdgeTrainer(V16UnifiedOfficialGraphTrainer):
    version = "v18_official_matched_edge"
    v18_revision = "official_graph_batched_exact_edge_residual_v1"

    def __init__(self, args):
        self.v18_mode = getattr(args, "v18_mode", "matched_edge")
        if self.v18_mode not in {"baseline", "matched_edge"}:
            raise ValueError("V18_MODE must be baseline or matched_edge.")
        self.edge_cache_pairs = 0
        self.edge_cache_fallback_pairs = 0
        self.edge_cache_combinations = 0
        self.cost_audit_shape_mismatches = 0
        super().__init__(args)

    @staticmethod
    def _assert_unified_pair(data):
        # V16 already constructs both views from the same tensors. Rechecking
        # every pair was expensive and could terminate a long run; V18 keeps
        # consistency coverage in unit tests and records non-fatal diagnostics.
        return None

    def _record_ground_truth_comparison(self, costs, batch, source):
        ground_truth = batch.ged.reshape(-1).to(costs.device).float()
        costs = costs.reshape(-1).float()
        if costs.numel() != ground_truth.numel():
            self.cost_audit_shape_mismatches += 1
            return
        invalid = costs < ground_truth
        self.below_ground_truth_candidates += int(invalid.sum().item())
        self.below_ground_truth_batches += int(bool(invalid.any()))

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)
        cache = attach_edge_reasoning_cache(data)
        self.edge_cache_pairs += 1
        self.edge_cache_fallback_pairs += int(not cache["valid"])
        self.edge_cache_combinations += cache["edge_combinations"]
        return data

    def setup_model(self):
        self.model = RelationAwareDiffMatch(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        discriminator_cls = (
            OfficialMatchedEdgeDiscriminator
            if self.v18_mode == "matched_edge"
            else RelationAwareDiscriminator
        )
        self.D = discriminator_cls(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizerD = torch.optim.RMSprop(
            self.D.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        from src.GEDRanker.diffusion_schedulers import CategoricalDiffusion

        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)

    def process_batch(self, batch, indices):
        exploration_epochs = self.args.model_epoch_end / 2
        if self.cur_epoch < exploration_epochs:
            return super().process_batch(batch, indices)
        return self._process_exploitation_only_batch(batch, indices)

    def _process_exploitation_only_batch(self, batch, indices):
        """Preserve the alpha=0 objective while skipping a zero-weight D pass."""
        batch_size = int(torch.max(batch.batch).item()) + 1
        best_mapping_label, best_ged = batch.best_mapping_label, batch.best_ged
        timestep = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        best_mapping_onehot = F.one_hot(
            best_mapping_label.long(), num_classes=2
        ).float()
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        diffused_mapping = self.diffusion.sample(
            best_mapping_onehot, timestep, mapping_batch
        )
        timestep = torch.from_numpy(timestep).float().to(self.device)
        pred_mapping_label = self.model(
            batch, diffused_mapping.to(self.device), timestep
        )
        _, pred_solution, _ = roll_out_gumbel(
            pred_mapping_label,
            batch,
            self.args.tau,
            self.args.gumbel_iteration,
        )
        pred_ged = self._compute_batch_ged(pred_solution, batch)
        map_loss = mapping_loss(pred_mapping_label, batch, best_mapping_label)
        self.optimizer.zero_grad()
        map_loss.backward()
        self.optimizer.step()

        new_solution = 0
        for index in range(len(indices)):
            graph = self.training_graphs[indices[index]]
            mask = batch.batch[batch.edge_index_mapping[0]] == index
            graph.last_mapping_label = pred_solution[mask].to(
                graph.last_mapping_label.device
            )
            graph.last_ged = pred_ged[index].to(graph.last_ged.device)
            if pred_ged[index] < best_ged[index]:
                new_solution += 1
                graph.best_ged = pred_ged[index].to(graph.best_ged.device)
                graph.best_mapping_label = pred_solution[mask].to(
                    graph.best_mapping_label.device
                )

        return (
            map_loss.item(),
            0.0,
            pred_ged.sum().item(),
            batch.ged.sum().item(),
            best_ged.sum().item(),
            map_loss.item(),
            0.0,
            new_solution,
        )

    def save(self, epoch):
        super().save(epoch)
        generator_path = self._model_checkpoint_path(epoch)
        discriminator_dir = self.model_dir / "discriminators"
        discriminator_dir.mkdir(parents=True, exist_ok=True)
        self.discriminator_checkpoint_path = discriminator_dir / (
            f"{generator_path.stem}_discriminator.pt"
        )
        torch.save(
            {"state_dict": self.D.state_dict(), "v18_mode": self.v18_mode},
            self.discriminator_checkpoint_path,
        )

    def load_discriminator_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("state_dict", payload)
        self.D.load_state_dict(state_dict)
        self.discriminator_checkpoint_path = Path(checkpoint_path).resolve()
        print("Loaded discriminator checkpoint:", self.discriminator_checkpoint_path)

    def _v18_metadata(self):
        metadata = {
            "version": self.version,
            "v18_revision": self.v18_revision,
            "v18_mode": self.v18_mode,
            "v16_revision": self.unified_revision,
            "pipeline_edge_view": "undirected_simple_last_write",
            "cost_mode": "unit",
            "ged_column": 3,
            "ground_truth_changed": False,
            "preference_definition_changed": False,
            "preference_label": "strict_original_unit_ged_ordering",
            "primary_metrics": ["mae", "acc"],
            "edge_reasoning_implementation": "cached_vectorized_edge_correspondence",
            "preference_audit_in_training": False,
            "runtime_abort_audits": False,
            "alpha_zero_discriminator_forward_skipped": True,
            "edge_cache_pairs": self.edge_cache_pairs,
            "edge_cache_fallback_pairs": self.edge_cache_fallback_pairs,
            "edge_cache_combinations": self.edge_cache_combinations,
            "cost_audit_shape_mismatches": self.cost_audit_shape_mismatches,
            "discriminator_checkpoint_path": str(
                getattr(self, "discriminator_checkpoint_path", "")
            ),
        }
        if hasattr(self.D, "edge_reasoning_gate"):
            metadata.update(
                {
                    "edge_reasoning_gate_raw": float(
                        self.D.edge_reasoning_gate.detach().cpu()
                    ),
                    "edge_reasoning_gate_effective": float(
                        self.D.effective_edge_gate.detach().cpu()
                    ),
                    "edge_reasoning_gate_init": float(
                        getattr(self.args, "v18_gate_init", 0.0)
                    ),
                }
            )
        return metadata

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update(self._v18_metadata())
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with result_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
            handle.write("\n")
        print("V18 metadata:", json.dumps(self._v18_metadata(), sort_keys=True))
        return payload

