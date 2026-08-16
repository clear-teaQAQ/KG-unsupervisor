"""V14 trainer with unchanged unit-GED supervision and a new discriminator."""

from pathlib import Path
import importlib.util
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v14_models import (  # noqa: E402
    MatchingConditionedEdgeDiscriminator,
    RelationAwareDiffMatch,
    RelationAwareDiscriminator,
    attach_v14_edge_reasoning_cache,
)
from src.GEDRanker.loss_fn import mapping_loss, roll_out_gumbel  # noqa: E402


def _load_v11_trainer_class():
    module_path = V11_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v11_relation_aware_trainer_for_v14", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V11 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RelationAwareTrainer


V11RelationAwareTrainer = _load_v11_trainer_class()


class V14RelationAwareTrainer(V11RelationAwareTrainer):
    version = "v14_matching_conditioned_edge_reasoning"
    relation_revision = "v11_symmetric_gine_raw_relation_v1"
    v14_revision = "zero_gated_bidirectional_exact_edge_events_v1"

    def __init__(self, args):
        self.v14_mode = getattr(args, "v14_mode", "baseline")
        if self.v14_mode not in {"baseline", "matched_edge"}:
            raise ValueError("V14_MODE must be baseline or matched_edge.")
        self.preference_audit_history = []
        self._preference_audit_step = 0
        self.edge_cache_pairs = 0
        self.edge_cache_fallback_pairs = 0
        self.edge_cache_combinations = 0
        super().__init__(args)

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)
        if self.v14_mode == "matched_edge" and bool(
            getattr(self.args, "v14_edge_cache", False)
        ):
            cache = attach_v14_edge_reasoning_cache(data)
            self.edge_cache_pairs += 1
            self.edge_cache_fallback_pairs += int(not cache["valid"])
            self.edge_cache_combinations += cache["edge_combinations"]
        return data

    def setup_model(self):
        self.model = RelationAwareDiffMatch(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        discriminator_cls = (
            MatchingConditionedEdgeDiscriminator
            if self.v14_mode == "matched_edge"
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

    def fit(self):
        self._audit_correct = 0
        self._audit_total = 0
        self._audit_gap_one_correct = 0
        self._audit_gap_one_total = 0
        self._audit_gap_large_correct = 0
        self._audit_gap_large_total = 0
        super().fit()
        record = {
            "epoch": int(self.cur_epoch + 1),
            "best_vs_last_pref_acc": (
                self._audit_correct / self._audit_total
                if self._audit_total
                else None
            ),
            "strict_pairs": self._audit_total,
            "gap_one_pref_acc": (
                self._audit_gap_one_correct / self._audit_gap_one_total
                if self._audit_gap_one_total
                else None
            ),
            "gap_one_pairs": self._audit_gap_one_total,
            "gap_large_pref_acc": (
                self._audit_gap_large_correct / self._audit_gap_large_total
                if self._audit_gap_large_total
                else None
            ),
            "gap_large_pairs": self._audit_gap_large_total,
        }
        self.preference_audit_history.append(record)
        print("V14 preference audit:", json.dumps(record, sort_keys=True))

    def process_batch(self, batch, indices):
        exploration_epochs = self.args.model_epoch_end / 2
        use_fast_path = bool(getattr(self.args, "v14_fast_path", False))
        if self.cur_epoch < exploration_epochs or not use_fast_path:
            result = super().process_batch(batch, indices)
        else:
            result = self._process_exploitation_only_batch(batch, indices)

        self._preference_audit_step += 1
        interval = int(getattr(self.args, "v14_pref_audit_interval", 0))
        if interval <= 0:
            return result
        if self._preference_audit_step % interval != 0:
            return result

        strict = batch.best_ged < batch.last_ged
        if not strict.any():
            return result
        with torch.no_grad():
            best_score = self.D(batch, batch.best_mapping_label)
            last_score = self.D(batch, batch.last_mapping_label)
        correct = best_score > last_score
        gap = batch.last_ged - batch.best_ged
        self._audit_correct += int(correct[strict].sum().item())
        self._audit_total += int(strict.sum().item())
        gap_one = strict & (gap == 1)
        gap_large = strict & (gap > 1)
        self._audit_gap_one_correct += int(correct[gap_one].sum().item())
        self._audit_gap_one_total += int(gap_one.sum().item())
        self._audit_gap_large_correct += int(correct[gap_large].sum().item())
        self._audit_gap_large_total += int(gap_large.sum().item())
        return result

    def _process_exploitation_only_batch(self, batch, indices):
        """Preserve V14's alpha=0 objective without zero-weight D forwards."""
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
            {
                "state_dict": self.D.state_dict(),
                "v14_mode": self.v14_mode,
                "preference_audit": self.preference_audit_history,
            },
            self.discriminator_checkpoint_path,
        )

    def load_discriminator_checkpoint(self, checkpoint_path):
        payload = torch.load(checkpoint_path, map_location=self.device)
        state_dict = payload.get("state_dict", payload)
        saved_mode = payload.get("v14_mode") if isinstance(payload, dict) else None
        if saved_mode is not None and saved_mode != self.v14_mode:
            raise ValueError(
                f"Discriminator mode mismatch: saved={saved_mode}, current={self.v14_mode}."
            )
        self.D.load_state_dict(state_dict)
        if isinstance(payload, dict):
            self.preference_audit_history = payload.get("preference_audit", [])
        self.discriminator_checkpoint_path = Path(checkpoint_path).resolve()
        print("Loaded discriminator checkpoint:", self.discriminator_checkpoint_path)

    def _v14_metadata(self):
        metadata = {
            "v14_revision": self.v14_revision,
            "v14_mode": self.v14_mode,
            "preserves_cost_mode": "unit",
            "preference_label": "strict_original_unit_ged_ordering",
            "primary_metrics": ["mae", "acc"],
            "preference_audit": self.preference_audit_history,
            "preference_audit_interval": int(
                getattr(self.args, "v14_pref_audit_interval", 0)
            ),
            "edge_reasoning_implementation": (
                "cached_vectorized_raw_last_write"
                if bool(getattr(self.args, "v14_vectorized_edge", False))
                else (
                    "cached_pairwise_raw_last_write"
                    if bool(getattr(self.args, "v14_edge_cache", False))
                    else "legacy_pairwise_dense_adjacency"
                )
            ),
            "alpha_zero_discriminator_forward_skipped": bool(
                getattr(self.args, "v14_fast_path", False)
            ),
            "edge_cache_pairs": self.edge_cache_pairs,
            "edge_cache_fallback_pairs": self.edge_cache_fallback_pairs,
            "edge_cache_combinations": self.edge_cache_combinations,
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
                        getattr(self.args, "v14_gate_init", 0.0)
                    ),
                }
            )
        return metadata

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update(self._v14_metadata())
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("V14 metadata:", json.dumps(self._v14_metadata(), sort_keys=True))
        return payload
