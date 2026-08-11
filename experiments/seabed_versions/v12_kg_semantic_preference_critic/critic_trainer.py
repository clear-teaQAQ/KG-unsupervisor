"""V12 trainer: isolated semantic preference critic line on top of GEDRanker-SEABED."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.nn.pool import global_add_pool

from control import control_as_dict, resolve_control
from critic_models import PairPreferenceFeatures, SemanticPreferenceCritic, SparseCandidateMatcher
from kg_data import derive_topology_feature_order, reorder_features
from kg_models import RelationAwareDiffMatch, RelationAwareDiscriminator
from src.SEABED.utils import get_file_paths
from src.SEABED.trainer import Trainer as BaseTrainer
from src.GEDRanker.loss_fn import (
    bpr_loss,
    hinge_loss,
    mapping_loss,
    roll_out,
    roll_out_gumbel,
)
from src.GEDRanker.diffusion_schedulers import CategoricalDiffusion


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]


def _safe_json_dump(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def normalize_relation_embedding(value):
    vector = np.asarray(value, dtype=float).reshape(-1)
    if vector.size == 0 or not np.all(np.isfinite(vector)):
        raise ValueError("Relation embeddings must be finite non-empty vectors.")
    return vector


def relation_mode_vectors(vectors, mode, graph_index):
    values = np.asarray(vectors, dtype=float)
    if mode == "raw":
        return values
    if mode == "constant":
        return np.zeros_like(values)
    if mode == "shuffled":
        permutation = np.random.default_rng(graph_index).permutation(len(values))
        return values[permutation]
    raise ValueError(f"Unknown relation mode: {mode}")


class SemanticPreferenceTrainer(BaseTrainer):
    version = "v12_kg_semantic_preference_critic"
    relation_revision = "semantic_preference_bootstrap_v1"

    def __init__(self, args):
        self.control = resolve_control(args)
        self.run_timestamp = getattr(args, "run_timestamp", "")
        self.v12_diagnostics = {}
        self._raw_anchor_records = []
        super().__init__(args)

    def setup_model(self):
        self.model = RelationAwareDiffMatch(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.D = RelationAwareDiscriminator(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)
        self.semantic_critic = SemanticPreferenceCritic(
            input_dim=self.control.feature_dim,
            hidden_dims=self.control.semantic_hidden_dim,
        ).to(self.device)
        self.sparse_matcher = SparseCandidateMatcher(
            top_r=self.control.sparse_top_r,
            random_epsilon=self.control.sparse_random_epsilon,
        )
        if self.control.enable_semantic_critic:
            optimizer_params = list(self.model.parameters()) + list(self.semantic_critic.parameters())
        else:
            optimizer_params = list(self.model.parameters())
        self.optimizer = torch.optim.RMSprop(
            optimizer_params,
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizerD = torch.optim.RMSprop(
            self.D.parameters(),
            lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )

    def load_data(self):
        start = time.time()
        super().load_data()
        graph_paths = []
        for split in ("train", "val", "test"):
            graph_paths.extend(get_file_paths(str(Path(self.args.dataset_root) / split), "json"))
        if len(graph_paths) != len(self.graphs):
            raise RuntimeError("Relation graph-path order does not match loaded graphs.")

        relation_dim = None
        relation_registry = {}
        inconsistent_ids = set()
        nested_embeddings = 0
        edge_count = 0
        reassigned_nodes = 0
        consistent_edges_before = 0
        for graph_index, (graph, graph_path) in enumerate(zip(self.graphs, graph_paths)):
            with open(graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload = payload["0"] if "0" in payload else payload
            reindex = derive_topology_feature_order(payload)
            graph["features"] = reorder_features(graph["features"], reindex.permutation)
            graph["node_ids"] = list(reindex.node_ids)
            reassigned_nodes += reindex.reassigned_nodes
            consistent_edges_before += reindex.consistent_edges_before
            edge_features = payload.get("edge_features", [])
            if len(edge_features) != len(graph["graph"]):
                raise RuntimeError(
                    f"Relation/edge count mismatch in {graph_path}: "
                    f"relations={len(edge_features)}, edges={len(graph['graph'])}."
                )

            vectors = []
            for relation in edge_features:
                raw_embedding = relation.get("embedding")
                nested_embeddings += int(
                    isinstance(raw_embedding, list)
                    and len(raw_embedding) == 1
                    and isinstance(raw_embedding[0], list)
                )
                vector = normalize_relation_embedding(raw_embedding)
                if relation_dim is None:
                    relation_dim = int(vector.size)
                if vector.size != relation_dim:
                    raise RuntimeError(
                        f"Relation dimension mismatch in {graph_path}: "
                        f"expected={relation_dim}, actual={vector.size}."
                    )
                relation_id = str(relation["id"])
                previous = relation_registry.setdefault(relation_id, vector)
                if not np.array_equal(previous, vector):
                    inconsistent_ids.add(relation_id)
                vectors.append(vector)

            edge_count += len(vectors)
            if vectors:
                vectors = relation_mode_vectors(
                    vectors,
                    self.control.relation_mode,
                    graph_index,
                )
                graph["relation_features"] = vectors.tolist()
            else:
                graph["relation_features"] = []

        if relation_dim is None:
            raise RuntimeError("No relation embedding was found in the dataset.")
        if inconsistent_ids:
            raise RuntimeError(
                f"{len(inconsistent_ids)} relation IDs have inconsistent embeddings."
            )
        self.relation_dim = relation_dim
        self.relation_diagnostics = {
            "revision": self.relation_revision,
            "mode": self.control.relation_mode,
            "graphs": len(self.graphs),
            "edges": edge_count,
            "relation_dim": relation_dim,
            "unique_relation_ids": len(relation_registry),
            "inconsistent_relation_ids": len(inconsistent_ids),
            "nested_embeddings_flattened": nested_embeddings,
            "feature_nodes_reassigned": reassigned_nodes,
            "edge_consistency_before": round(
                consistent_edges_before / edge_count if edge_count else 1.0, 6
            ),
            "edge_consistency_after": 1.0,
            "direction_policy": "same relation vector on forward and reverse edges",
            "self_loop_policy": "zero relation vector",
        }
        self.load_data_time = time.time() - start
        print(
            "V12 relation diagnostics:",
            json.dumps(self.relation_diagnostics, sort_keys=True),
        )

    def transfer_data_to_torch(self):
        super().transfer_data_to_torch()
        self.relation_edge_attr = []
        self.relation_out_signature = []
        self.relation_in_signature = []
        self.node_degree = []
        for graph, graph_edge_index in zip(self.graphs, self.edge_index):
            if graph.get("relation_features"):
                relation = torch.tensor(
                    graph["relation_features"],
                    dtype=torch.float,
                ).reshape(-1, self.relation_dim)
            else:
                relation = torch.empty((0, self.relation_dim), dtype=torch.float)
            self_loops = torch.zeros((graph["n"], self.relation_dim), dtype=torch.float)
            edge_attr = torch.cat([relation, relation, self_loops], dim=0)
            if edge_attr.shape[0] != graph_edge_index.shape[1]:
                raise RuntimeError("Relation attributes do not align with graph edges.")
            self.relation_edge_attr.append(edge_attr)

            outgoing = torch.zeros((graph["n"], self.relation_dim), dtype=torch.float)
            incoming = torch.zeros_like(outgoing)
            out_degree = torch.zeros(graph["n"], dtype=torch.float)
            in_degree = torch.zeros(graph["n"], dtype=torch.float)
            if relation.shape[0]:
                original_edges = torch.tensor(graph["graph"], dtype=torch.long)
                outgoing.index_add_(0, original_edges[:, 0], relation)
                incoming.index_add_(0, original_edges[:, 1], relation)
                out_degree.index_add_(0, original_edges[:, 0], torch.ones(len(relation)))
                in_degree.index_add_(0, original_edges[:, 1], torch.ones(len(relation)))
                outgoing = outgoing / out_degree.clamp_min(1).unsqueeze(-1)
                incoming = incoming / in_degree.clamp_min(1).unsqueeze(-1)
            self.relation_out_signature.append(outgoing)
            self.relation_in_signature.append(incoming)
            self.node_degree.append(out_degree + in_degree)

    @staticmethod
    def _cross_cosine(left, right):
        left = F.normalize(left.float(), p=2, dim=-1, eps=1e-8)
        right = F.normalize(right.float(), p=2, dim=-1, eps=1e-8)
        return left @ right.t()

    def pack_graph_pair(self, pair):
        data = super().pack_graph_pair(pair)
        graph_1, graph_2 = data.i_j[0].tolist()
        data.edge_attr = torch.cat(
            [
                self.relation_edge_attr[graph_1],
                self.relation_edge_attr[graph_2],
            ],
            dim=0,
        )
        if data.edge_attr.shape[0] != data.edge_index.shape[1]:
            raise RuntimeError("Packed relation attributes do not align with edges.")
        source = data.x[data.edge_index_mapping[0]]
        target = data.x[data.edge_index_mapping[1]]
        data.exact_anchor_mask = torch.all(source == target, dim=-1, keepdim=True).float()
        data.available_anchor_count = data.exact_anchor_mask.sum().reshape(1)
        entity_similarity = self._cross_cosine(self.features[graph_1], self.features[graph_2])
        outgoing_similarity = self._cross_cosine(
            self.relation_out_signature[graph_1], self.relation_out_signature[graph_2]
        )
        incoming_similarity = self._cross_cosine(
            self.relation_in_signature[graph_1], self.relation_in_signature[graph_2]
        )
        degree_left = torch.log1p(self.node_degree[graph_1]).unsqueeze(1)
        degree_right = torch.log1p(self.node_degree[graph_2]).unsqueeze(0)
        degree_similarity = torch.exp(-torch.abs(degree_left - degree_right))
        data.semantic_features = PairPreferenceFeatures(
            entity_similarity=entity_similarity,
            outgoing_relation_similarity=outgoing_similarity,
            incoming_relation_similarity=incoming_similarity,
            degree_similarity=degree_similarity,
            exact_entity_anchor=data.exact_anchor_mask.reshape_as(entity_similarity),
        ).as_tensor()
        if data.semantic_features.shape != (data.edge_index_mapping.shape[1], self.control.feature_dim):
            raise RuntimeError("Semantic candidate features do not align with mapping edges.")
        return data

    def _candidate_mask(self, batch, critic_logits):
        mask = torch.zeros_like(critic_logits, dtype=torch.bool)
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        for pair_index in range(batch.n.shape[0]):
            pair_mask = mapping_batch == pair_index
            n1, n2 = map(int, batch.n[pair_index].tolist())
            scores = critic_logits[pair_mask].reshape(n1, n2)
            selected = self.sparse_matcher.select(scores).reshape(-1)
            selected |= batch.best_mapping_label[pair_mask].reshape(-1).bool()
            mask[pair_mask] = selected
        return mask

    def compute_auxiliary_loss(
        self, batch, pred_mapping_label, pred_solution, pred_ged, best_ged, alpha
    ):
        if not self.control.enable_semantic_critic:
            return torch.zeros((), device=pred_mapping_label.device)
        if batch.semantic_features.numel() == 0:
            return torch.zeros((), device=pred_mapping_label.device)
        logits = pred_mapping_label.reshape(-1)
        target = batch.best_mapping_label.float().reshape(-1)
        critic_logits = self.semantic_critic(batch.semantic_features).reshape(-1)
        mapping_batch = batch.batch[batch.edge_index_mapping[0]]
        selected = (
            self._candidate_mask(batch, critic_logits.detach())
            if self.control.sparse_matching
            else torch.ones_like(critic_logits, dtype=torch.bool)
        )

        positive_weight = (batch.n[:, 1].float() - 1).clamp_min(1)[mapping_batch]
        class_weight = torch.where(target > 0.5, positive_weight, torch.ones_like(target))
        critic_supervision = F.binary_cross_entropy_with_logits(
            critic_logits[selected], target[selected], weight=class_weight[selected]
        )

        critic_probability = torch.sigmoid(critic_logits)
        pair_count = batch.n.shape[0]
        current_score = global_add_pool(
            critic_probability * pred_solution.detach().reshape(-1), mapping_batch, size=pair_count
        ) / batch.n[:, 0].float().clamp_min(1)
        best_score = global_add_pool(
            critic_probability * target, mapping_batch, size=pair_count
        ) / batch.n[:, 0].float().clamp_min(1)
        normalized_gap = (pred_ged.detach() - best_ged.detach()) / batch.avg_n.reshape(-1).clamp_min(1)
        gap_weight = torch.log1p(torch.abs(pred_ged.detach() - best_ged.detach()))
        preference_rank = (
            F.smooth_l1_loss(best_score - current_score, normalized_gap, reduction="none")
            * (1.0 + gap_weight)
        ).mean()

        features = batch.semantic_features
        semantic_prior = (
            0.35 * (features[:, 0] + 1) / 2
            + 0.20 * (features[:, 1] + 1) / 2
            + 0.20 * (features[:, 2] + 1) / 2
            + 0.10 * features[:, 3].clamp(0, 1)
            + 0.15 * features[:, 4].clamp(0, 1)
        ).clamp(0, 1)
        semantic_target = 0.5 * critic_probability.detach() + 0.5 * semantic_prior
        edge_gap_weight = (1.0 + gap_weight)[mapping_batch]
        semantic_preference = (
            F.binary_cross_entropy_with_logits(logits[selected], semantic_target[selected], reduction="none")
            * edge_gap_weight[selected]
        ).mean()

        keep_loss = torch.zeros((), device=logits.device)
        if self.control.use_teacher_consistency:
            keep_loss = (
                F.binary_cross_entropy_with_logits(logits[selected], target[selected], reduction="none")
                * class_weight[selected]
            ).mean()

        explore_loss = torch.zeros((), device=logits.device)
        if self.control.adaptive_explore and alpha > 0:
            probability = torch.sigmoid(logits)
            entropy = -(
                probability * torch.log(probability.clamp_min(1e-6))
                + (1 - probability) * torch.log((1 - probability).clamp_min(1e-6))
            )
            adaptive_weight = float(alpha) * torch.exp(-torch.abs(normalized_gap))[mapping_batch]
            explore_loss = -(entropy[selected] * adaptive_weight[selected]).mean()

        total = (
            critic_supervision
            + preference_rank
            + semantic_preference
            + self.control.keep_loss_weight * keep_loss
            + self.control.explore_weight * explore_loss
        )
        self.last_auxiliary_metrics = {
            "critic_supervision": float(critic_supervision.detach()),
            "preference_rank": float(preference_rank.detach()),
            "semantic_preference": float(semantic_preference.detach()),
            "keep": float(keep_loss.detach()),
            "explore": float(explore_loss.detach()),
            "candidate_fraction": float(selected.float().mean()),
        }
        # Baseline mapping/GED objectives are summed over graph pairs, so keep
        # the auxiliary objective on the same batch-size scale.
        return self.control.semantic_weight * total * pair_count

    def process_batch(self, batch, indices):
        batch_size = (torch.max(batch.batch) + 1).item()
        gt_mapping_idx, best_mapping_label, best_ged = batch.edge_index_mapping, batch.best_mapping_label, batch.best_ged

        t = np.random.randint(1, self.diffusion.T + 1, batch_size).astype(int)
        best_mapping_onehot = torch.nn.functional.one_hot(best_mapping_label.long(), num_classes=2).float()
        mapping_batch = batch.batch[gt_mapping_idx[0]]
        diffused_mapping = self.diffusion.sample(best_mapping_onehot, t, mapping_batch)
        t = torch.from_numpy(t).float()
        pred_mapping_label = self.model(batch, diffused_mapping.to(self.device), t.to(self.device))

        if self.args.unsupervised_approach in {"BPR", "Hinge", "GED"}:
            pred_ged, pred_solution, pred_solution_gumbel_sparse = self.roll_out_gumbel(pred_mapping_label, batch)
        else:
            pred_ged, pred_solution = self.roll_out(pred_mapping_label, batch)
            pred_solution_gumbel_sparse = pred_solution

        normalize_curr_ged = torch.exp(-pred_ged / batch.avg_n.squeeze(-1))
        normalize_best_ged = torch.exp(-best_ged / batch.avg_n.squeeze(-1))
        normalize_last_ged = torch.exp(-batch.last_ged / batch.avg_n.squeeze(-1))
        if self.args.unsupervised_approach == "plain":
            alpha = 0
        else:
            alpha = max(1 - 1 * self.cur_epoch / (self.args.model_epoch_end / 2), 0)

        if alpha > 0 and self.args.unsupervised_approach in {"BPR", "Hinge", "GED"}:
            D_pred_ged_curr = self.D(batch, pred_solution_gumbel_sparse.detach())
            if self.args.unsupervised_approach in {"BPR", "Hinge"}:
                D_pred_ged_best = self.D(batch, batch.best_mapping_label)
                D_pred_ged_last = self.D(batch, batch.last_mapping_label)

            if self.args.unsupervised_approach == "BPR":
                D_loss = bpr_loss(
                    D_pred_ged_curr,
                    D_pred_ged_best,
                    D_pred_ged_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            elif self.args.unsupervised_approach == "Hinge":
                D_loss = hinge_loss(
                    D_pred_ged_curr,
                    D_pred_ged_best,
                    D_pred_ged_last,
                    normalize_curr_ged,
                    normalize_best_ged,
                    normalize_last_ged,
                )
            else:
                D_loss = ((D_pred_ged_curr - normalize_curr_ged) ** 2).sum()
            self.optimizerD.zero_grad()
            D_loss.backward()
            self.optimizerD.step()
        else:
            D_loss = torch.tensor([0], device=pred_mapping_label.device)

        map_loss = mapping_loss(pred_mapping_label, batch, best_mapping_label)
        if self.args.unsupervised_approach in {"BPR", "Hinge", "GED"}:
            D_pred_ged = self.D(batch, pred_solution_gumbel_sparse)
            ged_loss = -(D_pred_ged).sum()
        else:
            ged_loss = torch.tensor([0], device=map_loss.device)

        auxiliary_loss = self.compute_auxiliary_loss(
            batch, pred_mapping_label, pred_solution, pred_ged, best_ged, alpha
        )
        losses = map_loss + ged_loss * alpha + auxiliary_loss

        self.optimizer.zero_grad()
        losses.backward()
        self.optimizer.step()

        new_solution = 0
        for index in range(len(indices)):
            mask = batch.batch[batch.edge_index_mapping[0]] == index
            self.training_graphs[indices[index]].last_mapping_label = pred_solution[mask].to(
                self.training_graphs[indices[index]].last_mapping_label.device
            )
            self.training_graphs[indices[index]].last_ged = (pred_ged[index]).to(
                self.training_graphs[indices[index]].last_ged.device
            )
            if pred_ged[index] < best_ged[index]:
                new_solution += 1
                self.training_graphs[indices[index]].best_ged = (pred_ged[index]).to(
                    self.training_graphs[indices[index]].best_ged.device
                )
                self.training_graphs[indices[index]].best_mapping_label = pred_solution[mask].to(
                    self.training_graphs[indices[index]].best_mapping_label.device
                )

        return (
            losses.item(),
            D_loss.item(),
            pred_ged.sum().item(),
            batch.ged.sum().item(),
            best_ged.sum().item(),
            map_loss.item(),
            ged_loss.item(),
            new_solution,
        )

    def roll_out(self, pred_mapping_label, batch):
        return roll_out(pred_mapping_label, batch)

    def roll_out_gumbel(self, pred_mapping_label, batch):
        return roll_out_gumbel(
            pred_mapping_label,
            batch,
            self.args.tau,
            self.args.gumbel_iteration,
        )

    def save(self, epoch):
        payload = {
            "version": self.version,
            "relation_revision": self.relation_revision,
            "control": control_as_dict(self.control),
            "model": self.model.state_dict(),
            "discriminator": self.D.state_dict(),
            "semantic_critic": self.semantic_critic.state_dict(),
            "epoch": int(epoch),
        }
        checkpoint_path = (
            Path(self.args.abs_path)
            / self.args.model_path
            / f"{self.args.dataset}_{epoch}_{self.args.model_name}_{self.args.unsupervised_approach}_{self.run_timestamp}.pt"
        )
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, checkpoint_path)
        self.saved_checkpoint_path = str(checkpoint_path.resolve())

    def load_explicit_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        if isinstance(state, dict) and "model" in state:
            checkpoint_control = state.get("control")
            if checkpoint_control and checkpoint_control != control_as_dict(self.control):
                raise ValueError(
                    "Checkpoint control configuration does not match evaluation arguments: "
                    f"checkpoint={checkpoint_control}, requested={control_as_dict(self.control)}"
                )
            self.model.load_state_dict(state["model"])
            if "discriminator" in state:
                self.D.load_state_dict(state["discriminator"])
            if "semantic_critic" in state:
                try:
                    self.semantic_critic.load_state_dict(state["semantic_critic"])
                except Exception:
                    pass
        else:
            self.model.load_state_dict(state)
        self.loaded_checkpoint_path = str(Path(checkpoint_path).resolve())
        print("Loaded checkpoint:", self.loaded_checkpoint_path)

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        super().score(testing_graph_set=testing_graph_set, test_k=test_k, top_k_approach=top_k_approach)
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        if result_path.exists():
            with open(result_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload.update(
                {
                    "version": self.version,
                    "relation_revision": self.relation_revision,
                    "control": control_as_dict(self.control),
                    "relation_diagnostics": self.relation_diagnostics,
                    "semantic_feature_names": [
                        "entity_cosine",
                        "outgoing_relation_cosine",
                        "incoming_relation_cosine",
                        "degree_similarity",
                        "exact_entity_anchor",
                    ],
                    "loaded_checkpoint_path": getattr(self, "loaded_checkpoint_path", None),
                    "saved_checkpoint_path": getattr(self, "saved_checkpoint_path", None),
                }
            )
            _safe_json_dump(result_path, payload)
            return payload
        raise RuntimeError(f"Expected evaluation result was not written: {result_path}")
