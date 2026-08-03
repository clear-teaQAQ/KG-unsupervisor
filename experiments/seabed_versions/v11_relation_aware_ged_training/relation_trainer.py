"""V11 trainer: unchanged GED-only learning with relation-aware graph encoding."""

import json
from pathlib import Path
import sys
import time

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V4_DIR = CURRENT_DIR.parent / "v4_corrected_training"
for path in (PROJECT_ROOT, V4_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from corrected_training_trainer import CorrectedTrainingTrainer
from relation_models import RelationAwareDiffMatch, RelationAwareDiscriminator
from src.SEABED.utils import get_file_paths


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


class RelationAwareTrainer(CorrectedTrainingTrainer):
    version = "v11_relation_aware_ged_training"
    relation_revision = "symmetric_gine_raw_relation_v1"

    def __init__(self, args):
        self._raw_anchor_records = []
        super().__init__(args)

    def setup_model(self):
        self.model = RelationAwareDiffMatch(
            self.args,
            self.number_of_labels,
            self.relation_dim,
        ).to(self.device)
        self.D = RelationAwareDiscriminator(
            self.args,
            self.number_of_labels,
            self.relation_dim,
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

    def load_data(self):
        start = time.time()
        super().load_data()
        graph_paths = []
        for split in ("train", "val", "test"):
            graph_paths.extend(
                get_file_paths(str(Path(self.args.dataset_root) / split), "json")
            )
        if len(graph_paths) != len(self.graphs):
            raise RuntimeError("Relation graph-path order does not match loaded graphs.")

        relation_dim = None
        relation_registry = {}
        inconsistent_ids = set()
        nested_embeddings = 0
        edge_count = 0
        for graph_index, (graph, graph_path) in enumerate(zip(self.graphs, graph_paths)):
            with open(graph_path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            payload = payload["0"] if "0" in payload else payload
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
                    self.args.relation_mode,
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
            "mode": self.args.relation_mode,
            "graphs": len(self.graphs),
            "edges": edge_count,
            "relation_dim": relation_dim,
            "unique_relation_ids": len(relation_registry),
            "inconsistent_relation_ids": len(inconsistent_ids),
            "nested_embeddings_flattened": nested_embeddings,
            "direction_policy": "same relation vector on forward and reverse edges",
            "self_loop_policy": "zero relation vector",
        }
        self.load_data_time = time.time() - start
        print(
            "Relation diagnostics:",
            json.dumps(self.relation_diagnostics, sort_keys=True),
        )

    def transfer_data_to_torch(self):
        super().transfer_data_to_torch()
        self.relation_edge_attr = []
        for graph, graph_edge_index in zip(self.graphs, self.edge_index):
            if graph["relation_features"]:
                relation = torch.tensor(
                    graph["relation_features"],
                    dtype=torch.float,
                ).reshape(-1, self.relation_dim)
            else:
                relation = torch.empty((0, self.relation_dim), dtype=torch.float)
            self_loops = torch.zeros(
                (graph["n"], self.relation_dim),
                dtype=torch.float,
            )
            edge_attr = torch.cat([relation, relation, self_loops], dim=0)
            if edge_attr.shape[0] != graph_edge_index.shape[1]:
                raise RuntimeError("Relation attributes do not align with graph edges.")
            self.relation_edge_attr.append(edge_attr)

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
        data.exact_anchor_mask = torch.all(
            source == target,
            dim=-1,
            keepdim=True,
        ).float()
        data.available_anchor_count = data.exact_anchor_mask.sum().reshape(1)
        return data

    def diffusion_ged_parallel(self, batch, test_k=100):
        ged, solution, running_time = super().diffusion_ged_parallel(batch, test_k)
        n1 = int(batch.n[0, 0].item())
        n2 = int(batch.n[0, 1].item())
        anchors = batch.exact_anchor_mask.reshape(n1, n2).bool()
        self._raw_anchor_records.append(
            {
                "selected": int((solution[:n1].bool() & anchors).sum().item()),
                "available": int(batch.available_anchor_count.item()),
                "mapped_nodes": n1,
            }
        )
        return ged, solution, running_time

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        if top_k_approach != "parallel":
            raise ValueError("V11 raw correspondence requires parallel inference.")
        self._raw_anchor_records = []
        result = super().score(testing_graph_set, test_k, top_k_approach)
        selected = sum(item["selected"] for item in self._raw_anchor_records)
        available = sum(item["available"] for item in self._raw_anchor_records)
        mapped_nodes = sum(item["mapped_nodes"] for item in self._raw_anchor_records)
        covered = [item for item in self._raw_anchor_records if item["available"]]
        raw_correspondence = {
            "selection": "first minimum-GED sample from unchanged best-of-k inference",
            "postprocessing": "none",
            "test_k": test_k,
            "pairs": len(self._raw_anchor_records),
            "pairs_with_exact_anchors": len(covered),
            "pairs_with_perfect_anchor_recall": sum(
                item["selected"] == item["available"] for item in covered
            ),
            "selected_exact_anchors": selected,
            "available_exact_anchors": available,
            "exact_anchor_recall": selected / available if available else None,
            "exact_anchor_rate_over_mapped_nodes": (
                selected / mapped_nodes if mapped_nodes else None
            ),
        }
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with open(result_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        payload.update(
            {
                "version": self.version,
                "relation_revision": self.relation_revision,
                "relation_diagnostics": self.relation_diagnostics,
                "loaded_checkpoint_path": getattr(
                    self,
                    "loaded_checkpoint_path",
                    None,
                ),
                "raw_correspondence": raw_correspondence,
            }
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("V11 raw correspondence:", json.dumps(raw_correspondence, sort_keys=True))
        return payload

    def load_explicit_checkpoint(self, checkpoint_path):
        state = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.loaded_checkpoint_path = str(Path(checkpoint_path).resolve())
        print("Loaded checkpoint:", self.loaded_checkpoint_path)
