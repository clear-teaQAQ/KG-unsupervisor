"""V17 trainer: V16 official graph semantics with a cross-graph matcher."""

import importlib.util
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V16_DIR = CURRENT_DIR.parent / "v16_unified_official_graph"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))


def _load_v16():
    path = V16_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location("v16_for_v17", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V16 trainer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.V16UnifiedOfficialGraphTrainer


V16Trainer = _load_v16()
from v17_models import CrossGraphSinkhornMatcher  # noqa: E402


class V17CrossGraphSinkhornTrainer(V16Trainer):
    version = "v17_cross_graph_sinkhorn"
    v17_revision = "cross_graph_gine_attention_sinkhorn_logits_v1"

    def setup_model(self):
        self.model = CrossGraphSinkhornMatcher(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        # Keep the ranking critic unchanged so the only model change is the
        # candidate-generation architecture.
        from experiments.seabed_versions.v11_relation_aware_ged_training.relation_models import (
            RelationAwareDiscriminator,
        )
        self.D = RelationAwareDiscriminator(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.optimizer = torch.optim.RMSprop(
            self.model.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        self.optimizerD = torch.optim.RMSprop(
            self.D.parameters(), lr=self.args.learning_rate,
            weight_decay=self.args.weight_decay,
        )
        from src.GEDRanker.diffusion_schedulers import CategoricalDiffusion
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update({
            "version": self.version,
            "v17_revision": self.v17_revision,
            "generator_architecture": "cross_graph_gine_attention",
            "assignment_relaxation": "existing_gumbel_sinkhorn_rollout",
            "cost_mode": "unit",
            "ged_column": 3,
            "ground_truth_changed": False,
            "preference_definition_changed": False,
        })
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        result_path.write_text(__import__("json").dumps(payload, indent=2) + "\n", encoding="utf-8")
        return payload
