"""V13 trainer: preserve V11 and optionally add a gated exact-anchor residual."""

from pathlib import Path
import importlib.util
import json
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from v13_models import (  # noqa: E402
    GatedAnchorResidualDiffMatch,
    RelationAwareDiffMatch,
    RelationAwareDiscriminator,
)


def _load_v11_trainer_class():
    module_path = V11_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v11_relation_aware_trainer_for_v13",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V11 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.RelationAwareTrainer


V11RelationAwareTrainer = _load_v11_trainer_class()


class V13RelationAwareTrainer(V11RelationAwareTrainer):
    version = "v13_v11_preserving_gated_anchor_residual"
    relation_revision = "v11_symmetric_gine_raw_relation_v1"
    v13_revision = "gated_exact_anchor_residual_v1"

    def __init__(self, args):
        self.v13_mode = getattr(args, "v13_mode", "baseline")
        if self.v13_mode not in {"baseline", "gated_anchor"}:
            raise ValueError("V13_MODE must be baseline or gated_anchor.")
        super().__init__(args)

    def setup_model(self):
        model_cls = (
            GatedAnchorResidualDiffMatch
            if self.v13_mode == "gated_anchor"
            else RelationAwareDiffMatch
        )
        self.model = model_cls(
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

    def _v13_metadata(self):
        metadata = {
            "v13_revision": self.v13_revision,
            "v13_mode": self.v13_mode,
            "preserves_cost_mode": "unit",
            "primary_metrics": ["mae", "acc"],
        }
        if hasattr(self.model, "v13_anchor_gate"):
            metadata["anchor_gate"] = float(self.model.v13_anchor_gate.detach().cpu())
            metadata["anchor_gate_init"] = float(
                getattr(self.args, "v13_anchor_gate_init", 0.0)
            )
        return metadata

    def score(self, testing_graph_set="test", test_k=100, top_k_approach="parallel"):
        payload = super().score(testing_graph_set, test_k, top_k_approach)
        payload.update(self._v13_metadata())
        result_path = self._result_file_path(
            self._result_stem("result_SEABED", testing_graph_set)
        )
        with open(result_path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print("V13 metadata:", json.dumps(self._v13_metadata(), sort_keys=True))
        return payload
