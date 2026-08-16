"""V19: matching-conditioned relation evidence inside the generator."""

import importlib.util
from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V18_DIR = CURRENT_DIR.parent / "v18_official_matched_edge"
for path in (PROJECT_ROOT, V18_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from src.GEDRanker.diffusion_schedulers import CategoricalDiffusion  # noqa: E402
from v19_models import (  # noqa: E402
    OfficialMatchedEdgeDiscriminator,
    RelationAwareDiffMatch,
    RelationGuidedDiffMatch,
)


def _load_v18_trainer_class():
    module_path = V18_DIR / "relation_trainer.py"
    spec = importlib.util.spec_from_file_location(
        "v18_official_trainer_for_v19", module_path
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load V18 trainer from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.V18OfficialMatchedEdgeTrainer


V18OfficialMatchedEdgeTrainer = _load_v18_trainer_class()


class V19GeneratorEdgeGuidanceTrainer(V18OfficialMatchedEdgeTrainer):
    version = "v19_generator_edge_guidance"
    v19_revision = "zero_gated_batched_generator_edge_evidence_v1"

    def __init__(self, args):
        self.v19_mode = getattr(args, "v19_mode", "generator_edge")
        if self.v19_mode not in {"baseline", "generator_edge"}:
            raise ValueError("V19_MODE must be baseline or generator_edge.")
        # Both modes deliberately keep the V18 discriminator. The only
        # controlled variable is the generator-side residual.
        args.v18_mode = "matched_edge"
        super().__init__(args)

    def setup_model(self):
        generator_cls = (
            RelationGuidedDiffMatch
            if self.v19_mode == "generator_edge"
            else RelationAwareDiffMatch
        )
        self.model = generator_cls(
            self.args, self.number_of_labels, self.relation_dim
        ).to(self.device)
        self.D = OfficialMatchedEdgeDiscriminator(
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
        self.diffusion = CategoricalDiffusion(T=self.args.diffusion_steps)

    def _v18_metadata(self):
        metadata = super()._v18_metadata()
        metadata.update(
            {
                "version": self.version,
                "v19_revision": self.v19_revision,
                "v19_mode": self.v19_mode,
                "v19_controlled_variable": "generator_edge_residual_only",
                "generator_edge_implementation": (
                    "cached_batched_index_add"
                    if self.v19_mode == "generator_edge"
                    else "disabled"
                ),
                "generator_edge_features": [
                    "current_mapping",
                    "exact_relation_support",
                    "wrong_relation_support",
                    "missing_topology_support",
                    "diffusion_timestep",
                ],
                "generator_edge_gate_init": float(
                    getattr(self.args, "v19_gate_init", 0.0)
                ),
                "runtime_abort_audits": False,
                "ground_truth_changed": False,
                "preference_definition_changed": False,
                "cost_mode": "unit",
                "ged_column": 3,
                "primary_metrics": ["mae", "acc"],
            }
        )
        if hasattr(self.model, "generator_edge_gate"):
            metadata.update(
                {
                    "generator_edge_gate_raw": float(
                        self.model.generator_edge_gate.detach().cpu()
                    ),
                    "generator_edge_gate_effective": float(
                        self.model.effective_generator_edge_gate.detach().cpu()
                    ),
                }
            )
        return metadata

