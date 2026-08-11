"""V13 model additions that preserve the V11 relation-aware backbone."""

from pathlib import Path
import sys

import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V11_DIR = CURRENT_DIR.parent / "v11_relation_aware_ged_training"
for path in (PROJECT_ROOT, V11_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_models import (  # noqa: E402
    RelationAwareDiffMatch as V11RelationAwareDiffMatch,
    RelationAwareDiscriminator,
)


class GatedAnchorResidualDiffMatch(V11RelationAwareDiffMatch):
    """V11 DiffMatch plus a unit-cost-aligned exact-anchor residual.

    The residual is deliberately tiny in scope. It only sees
    ``data.exact_anchor_mask``, which is already produced by the V11 trainer and
    corresponds to exact raw-feature equality. This is the semantic signal most
    directly aligned with the original unit substitution cost.

    With ``anchor_gate_init=0.0`` the first forward pass is exactly V11:

    ``output = v11_output + 0 * exact_anchor_mask``.
    """

    def __init__(self, args, number_of_labels, relation_dim):
        super().__init__(args, number_of_labels, relation_dim)
        gate_init = float(getattr(args, "v13_anchor_gate_init", 0.0))
        self.v13_anchor_gate = torch.nn.Parameter(
            torch.tensor(gate_init, dtype=torch.float)
        )

    def forward(self, data, noise_mapping_attr, timestep):
        logits = super().forward(data, noise_mapping_attr, timestep)
        exact_anchor_mask = getattr(data, "exact_anchor_mask", None)
        if exact_anchor_mask is None:
            return logits
        return logits + self.v13_anchor_gate * exact_anchor_mask.to(
            device=logits.device,
            dtype=logits.dtype,
        )


RelationAwareDiffMatch = V11RelationAwareDiffMatch

