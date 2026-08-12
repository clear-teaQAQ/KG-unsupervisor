"""Evaluate a frozen V15 checkpoint and save same-mapping cost audits."""

from pathlib import Path
import os
import sys


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from evaluate_main import seed_everything  # noqa: E402
from frozen_mapping_audit import V15FrozenMappingAuditTrainer  # noqa: E402
from src.SEABED.param_parser import parameter_parser  # noqa: E402
from src.SEABED.utils import tab_printer  # noqa: E402


def main():
    seed = int(os.environ.get("SEED", "0"))
    seed_everything(seed)
    args = parameter_parser()
    args.audit_seed = seed
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.relation_mode = os.environ.get("RELATION_MODE", "raw")
    args.v15_mode = os.environ.get("V15_MODE", "projected_input")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH to the frozen V15 checkpoint.")
    if args.model_train != 0:
        raise ValueError("Frozen mapping audit requires --model-train 0.")
    if args.dataset != "SWDF" or args.v15_mode != "projected_input":
        raise ValueError("This audit fixes SWDF and V15_MODE=projected_input.")
    if args.relation_mode != "raw":
        raise ValueError("This audit requires RELATION_MODE=raw.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("Audit preserves V11 raw-feature, column-3 unit-cost GED.")

    print(
        "Frozen V15 audit: unchanged legacy candidate selection plus "
        "same-mapping simple last-write path replay."
    )
    tab_printer(args)
    trainer = V15FrozenMappingAuditTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()

