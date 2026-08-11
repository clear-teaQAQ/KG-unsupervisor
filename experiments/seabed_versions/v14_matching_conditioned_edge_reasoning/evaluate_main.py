from pathlib import Path
import os
import random
import sys

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from relation_trainer import V14RelationAwareTrainer  # noqa: E402
from src.SEABED.param_parser import parameter_parser  # noqa: E402
from src.SEABED.utils import tab_printer  # noqa: E402


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(int(os.environ.get("SEED", "0")))
    args = parameter_parser()
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.relation_mode = os.environ.get("RELATION_MODE", "raw")
    args.v14_mode = os.environ.get("V14_MODE", "baseline")
    args.v14_gate_init = float(os.environ.get("V14_GATE_INIT", "0.0"))
    args.v14_edge_hidden_dim = int(os.environ.get("V14_EDGE_HIDDEN_DIM", "32"))
    args.v14_pref_audit_interval = int(
        os.environ.get("V14_PREF_AUDIT_INTERVAL", "20")
    )
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH to a V14 generator checkpoint.")
    if args.model_train != 0:
        raise ValueError("V14 checkpoint evaluation requires --model-train 0.")
    if args.v14_mode not in {"baseline", "matched_edge"}:
        raise ValueError("V14_MODE must be baseline or matched_edge.")
    if args.relation_mode not in {"raw", "constant", "shuffled"}:
        raise ValueError("RELATION_MODE must be raw, constant, or shuffled.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError(
            "V14 evaluation preserves V11 unit-cost column-3 GED."
        )

    print(
        "V14 evaluation: unchanged unit-cost GED with MAE/ACC as primary metrics."
    )
    tab_printer(args)
    trainer = V14RelationAwareTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    discriminator_checkpoint_path = os.environ.get(
        "DISCRIMINATOR_CHECKPOINT_PATH"
    )
    if discriminator_checkpoint_path:
        trainer.load_discriminator_checkpoint(discriminator_checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()
