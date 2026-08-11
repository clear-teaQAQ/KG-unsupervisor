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

from relation_trainer import V15ProjectedRelationTrainer  # noqa: E402
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
    args.v15_mode = os.environ.get("V15_MODE", "raw")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH to a V15 checkpoint.")
    if args.model_train != 0:
        raise ValueError("V15 checkpoint evaluation requires --model-train 0.")
    if args.v15_mode not in {"raw", "projected_input"}:
        raise ValueError("V15_MODE must be raw or projected_input.")
    if args.relation_mode != "raw":
        raise ValueError("V15 evaluation requires RELATION_MODE=raw.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V15 evaluation preserves V11 unit-cost column-3 GED.")

    print("V15 evaluation: unchanged V11 unit-GED with MAE/ACC as primary metrics.")
    tab_printer(args)
    trainer = V15ProjectedRelationTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()
