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

from relation_trainer import RelationAwareTrainer
from src.SEABED.param_parser import parameter_parser
from src.SEABED.utils import tab_printer


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(0)
    args = parameter_parser()
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.relation_mode = os.environ.get("RELATION_MODE", "raw")
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH to a V11 checkpoint.")
    if args.model_train != 0:
        raise ValueError("V11 checkpoint evaluation requires --model-train 0.")
    if args.relation_mode not in {"raw", "constant", "shuffled"}:
        raise ValueError("RELATION_MODE must be raw, constant, or shuffled.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError(
            "V11 evaluation fixes use_raw_features=1, ged_column=3, cost_mode=unit."
        )

    tab_printer(args)
    trainer = RelationAwareTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()
