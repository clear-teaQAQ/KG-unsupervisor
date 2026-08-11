from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
for path in (PROJECT_ROOT, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from param_parser import parameter_parser
from critic_trainer import SemanticPreferenceTrainer
from src.SEABED.utils import tab_printer


def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _default_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def main():
    args = parameter_parser()
    seed_everything(args.seed)
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.abs_path = str(CURRENT_DIR.resolve()) + "/"
    args.model_path = str((CURRENT_DIR / "checkpoints").resolve())
    args.result_path = str((CURRENT_DIR / "raw_eval_results").resolve())
    if not args.run_timestamp:
        args.run_timestamp = _default_timestamp()

    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    if not checkpoint_path:
        raise ValueError("Set CHECKPOINT_PATH to a V12 checkpoint.")
    if args.model_train != 0:
        raise ValueError("V12 checkpoint evaluation requires --model-train 0.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V12 evaluation fixes use_raw_features=1, ged_column=3, cost_mode=unit.")

    for subdir in (Path(args.abs_path) / args.model_path, Path(args.abs_path) / args.result_path):
        subdir.mkdir(parents=True, exist_ok=True)

    tab_printer(args)
    trainer = SemanticPreferenceTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()
