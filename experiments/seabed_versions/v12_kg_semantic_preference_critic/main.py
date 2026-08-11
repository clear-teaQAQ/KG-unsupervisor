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
    args.result_path = str((CURRENT_DIR / "training_results").resolve())
    if not args.run_timestamp:
        args.run_timestamp = _default_timestamp()

    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V12 fixes use_raw_features=1, ged_column=3, and cost_mode=unit.")
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V12 trains from scratch with model_train=1 and epoch_start=0.")

    for subdir in (Path(args.abs_path) / args.model_path, Path(args.abs_path) / args.result_path):
        subdir.mkdir(parents=True, exist_ok=True)

    tab_printer(args)
    trainer = SemanticPreferenceTrainer(args)
    for epoch in range(args.model_epoch_end):
        trainer.cur_epoch = epoch
        trainer.fit()
    trainer.save(args.model_epoch_end)
    result = trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )

    manifest_path = (
        Path(args.abs_path)
        / args.result_path
        / f"manifest_{args.dataset}_{args.control_mode}_epoch{args.model_epoch_end}_{args.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "relation_revision": trainer.relation_revision,
        "control": trainer.control.__dict__,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": getattr(trainer, "saved_checkpoint_path", None),
        "result": result,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Saved V12 manifest:", manifest_path)


if __name__ == "__main__":
    main()
