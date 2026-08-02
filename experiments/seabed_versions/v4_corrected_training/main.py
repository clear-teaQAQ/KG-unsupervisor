from pathlib import Path
import json
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

from corrected_training_trainer import CorrectedTrainingTrainer
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
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError(
            "V4 fixes use_raw_features=1, ged_column=3, and cost_mode=unit."
        )
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V4 trains from scratch with --model-train 1 --model-epoch-start 0.")
    if args.model_epoch_end <= 0:
        raise ValueError("--model-epoch-end must be positive.")

    tab_printer(args)
    trainer = CorrectedTrainingTrainer(args)
    for epoch in range(args.model_epoch_end):
        trainer.cur_epoch = epoch
        trainer.fit()
    trainer.save(args.model_epoch_end)
    result = trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )

    manifest_path = trainer.result_dir / (
        f"manifest_{args.dataset}_epoch{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "result": result,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Saved V4 manifest:", manifest_path)


if __name__ == "__main__":
    main()
