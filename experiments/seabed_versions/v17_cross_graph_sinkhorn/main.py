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

from relation_trainer import V17CrossGraphSinkhornTrainer  # noqa: E402
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
    args.relation_mode = "raw"
    args.v15_mode = "projected_input"
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V17 preserves raw features, column-3 GED, and unit cost.")
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V17 training starts from epoch 0.")
    print(
        "V17 objective: replace the diffusion matcher with cross-graph GINE "
        "attention while preserving V16 official unit-GED and BPR supervision."
    )
    tab_printer(args)
    trainer = V17CrossGraphSinkhornTrainer(args)
    for epoch in range(args.model_epoch_end):
        trainer.cur_epoch = epoch
        trainer.fit()
        if (epoch + 1) % 100 == 0 or epoch + 1 == args.model_epoch_end:
            trainer.save(epoch + 1)
    result = trainer.score(args.testset, args.test_k, args.topk_approach)
    manifest = {
        "version": trainer.version,
        "v17_revision": trainer.v17_revision,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "cost_mode": "unit",
        "ged_column": 3,
        "ground_truth_changed": False,
        "preference_definition_changed": False,
        "result": result,
    }
    manifest_path = trainer.result_dir / (
        f"manifest_{args.dataset}_cross_graph_sinkhorn_epoch"
        f"{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print("Saved V17 manifest:", manifest_path)


if __name__ == "__main__":
    main()
