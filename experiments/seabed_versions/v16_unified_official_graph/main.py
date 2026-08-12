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

from relation_trainer import V16UnifiedOfficialGraphTrainer  # noqa: E402
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
    args.v15_mode = "projected_input"
    if args.relation_mode != "raw":
        raise ValueError("V16 requires raw relation features before projection.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V16 preserves raw features, column-3 GED, and unit cost.")
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V16 formal training starts from scratch at epoch 0.")

    print(
        "V16 objective: one official last-write graph for GINE, candidate GED, "
        "BPR preference, best mapping updates, inference selection, and evaluation."
    )
    tab_printer(args)
    trainer = V16UnifiedOfficialGraphTrainer(args)
    for epoch in range(args.model_epoch_end):
        trainer.cur_epoch = epoch
        trainer.fit()
        if (epoch + 1) % 100 == 0 or epoch + 1 == args.model_epoch_end:
            trainer.save(epoch + 1)

    result = trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )
    manifest_path = trainer.result_dir / (
        f"manifest_{args.dataset}_unified_official_epoch"
        f"{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "v16_revision": trainer.unified_revision,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "cost_mode": "unit",
        "ged_column": 3,
        "primary_metrics": ["mae", "acc"],
        "result": result,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("Saved V16 manifest:", manifest_path)


if __name__ == "__main__":
    main()

