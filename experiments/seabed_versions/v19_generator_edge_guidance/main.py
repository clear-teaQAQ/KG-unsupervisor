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

from relation_trainer import V19GeneratorEdgeGuidanceTrainer  # noqa: E402
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


def configure_v19(args):
    args.dataset_root = str(Path(args.dataset_root).resolve())
    args.relation_mode = "raw"
    args.v15_mode = "projected_input"
    args.v18_mode = "matched_edge"
    args.v18_gate_init = float(os.environ.get("V18_GATE_INIT", "0.0"))
    args.v18_edge_hidden_dim = int(os.environ.get("V18_EDGE_HIDDEN_DIM", "32"))
    args.v19_mode = os.environ.get("V19_MODE", "generator_edge")
    args.v19_gate_init = float(os.environ.get("V19_GATE_INIT", "0.0"))
    args.v19_edge_hidden_dim = int(os.environ.get("V19_EDGE_HIDDEN_DIM", "32"))
    if args.v19_mode not in {"baseline", "generator_edge"}:
        raise ValueError("V19_MODE must be baseline or generator_edge.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V19 requires raw features and unchanged column-3 unit GED.")
    return args


def main():
    seed_everything(int(os.environ.get("SEED", "0")))
    args = configure_v19(parameter_parser())
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V19 formal training starts from epoch 0.")

    print(
        "V19 objective: unchanged column-3 unit GED and BPR preference; "
        "relation evidence changes only generator candidate production."
    )
    print(
        "V19 runtime policy: cached batched index_add evidence and no "
        "data-dependent abort audits."
    )
    tab_printer(args)
    trainer = V19GeneratorEdgeGuidanceTrainer(args)
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
        f"manifest_{args.dataset}_{args.v19_mode}_epoch"
        f"{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "v19_revision": trainer.v19_revision,
        "v19_mode": args.v19_mode,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "discriminator_checkpoint_path": str(
            getattr(trainer, "discriminator_checkpoint_path", "")
        ),
        "cost_mode": "unit",
        "ged_column": 3,
        "ground_truth_changed": False,
        "preference_definition_changed": False,
        "primary_metrics": ["mae", "acc"],
        "result": result,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("Saved V19 manifest:", manifest_path)


if __name__ == "__main__":
    main()

