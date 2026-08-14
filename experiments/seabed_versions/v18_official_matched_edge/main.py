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

from relation_trainer import V18OfficialMatchedEdgeTrainer  # noqa: E402
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
    args.v18_mode = os.environ.get("V18_MODE", "matched_edge")
    args.v18_gate_init = float(os.environ.get("V18_GATE_INIT", "0.0"))
    args.v18_edge_hidden_dim = int(os.environ.get("V18_EDGE_HIDDEN_DIM", "32"))
    if args.v18_mode not in {"baseline", "matched_edge"}:
        raise ValueError("V18_MODE must be baseline or matched_edge.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError("V18 requires raw features and unchanged column-3 unit GED.")
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V18 formal training starts from epoch 0.")

    print(
        "V18 objective: V16 official graph plus cost-preserving, "
        "matching-conditioned exact-edge preference evidence."
    )
    print(
        "V18 runtime policy: no data-dependent abort audits; cached batched edge "
        "reasoning; skip the zero-weight discriminator pass after alpha reaches 0."
    )
    tab_printer(args)
    trainer = V18OfficialMatchedEdgeTrainer(args)
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
        f"manifest_{args.dataset}_{args.v18_mode}_epoch"
        f"{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "v18_revision": trainer.v18_revision,
        "v18_mode": args.v18_mode,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "discriminator_checkpoint_path": str(
            getattr(trainer, "discriminator_checkpoint_path", "")
        ),
        "cost_mode": "unit",
        "ged_column": 3,
        "primary_metrics": ["mae", "acc"],
        "result": result,
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
        handle.write("\n")
    print("Saved V18 manifest:", manifest_path)


if __name__ == "__main__":
    main()

