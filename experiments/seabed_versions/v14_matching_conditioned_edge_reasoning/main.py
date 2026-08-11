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
    if args.v14_mode not in {"baseline", "matched_edge"}:
        raise ValueError("V14_MODE must be baseline or matched_edge.")
    if args.relation_mode not in {"raw", "constant", "shuffled"}:
        raise ValueError("RELATION_MODE must be raw, constant, or shuffled.")
    if args.use_raw_features != 1 or args.ged_column != 3 or args.cost_mode != "unit":
        raise ValueError(
            "V14 preserves V11: use_raw_features=1, ged_column=3, cost_mode=unit."
        )
    if args.model_train != 1 or args.model_epoch_start != 0:
        raise ValueError("V14 trains from scratch with model_train=1 and epoch_start=0.")

    print(
        "V14 objective: change candidate evidence, not unit-GED cost or BPR labels; "
        "V11 MAE/ACC remain the hard baseline."
    )
    tab_printer(args)
    trainer = V14RelationAwareTrainer(args)
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
        f"manifest_{args.dataset}_{args.v14_mode}_{args.relation_mode}_epoch"
        f"{args.model_epoch_end}_{trainer.run_timestamp}.json"
    )
    manifest = {
        "version": trainer.version,
        "v14_revision": trainer.v14_revision,
        "v14_mode": args.v14_mode,
        "relation_revision": trainer.relation_revision,
        "relation_mode": args.relation_mode,
        "dataset": args.dataset,
        "epochs": args.model_epoch_end,
        "checkpoint_path": str(trainer.saved_checkpoint_path),
        "discriminator_checkpoint_path": str(
            getattr(trainer, "discriminator_checkpoint_path", "")
        ),
        "primary_metrics": ["mae", "acc"],
        "result": result,
    }
    with open(manifest_path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    print("Saved V14 manifest:", manifest_path)


if __name__ == "__main__":
    main()
