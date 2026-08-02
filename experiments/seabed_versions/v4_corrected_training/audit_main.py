from pathlib import Path
import os
import random
import sys

import numpy as np
import torch


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parents[2]
V3_DIR = CURRENT_DIR.parent / "v3_topology_feature_reindex"
for path in (PROJECT_ROOT, V3_DIR, CURRENT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from corrected_audit_trainer import CorrectedTrainingAuditTrainer
from experiments.seabed_versions.v3_topology_feature_reindex.param_parser import (
    parameter_parser,
)


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
    args.checkpoint_path = str(Path(args.checkpoint_path).resolve())
    args.result_path = "experiments/seabed_versions/v4_corrected_training/audit_results"
    trainer = CorrectedTrainingAuditTrainer(args)
    trainer.load_explicit_checkpoint(args.checkpoint_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()
