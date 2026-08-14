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
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    discriminator_path = os.environ.get("DISCRIMINATOR_CHECKPOINT_PATH")
    if not checkpoint_path or not discriminator_path:
        raise ValueError("Set both V18 generator and discriminator checkpoints.")
    if args.model_train != 0:
        raise ValueError("V18 checkpoint evaluation requires --model-train 0.")

    print("V18 evaluation: frozen official-graph generator and discriminator.")
    tab_printer(args)
    trainer = V18OfficialMatchedEdgeTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.load_discriminator_checkpoint(discriminator_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()

