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

from main import configure_v19  # noqa: E402
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


def main():
    seed_everything(int(os.environ.get("SEED", "0")))
    args = configure_v19(parameter_parser())
    checkpoint_path = os.environ.get("CHECKPOINT_PATH")
    discriminator_path = os.environ.get("DISCRIMINATOR_CHECKPOINT_PATH")
    if not checkpoint_path or not discriminator_path:
        raise ValueError("Set both V19 generator and discriminator checkpoints.")
    if args.model_train != 0:
        raise ValueError("V19 checkpoint evaluation requires --model-train 0.")

    print("V19 evaluation: frozen generator and V18 matched-edge discriminator.")
    tab_printer(args)
    trainer = V19GeneratorEdgeGuidanceTrainer(args)
    trainer.load_explicit_checkpoint(checkpoint_path)
    trainer.load_discriminator_checkpoint(discriminator_path)
    trainer.score(
        testing_graph_set=args.testset,
        test_k=args.test_k,
        top_k_approach=args.topk_approach,
    )


if __name__ == "__main__":
    main()

