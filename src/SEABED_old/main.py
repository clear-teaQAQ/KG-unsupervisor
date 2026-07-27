from pathlib import Path
import os
import random
import numpy as np
import torch
import sys

CURRENT_DIR = Path(__file__).resolve().parent
if str(CURRENT_DIR) not in sys.path:
    sys.path.insert(0, str(CURRENT_DIR))

from param_parser import parameter_parser
from trainer import Trainer
from utils import tab_printer


def seed_everything(torch_seed):
    random.seed(torch_seed)
    os.environ["PYTHONHASHSEED"] = str(torch_seed)
    np.random.seed(torch_seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    seed_everything(0)
    args = parameter_parser()

    if args.dataset_root is not None:
        args.dataset_root = str(Path(args.dataset_root).resolve())

    tab_printer(args)
    trainer = Trainer(args)

    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)

    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            trainer.cur_epoch = epoch
            trainer.fit()
        trainer.save(args.model_epoch_end)
        trainer.score(testing_graph_set="test", test_k=args.test_k, top_k_approach="parallel")
    else:
        trainer.score(
            testing_graph_set=args.testset,
            test_k=args.test_k,
            top_k_approach=args.topk_approach,
        )


if __name__ == "__main__":
    main()
