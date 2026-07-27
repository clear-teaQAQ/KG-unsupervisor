from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import os
import numpy as np
import torch
import sys
def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    seed_everything(0)
    args = parameter_parser()
    tab_printer(args)
    trainer = Trainer(args)
    if args.model_name == "GEDGW":
        trainer.score('test',test_k=100)
        
    else:
        if args.model_epoch_start > 0:
            trainer.load(args.model_epoch_start)
        if args.model_train == 1:
            for epoch in range(args.model_epoch_start, args.model_epoch_end):
                trainer.cur_epoch = epoch
                trainer.fit()
            trainer.save(args.model_epoch_end)
            trainer.score('test')
            
        else:
            trainer.cur_epoch = args.model_epoch_start
            
            trainer.score('test')
            #trainer.score_my('test2')
                      


if __name__ == "__main__":
    main()