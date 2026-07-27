from utils import tab_printer
from trainer import Trainer
from param_parser import parameter_parser
import random
import numpy as np
import os
import torch
from datetime import datetime

def seed_everything(TORCH_SEED):
	random.seed(TORCH_SEED)
	os.environ['PYTHONHASHSEED'] = str(TORCH_SEED)
	np.random.seed(TORCH_SEED)
	torch.manual_seed(TORCH_SEED)
	torch.cuda.manual_seed_all(TORCH_SEED)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

def main():
    
    seed_everything(0)
    """
    Parsing command line parameters, reading data.
    Fitting and scoring a SimGNN model.
    """
    args = parameter_parser()
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(base_dir))
    args.abs_path = project_root.rstrip("/\\") + "/"
    args.run_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(project_root, args.result_path), exist_ok=True)
    os.makedirs(os.path.join(project_root, args.model_path), exist_ok=True)
    tab_printer(args)

    trainer = Trainer(args)
    
    if args.model_epoch_start > 0:
        trainer.load(args.model_epoch_start)

    if args.model_train == 1:
        for epoch in range(args.model_epoch_start, args.model_epoch_end):
            trainer.cur_epoch = epoch
            trainer.fit()
        trainer.save(args.model_epoch_end)
            
        
        trainer.score('test',test_k=100,top_k_approach='parallel')

        
    else:
        
        trainer.score(testing_graph_set=args.testset,test_k=args.test_k,top_k_approach=args.topk_approach)
       
            
           

if __name__ == "__main__":
    main()
