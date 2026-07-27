"""Getting params from the command line."""

import argparse

def parameter_parser():
    """
    A method to parse up command line parameters.
    The default hyperparameters give a high performance model without grid search.
    """
    parser = argparse.ArgumentParser(description="Run GEDRanker on SEABED-format data.")
    
    parser.add_argument('--topk-approach', choices=['parallel','sequential'],default='parallel', help="Choose a top-k mapping generation approach: parallel, or sequential.")
    parser.add_argument('--unsupervised-approach', choices=['plain','GED','BPR','Hinge'],default='BPR', help="Choose an unsupervised training strategy.")
    parser.add_argument('--test-k', type=int, default=100,help='Set k for inference.')
    
    parser.add_argument('--k-range', type=list, default=[1,10,20,30,40,50,60,70,80,90,100],help='range of k for top-k approach analysis.')
    
    parser.add_argument('--experiment', choices=['test', 'topk_analysis', 'diversity_analysis'],default='test', help="Choose an experiment: test, topk_analysis, or diversity_analysis.")
    
    parser.add_argument('--testset', choices=['test', 'val', 'train'],default='test', help="Choose a testing graph set: test, val, or train.")

    parser.add_argument('--diffusion-steps', type=int, default=1000)

    parser.add_argument('--inference-diffusion_steps', type=int, default=10)
    parser.add_argument('--tau',type=float,default=1)
    parser.add_argument('--gumbel-iteration',type=int,default=5)
    parser.add_argument("--hidden-dim",
                        type=list,
                        default=[128,64,32,32,32,32],
	                help="List of hidden dimensions.")
                    
    parser.add_argument("--d_hidden-dim",
                        type=list,
                        default=[128,64,32],
	                help="List of hidden dimensions.")

    parser.add_argument("--tensor-neurons",
                        type=int,
                        default=16,
	                help="Neurons in tensor network layer. Default is 16.")
    
    parser.add_argument("--bottle-neck-neurons",
                        type=list,
                        default=[16,8,4],
	                help="List of bottle neck layer neurons.")
    
    parser.add_argument("--weight-matrix-dim",
                        type=int,
                        default=16,
                        help="the size of weight matrix in GedMatrixModule. Default is 16.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--pretrain-batch-size",
                        type=int,
                        default=128,
                        help="Number of graph pairs per batch. Default is 128.")

    parser.add_argument("--pretrain",
                        type=bool,
                        default=False,
                        help="contrastive pretrain")

    parser.add_argument("--learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")
    
    parser.add_argument("--pretrain-learning-rate",
                        type=float,
                        default=0.001,
	                help="Learning rate. Default is 0.001.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=5*10**-4,
	                help="Adam weight decay. Default is 5*10^-4.")


    parser.add_argument("--dataset-root",
                        type=str,
                        required=True,
                        help="Path to one SEABED dataset directory containing train/ val/ test/ and *_GEDINFO.json.")

    parser.add_argument("--result-path",
                        type=str,
                        default='result/',
                        help="Where to save the evaluation results")

    parser.add_argument("--model-train",
                        type=int,
                        default=1,
                        help='Whether to train the model')

    parser.add_argument("--model-path",
                        type=str,
                        default='model_save/',
                        help="Where to save the trained model")

    parser.add_argument("--model-epoch-start",
                        type=int,
                        default=0,
                        help="The number of epochs the initial saved model has been trained.")

    parser.add_argument("--model-epoch-end",
                        type=int,
                        default=200,
                        help="The number of epochs the final saved model has been trained.")

    parser.add_argument("--dataset",
                        type=str,
                        default='AIDS',
                        help="dataset name")

    parser.add_argument("--model-name",
                        type=str,
                        default='GEDRankerSEABEDStrict',
                        help="model name")

    

    return parser.parse_args()
