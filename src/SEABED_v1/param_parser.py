"""Getting params from the command line."""

import argparse


def parameter_parser():
    parser = argparse.ArgumentParser(description="Run GEDRanker on SEABED-format data without edge-label GED.")

    parser.add_argument("--dataset", type=str, default="LUBM", help="Dataset name for logging/output naming.")
    parser.add_argument(
        "--dataset-root",
        type=str,
        required=True,
        help="Path to one SEABED dataset directory containing train/ val/ test/ and *_GEDINFO.json.",
    )
    parser.add_argument(
        "--use-raw-features",
        type=int,
        default=1,
        help="Use node embedding features from SEABED json files. Set 0 to use constant features instead.",
    )
    parser.add_argument(
        "--topk-approach",
        choices=["parallel", "sequential"],
        default="parallel",
        help="Choose a top-k mapping generation approach: parallel, or sequential.",
    )
    parser.add_argument(
        "--unsupervised-approach",
        choices=["plain", "GED", "BPR", "Hinge"],
        default="BPR",
        help="Choose an unsupervised training strategy.",
    )
    parser.add_argument("--test-k", type=int, default=100, help="Set k for inference.")
    parser.add_argument("--testset", choices=["test", "val", "train"], default="test", help="Choose a testing split.")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-diffusion_steps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gumbel-iteration", type=int, default=5)
    parser.add_argument("--hidden-dim", type=list, default=[128, 64, 32, 32, 32, 32], help="List of hidden dimensions.")
    parser.add_argument("--d_hidden-dim", type=list, default=[128, 64, 32], help="List of discriminator hidden dimensions.")
    parser.add_argument("--batch-size", type=int, default=128, help="Number of graph pairs per batch.")
    parser.add_argument("--learning-rate", type=float, default=0.001, help="Learning rate.")
    parser.add_argument("--weight-decay", type=float, default=5 * 10 ** -4, help="Adam weight decay.")
    parser.add_argument("--abs-path", type=str, default="../", help="Base path for outputs, relative to execution cwd.")
    parser.add_argument("--result-path", type=str, default="result/", help="Where to save the evaluation results.")
    parser.add_argument("--model-train", type=int, default=1, help="Whether to train the model.")
    parser.add_argument("--model-path", type=str, default="model_save/", help="Where to save the trained model.")
    parser.add_argument("--model-epoch-start", type=int, default=0, help="Checkpoint epoch to load before training/testing.")
    parser.add_argument("--model-epoch-end", type=int, default=200, help="Final epoch.")
    parser.add_argument("--model-name", type=str, default="GEDRankerSEABEDv1", help="Model name.")

    return parser.parse_args()
