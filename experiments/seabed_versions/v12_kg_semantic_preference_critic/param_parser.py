"""V12 command-line parsing for the isolated KG semantic preference critic line."""

from __future__ import annotations

import argparse
import os


def _parse_int_list(value):
    if isinstance(value, list):
        return [int(v) for v in value]
    if value is None:
        return []
    text = str(value).strip()
    if not text:
        return []
    text = text.strip("[]")
    parts = [item.strip() for item in text.split(",") if item.strip()]
    return [int(item) for item in parts]


def parameter_parser():
    parser = argparse.ArgumentParser(
        description="Run the isolated V12 KG semantic preference critic experiment."
    )

    parser.add_argument("--dataset", type=str, default="LUBM")
    parser.add_argument("--dataset-root", type=str, required=True)
    parser.add_argument("--use-raw-features", type=int, default=1)
    parser.add_argument("--ged-column", type=int, default=3)
    parser.add_argument("--cost-mode", choices=["unit", "containment"], default="unit")
    parser.add_argument("--topk-approach", choices=["parallel", "sequential"], default="parallel")
    parser.add_argument("--unsupervised-approach", choices=["plain", "GED", "BPR", "Hinge"], default="BPR")
    parser.add_argument("--test-k", type=int, default=100)
    parser.add_argument("--testset", choices=["test", "val", "train"], default="test")
    parser.add_argument("--max-train-pairs", type=int, default=0)
    parser.add_argument("--max-val-pairs", type=int, default=0)
    parser.add_argument("--max-test-pairs", type=int, default=0)
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--inference-diffusion_steps", type=int, default=10)
    parser.add_argument("--tau", type=float, default=1.0)
    parser.add_argument("--gumbel-iteration", type=int, default=5)
    parser.add_argument(
        "--hidden-dim",
        type=_parse_int_list,
        default=[128, 64, 32, 32, 32, 32],
        help="Generator hidden dimensions.",
    )
    parser.add_argument(
        "--d_hidden-dim",
        type=_parse_int_list,
        default=[128, 64, 32],
        help="Discriminator hidden dimensions.",
    )
    parser.add_argument(
        "--semantic-hidden-dim",
        type=_parse_int_list,
        default=[64, 32],
        help="Auxiliary semantic critic hidden dimensions.",
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--abs-path", type=str, default="../")
    parser.add_argument("--result-path", type=str, default="result/")
    parser.add_argument("--model-train", type=int, default=1)
    parser.add_argument("--model-path", type=str, default="model_save/")
    parser.add_argument("--model-epoch-start", type=int, default=0)
    parser.add_argument("--model-epoch-end", type=int, default=200)
    parser.add_argument("--model-name", type=str, default="GEDRankerSEABED_v12_semantic_critic")

    parser.add_argument(
        "--control-mode",
        choices=[
            "full",
            "raw",
            "constant",
            "shuffled",
            "no_critic",
            "no_keep",
            "no_adapt",
            "no_sparse",
        ],
        default=os.environ.get("CONTROL_MODE", "full"),
        help="Isolated V12 control configuration.",
    )
    parser.add_argument("--relation-mode", choices=["raw", "constant", "shuffled"], default=os.environ.get("RELATION_MODE", "raw"))
    parser.add_argument("--semantic-weight", type=float, default=0.15)
    parser.add_argument("--keep-loss-weight", type=float, default=0.10)
    parser.add_argument("--explore-weight", type=float, default=0.05)
    parser.add_argument("--adaptive-explore", type=int, default=1)
    parser.add_argument("--sparse-matching", type=int, default=1)
    parser.add_argument("--sparse-top-r", type=int, default=32)
    parser.add_argument("--sparse-random-epsilon", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--run-timestamp", type=str, default="")

    args = parser.parse_args()
    args.hidden_dim = _parse_int_list(args.hidden_dim)
    args.d_hidden_dim = _parse_int_list(args.d_hidden_dim)
    args.semantic_hidden_dim = _parse_int_list(args.semantic_hidden_dim)
    return args
