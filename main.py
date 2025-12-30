import argparse
from argparse import Namespace

import torch

from subparsers import baseline, fine_tune, evaluate, rag
from utils.utils import MAP_MODELS


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m main",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model and folders
    parser.add_argument(
        "model_name",
        metavar="MODEL",
        choices=MAP_MODELS.keys(),
        type=str,
        help="Model to be fine-tuned.",
    )
    parser.add_argument(
        "data_folder",
        metavar="DATA_FOLDER",
        type=str,
        help="Path to the folder containing the data.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="output",
        help="Path to the output directory.",
    )

    # EFT parameters
    parser.add_argument(
        "--learning-threshold",
        "-lt",
        metavar="THRESHOLD",
        type=float,
        default=0.6,
        help="Threshold for each entity e. If entity_score >= threshold, the model has learned about entity e.",
    )

    # generation parameters
    parser.add_argument(
        "--batch-size",
        metavar="BATCH_SIZE",
        type=int,
        default=4,
        help="Batch size for the datasets.",
    )
    parser.add_argument(
        "--instruction",
        metavar="INSTR",
        type=str,
        default="Answer the following question using as few words as possible.",
        help="Instruction used for generation.",
    )
    parser.add_argument(
        "--max-new-tokens",
        metavar="N_TOKENS",
        type=int,
        default=25,
        help="Maximum number of new tokens to generate.",
    )
    parser.add_argument(
        "--max-len",
        metavar="N_TOKENS",
        type=int,
        default=4096,
        help="Maximum number of new tokens in the input.",
    )
    parser.add_argument(
        "--top-p",
        metavar="TOP_P",
        type=float,
        default=0,
        help="Top-p sampling parameter.",
    )
    parser.add_argument(
        "--temperature",
        metavar="TEMP",
        type=float,
        default=0,
        help="Temperature parameter.",
    )
    parser.add_argument(
        "--top-k",
        metavar="TOP_K",
        type=int,
        default=0,
        help="Top-k sampling parameter.",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Whether to sample or not.",
    )
    parser.add_argument(
        "--add-nes",
        action="store_true",
        help="Add named entities to the validation probes.",
    )
    parser.add_argument(
        "--override",
        action="store_true",
        help="Override previous results.",
    )

    # device, parallel and seed
    parser.add_argument(
        "--device",
        metavar="DEVICE",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for generation.",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Split the model across multiple GPUs.",
    )
    parser.add_argument(
        "--seed",
        metavar="SEED",
        type=int,
        default=42,
        help="Seed for reproducibility.",
    )

    subparsers = parser.add_subparsers(help="", dest="command")
    baseline.configure_subparsers(subparsers)
    fine_tune.configure_subparsers(subparsers)
    evaluate.configure_subparsers(subparsers)
    rag.configure_subparsers(subparsers)

    # parse arguments
    parsed_args = parser.parse_args()

    return parsed_args


if __name__ == "__main__":
    # get the arguments
    args = get_args()

    args.func(args)
