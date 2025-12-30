import argparse
from argparse import Namespace
from pathlib import Path

import torch
from peft.peft_model import PeftModel
from transformers import AutoModelForCausalLM

MAP_MODELS = {
    "olmo": "allenai/OLMo-7B-0724-hf",
    "olmo-1b": "allenai/OLMo-1B-0724-hf",
    "olmo-i": "allenai/OLMo-7B-0724-Instruct-hf",
    "llama3": "meta-llama/Llama-3.1-8B",
    "llama3-i": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-i-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python utils/merge_adapter_with_model.py",
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
        "epoch_folder",
        metavar="EPOCH_FOLDER",
        type=str,
        help="Path to the folder containing the checkpoint for the current epoch.",
    )

    return parser.parse_args()


def main(args):

    model_name = MAP_MODELS[args.model_name]

    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)

    peft_model = PeftModel.from_pretrained(model, args.epoch_folder, is_trainable=False)

    merged_model = peft_model.merge_and_unload()  # type: ignore
    merged_model.save_pretrained(Path(args.epoch_folder) / "merged_model")


if __name__ == "__main__":
    args = get_args()
    main(args)
