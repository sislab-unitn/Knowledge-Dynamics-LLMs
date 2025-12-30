import argparse
import json
import os
import random
from pathlib import Path

import torch
import numpy as np
from peft.peft_model import PeftModel
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data import (
    ValidationProbesDataset,
    ValidationProbesGenerationCollator,
    ControlProbesDataset,
    ControlProbesGenerationCollator,
)
from utils.utils import MAP_MODELS, evaluate_validation_probes, evaluate_control_probes


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser."""
    parser = subparsers.add_parser(
        "evaluate",
        help="Generate answers for validation and control probes for fine-tuned models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="The name of the experiment to evaluate.",
    )
    parser.add_argument(
        "--probes",
        metavar="PROBES",
        choices={"validation", "control", "all"},
        type=str,
        default="all",
        help="Whether to generate answers for 'validation' probes, 'control' probes, or 'all'.",
    )

    parser.set_defaults(func=main)


def main(args):
    experiment_folder: Path = (
        Path(args.out_dir) / args.model_name / args.experiment_name
    )
    assert (
        experiment_folder.is_dir()
    ), f"Experiment folder '{experiment_folder}' does not exist."

    model_name = MAP_MODELS[args.model_name]

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # Tok config
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    for epoch_folder in experiment_folder.iterdir():
        if epoch_folder.is_dir() and epoch_folder.name != "checkpoint":
            peft_model = PeftModel.from_pretrained(
                model, epoch_folder, is_trainable=False
            )

            output_folder = str(epoch_folder)

            if args.probes in ["validation", "all"]:
                validation_probes_file = (
                    "validation_probes.json"
                    if not args.add_nes
                    else "validation_probes_nes.json"
                )
                with open(
                    os.path.join(args.data_folder, validation_probes_file), "r"
                ) as f:
                    validation_probes = json.load(f)

                valid_probes_ds = ValidationProbesDataset(validation_probes)

                valid_probes_loader = DataLoader(
                    valid_probes_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=ValidationProbesGenerationCollator(
                        tokenizer=tokenizer,  # type: ignore
                        model_name=model_name,
                        instruction=args.instruction,
                        max_len=args.max_len,
                    ),
                )

                # set the seed for reproducibility
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                evaluate_validation_probes(
                    valid_probes_loader=valid_probes_loader,
                    model=peft_model,
                    tokenizer=tokenizer,  # type: ignore
                    output_folder=output_folder,
                    args=args,
                )

            if args.probes in ["control", "all"]:
                with open(os.path.join(args.data_folder, "math_probes.json"), "r") as f:
                    math_probes = json.load(f)

                with open(
                    os.path.join(args.data_folder, "reasoning_probes.json"), "r"
                ) as f:
                    reasoning_probes = json.load(f)

                already_generated_answers = set()
                generation_results = []
                if os.path.isfile(os.path.join(output_folder, "control_probes.json")):
                    with open(
                        os.path.join(output_folder, "control_probes.json"), "r"
                    ) as f:
                        generation_results = json.load(f)
                        already_generated_answers = {
                            p["question_id"] for p in generation_results
                        }

                control_probes_ds = ControlProbesDataset(
                    math_probes, reasoning_probes, already_generated_answers
                )

                control_probes_loader = DataLoader(
                    control_probes_ds,
                    batch_size=args.batch_size,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                    collate_fn=ControlProbesGenerationCollator(
                        tokenizer=tokenizer,  # type: ignore
                        model_name=model_name,
                        max_len=args.max_len,
                    ),
                )

                # set the seed for reproducibility
                random.seed(args.seed)
                np.random.seed(args.seed)
                torch.manual_seed(args.seed)

                evaluate_control_probes(
                    generation_results=generation_results,
                    control_probes_loader=control_probes_loader,
                    model=peft_model,
                    tokenizer=tokenizer,  # type: ignore
                    output_folder=output_folder,
                    args=args,
                )
