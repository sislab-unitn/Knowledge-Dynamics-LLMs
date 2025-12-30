import argparse
import json
import os
import random
from copy import deepcopy

import torch
import numpy as np
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
        "rag",
        help="Generate answers for validation and control probes for default models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--probes",
        metavar="PROBES",
        choices={"validation", "control", "all"},
        type=str,
        default="all",
        help="Whether to generate answers for 'validation' probes, 'control' probes, or 'all'.",
    )
    parser.add_argument(
        "--validation-probes-file",
        metavar="VALIDATION_PROBES_FILE",
        type=str,
        default="validation_probes_w_rag_gold.json",
        help="The file containing validation probes.",
    )

    parser.set_defaults(func=main)


def main(args):
    output_folder = os.path.join(args.out_dir, args.model_name, args.experiment_name)
    os.makedirs(output_folder, exist_ok=True)

    # save the arguments
    args_dict = deepcopy(vars(args))
    del args_dict["func"]
    with open(os.path.join(output_folder, "args.json"), "w") as f:
        json.dump(args_dict, f, indent=4)

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

    if args.probes in ["validation", "all"]:
        with open(
            os.path.join(args.data_folder, args.validation_probes_file), "r"
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
            model=model,
            tokenizer=tokenizer,  # type: ignore
            output_folder=output_folder,
            args=args,
        )

    if args.probes in ["control", "all"]:
        with open(os.path.join(args.data_folder, "math_probes.json"), "r") as f:
            math_probes = json.load(f)

        with open(os.path.join(args.data_folder, "reasoning_probes.json"), "r") as f:
            reasoning_probes = json.load(f)

        already_generated_answers = set()
        generation_results = []
        if os.path.isfile(os.path.join(output_folder, "control_probes.json")):
            with open(os.path.join(output_folder, "control_probes.json"), "r") as f:
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
            model=model,
            tokenizer=tokenizer,  # type: ignore
            output_folder=output_folder,
            args=args,
        )
