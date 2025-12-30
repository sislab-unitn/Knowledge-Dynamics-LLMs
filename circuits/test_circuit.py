import argparse
import json
from argparse import Namespace
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.graph import Graph
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import MAP_MODELS
from data import create_data, save_data, EAPDataset
from metrics import logit_diff, get_hit_at_10


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python -m test_circuit",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # model and folders
    parser.add_argument(
        "model_name",
        metavar="MODEL",
        choices=MAP_MODELS.keys(),
        type=str,
        help="Model to extract the circuit from.",
    )
    parser.add_argument(
        "circuit_folder",
        metavar="CIRCUIT_FOLDER",
        type=str,
        help="Path to the folder containing the knowledge_circuit.pt file.",
    )
    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "--epoch-folder",
        metavar="EPOCH_FOLDER",
        type=str,
        help="Path to the folder containing the checkpoint for the current epoch.",
    )

    return parser.parse_args()


def main():
    model_name = MAP_MODELS[args.model_name]
    if args.epoch_folder:
        model_path = Path(args.epoch_folder) / "merged_model"
        output_path = Path(args.epoch_folder) / "circuits"
        print(f"Using model from epoch folder: {model_path}")
    else:
        model_path = model_name
        output_path = Path("./circuits") / args.model_name
    hf_model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    model = HookedTransformer.from_pretrained(
        model_name,
        center_writing_weights=False,
        center_unembed=False,
        fold_ln=False,
        hf_model=hf_model,
        device="cuda",
        tokenizer=tokenizer,
    )
    model.cfg.use_split_qkv_input = True
    model.cfg.use_attn_result = True
    model.cfg.use_hook_mlp_in = True
    model.cfg.ungroup_grouped_query_attention = True

    clean, correct_idxs, corrupted, incorrect_idxs = create_data(model)
    save_data(clean, correct_idxs, corrupted, incorrect_idxs, "presidents.csv")
    dataset = EAPDataset("presidents.csv")
    dataloader = dataset.to_dataloader(1)

    g = Graph.from_pt(Path(args.circuit_folder) / "knowledge_circuit.pt")  # type: ignore

    output_path.mkdir(parents=True, exist_ok=True)

    results = {}
    results["logit_diff"] = evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()  # type: ignore
    results["hit_at_10"] = evaluate_graph(model, g, dataloader, partial(get_hit_at_10, loss=False, mean=False)).sum().item() / len(dataset)  # type: ignore

    with open(output_path / f"{args.experiment_name}.json", "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    main()
