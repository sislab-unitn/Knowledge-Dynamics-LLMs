import argparse
import json
import random
from argparse import Namespace
from collections import defaultdict
from functools import partial
from pathlib import Path
from statistics import mean, stdev

import numpy as np
import torch
from eap.attribute import attribute
from eap.evaluate import evaluate_graph, evaluate_baseline
from eap.graph import Graph
from transformer_lens import HookedTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import trange

from utils import MAP_MODELS  # type: ignore
from data import create_data, save_data, EAPDataset  # type: ignore
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
        prog="python -m discover_circuits",
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
        "--epoch-folder",
        metavar="EPOCH_FOLDER",
        type=str,
        help="Path to the folder containing the checkpoint for the current epoch.",
    )
    parser.add_argument(
        "--top-n",
        metavar="TOP_N",
        type=int,
        default=50000,
        help="Number of edges to consider in the circuit.",
    )
    parser.add_argument(
        "--ig-steps",
        metavar="IG_STEPS",
        type=int,
        default=30,
        help="Number of steps for Integrated Gradients.",
    )
    parser.add_argument(
        "--edges-in-figure",
        metavar="EDGES_IN_FIGURE",
        type=int,
        default=100,
        help="Number of edges to include in the final figure.",
    )
    parser.add_argument(
        "-n-runs",
        metavar="N_RUNS",
        type=int,
        default=5,
        help="Number of runs for evaluation metrics.",
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

    results = defaultdict(dict)
    results["baseline"]["logit_diff"] = evaluate_baseline(model, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item()  # type: ignore

    best_graph = None
    best_score = -float("inf")
    results["baseline"]["hit_at_10"] = evaluate_baseline(model, dataloader, partial(get_hit_at_10, loss=False, mean=False)).sum().item() / len(dataset)  # type: ignore

    logit_diffs = []
    hit_at_10s = []
    for run in trange(args.n_runs):
        random.seed(run)
        torch.manual_seed(run)
        np.random.seed(run)
        dataloader = dataset.to_dataloader(1, shuffle=True)

        g = Graph.from_model(model)
        attribute(
            model,
            g,
            dataloader,
            partial(logit_diff, loss=True, mean=True),  # type: ignore
            method="EAP-IG-inputs",  # type: ignore
            ig_steps=args.ig_steps,
        )

        output_path.mkdir(parents=True, exist_ok=True)
        g.apply_topn(args.top_n, True)

        results["circuits"][run] = {
            "logit_diff": evaluate_graph(model, g, dataloader, partial(logit_diff, loss=False, mean=False)).mean().item(),  # type: ignore
            "hit_at_10": evaluate_graph(model, g, dataloader, partial(get_hit_at_10, loss=False, mean=False)).sum().item() / len(dataset),  # type: ignore
        }

        logit_diffs.append(results["circuits"][run]["logit_diff"])
        hit_at_10s.append(results["circuits"][run]["hit_at_10"])

        if results["circuits"][run]["hit_at_10"] > best_score:
            best_score = results["circuits"][run]["hit_at_10"]
            best_graph = g

    results["mean"] = {
        "logit_diff": mean(logit_diffs),
        "hit_at_10": mean(hit_at_10s),
    }
    results["stdev"] = {
        "logit_diff": stdev(logit_diffs) if args.n_runs > 1 else 0.0,
        "hit_at_10": stdev(hit_at_10s) if args.n_runs > 1 else 0.0,
    }

    best_graph.to_pt(output_path / "knowledge_circuit.pt")  # type: ignore
    with open(output_path / "results.json", "w") as f:
        json.dump(results, f, indent=4)

    best_graph.apply_topn(args.edges_in_figure, True)  # type: ignore
    best_graph.to_image(output_path / "knowledge_circuit.png")  # type: ignore


if __name__ == "__main__":
    args = get_args()
    main()
