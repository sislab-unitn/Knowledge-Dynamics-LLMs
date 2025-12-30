import argparse
import gzip
import json
import os
import random

import torch
import numpy as np
from peft import LoraConfig, get_peft_model  # type: ignore
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data import (
    EFTDataset,
    ValidationProbesDataset,
    ValidationProbesGenerationCollator,
)

from utils.utils import (
    Checkpointer,
    MAP_MODELS,
    load_training_params,
    resume_training,
    save_training_params,
    train,
)


def configure_subparsers(subparsers: argparse._SubParsersAction):
    """Configure a new subparser ."""
    parser = subparsers.add_parser(
        "fine-tune",
        help="Fine-Tune a model on a specific dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "experiment_name",
        metavar="EXPERIMENT_NAME",
        type=str,
        help="Name of the experiment.",
    )
    parser.add_argument(
        "name_entities",
        metavar="NAME_ENTITIES",
        choices={"no", "all", "sub"},
        type=str,
        help="Whether to use a dataset without ('no'), with a subset ('sub'), or all ('all') NEs tags.",
    )
    parser.add_argument(
        "--add-chat-template",
        action="store_true",
        help="Add chat template to the input (only for Instruct Models).",
    )
    parser.add_argument(
        "--eft",
        action="store_true",
        help="Whether to optimize the training data based on the validation probes while training.",
    )
    parser.add_argument(
        "--main-docs",
        action="store_true",
        help="Whether to use ONLY the main documents for training.",
    )
    parser.add_argument(
        "--epochs",
        metavar="EPOCHS",
        type=int,
        default=5,
        help="Number of epochs for training.",
    )
    parser.add_argument(
        "--save-every",
        metavar="SAVE_EVERY",
        type=int,
        default=1000,
        help="Save model every SAVE_EVERY steps.",
    )
    parser.add_argument(
        "--lr",
        metavar="LR",
        type=float,
        default=1e-4,
        help="Learning rate for training.",
    )
    parser.add_argument(
        "--max-patience",
        metavar="MAX_PATIENCE",
        type=int,
        default=10,
        help="Maximum patience for early stopping.",
    )
    parser.add_argument(
        "--r",
        metavar="R",
        type=int,
        default=32,
        help="Rank for LoRA.",
    )
    parser.add_argument(
        "--lora-alpha",
        metavar="LORA_ALPHA",
        type=int,
        default=64,
        help="Alpha value for LoRA.",
    )
    parser.add_argument(
        "--add-kl",
        choices={"pre-trained", "last-epoch", "both", "best"},
        type=str,
        default=None,
        help="Add KL divergence loss to keep the model output close to the 'pre-trained' model, 'last-epoch', 'both', or 'best' model.",
    )
    parser.add_argument(
        "--use-jensen-shannon",
        "--use-js",
        action="store_true",
        help="Use Jensen-Shannon divergence instead of KL divergence.",
    )
    parser.add_argument(
        "--lambda-kl",
        metavar="LAMBDA_KL",
        type=float,
        default=0.2,
        help="Weight for the KL divergence loss.",
    )

    parser.set_defaults(func=main)


def main(args):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_name = MAP_MODELS[args.model_name]
    if args.add_chat_template:
        assert (
            "instruct" in model_name.lower()
        ), "Chat template can only be added to Instruct models."

    if args.add_kl in ["pre-trained", "last-epoch", "best"]:
        assert (
            torch.cuda.device_count() > 1
        ), f"KL divergence loss requires more than one GPU with '{args.add_kl}'."
    elif args.add_kl == "both":
        assert (
            torch.cuda.device_count() > 2
        ), "KL divergence loss requires more than two GPUs with 'both'."

    if args.add_kl == "best":
        assert (
            args.eft is True
        ), "KL divergence for the 'best' model can only be computed with EFT training."

    # removes the function from the args for serialization
    del args.func

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    output_folder = os.path.join(args.out_dir, args.model_name, args.experiment_name)
    if not os.path.exists(output_folder):
        # create output folder
        os.makedirs(output_folder)
        save_training_params(args, output_folder)
    else:
        # load training parameters from existing folder
        print(f"Experiment '{args.experiment_name}' already exists.")
        print(f"Loading training parameters from {output_folder}.")
        args = load_training_params(output_folder)

    # Loss function
    criterion = CrossEntropyLoss(ignore_index=-100, reduction="sum")

    # Tok config
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto" if args.parallel else args.device,
        torch_dtype=torch.bfloat16,
    )

    checkpointer = Checkpointer(args)
    early_stopping = checkpointer.checkpoint.early_stopping
    if os.path.exists(os.path.join(output_folder, "checkpoint")):
        checkpoint, model = checkpointer.load_checkpoint(model, output_folder)
        early_stopping = checkpoint.early_stopping
        start_epoch = checkpoint.epoch
    else:
        # Initialize training for the first time
        start_epoch = 0

        # default configuration from https://huggingface.co/docs/peft/en/developer_guides/quantization
        config = LoraConfig(
            r=args.r,
            lora_alpha=args.lora_alpha,
            init_lora_weights="gaussian",
            target_modules=["q_proj", "v_proj"],
            modules_to_save=["embed_tokens"],
        )

        model = get_peft_model(model, config)

    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)  # type: ignore

    if checkpointer.checkpoint.optimizer is not None:
        optimizer.load_state_dict(checkpointer.checkpoint.optimizer)

    train_file = f"train_{args.name_entities}_ne.json.gz"
    with gzip.open(os.path.join(args.data_folder, train_file), "r") as f:
        docs = json.load(f)

    with open(os.path.join(args.data_folder, "docs_per_entity.json"), "r") as f:
        docs_per_entity = json.load(f)

    main_doc_ids = None
    if args.main_docs:
        with open(os.path.join(args.data_folder, "main_doc_ids.json"), "r") as f:
            main_docs = json.load(f)
        main_doc_ids = set(main_docs.values())

    correct_entities = set()  # initialize as empty set
    train_ds = EFTDataset(
        correct_entities=correct_entities,
        docs_per_entity=docs_per_entity,
        docs=docs,
        tokenizer=tokenizer,  # type: ignore
        model_name=model_name,
        valid_doc_ids=main_doc_ids,
        add_chat_template=args.add_chat_template,
        max_len=args.max_len,
    )

    validation_probes_file = (
        "validation_probes.json" if not args.add_nes else "validation_probes_nes.json"
    )
    with open(os.path.join(args.data_folder, validation_probes_file), "r") as f:
        validation_probes = json.load(f)

    valid_probes_ds = ValidationProbesDataset(validation_probes)

    valid_probes_loader = DataLoader(
        valid_probes_ds,
        batch_size=4,
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

    kl_models = [None, None]
    if args.add_kl == "pre-trained":
        pretrained_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:1"
        )
        pretrained_model.eval()
        kl_models[0] = pretrained_model
    elif args.add_kl in ["last-epoch", "best"]:
        last_epoch_model = AutoModelForCausalLM.from_pretrained(
            model_name, torch_dtype=torch.bfloat16, device_map="cuda:1"
        )
        last_epoch_model.eval()
        kl_models[1] = last_epoch_model
    elif args.add_kl is None:
        # avoid error until we implement the other options
        ...
    else:
        raise ValueError(f"{args.add_kl} is not implemented yet")

    if resume_training(start_epoch, args, early_stopping):
        train(
            args=args,
            model=model,  # type: ignore
            tokenizer=tokenizer,  # type: ignore
            train_ds=train_ds,
            validation_probes_loader=valid_probes_loader,
            criterion=criterion,
            optimizer=optimizer,
            output_folder=output_folder,
            checkpointer=checkpointer,
            pad_token_id=tokenizer.pad_token_id,  # type: ignore
            kl_models=kl_models,  # type: ignore
        )
