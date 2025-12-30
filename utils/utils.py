import json
import math
import os
import random
import re
import tarfile
from argparse import Namespace
from datetime import datetime, timedelta
from typing import Dict, Iterator, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
import evaluate
from peft.peft_model import PeftModel
from rouge_score.rouge_scorer import RougeScorer
from tqdm import tqdm, trange
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, PreTrainedModel, PreTrainedTokenizer

from utils.data import EFTDataset, EFTCollator

MAP_MODELS = {
    "olmo": "allenai/OLMo-7B-0724-hf",
    "olmo-1b": "allenai/OLMo-1B-0724-hf",
    "olmo-i": "allenai/OLMo-7B-0724-Instruct-hf",
    "llama3": "meta-llama/Llama-3.1-8B",
    "llama3-i": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-i-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def compute_rouge(
    scorer: RougeScorer, predictions: List[str], references: List[str]
) -> Dict[str, float]:
    scores: Dict[str, List[float]] = {
        "f1": [],
        "recall": [],
    }

    for pred, ref in zip(predictions, references):
        score = scorer.score(target=ref, prediction=pred)
        scores["f1"].append(score["rouge1"].fmeasure)
        scores["recall"].append(score["rouge1"].recall)

    results: Dict[str, float] = {}
    for key in scores:
        results[key] = np.mean(scores[key])  # type: ignore

    return results


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def compute_nll_and_ppl(
    losses: List[float], unmasked_tokens: int
) -> Tuple[float, float]:
    nll = sum(losses) / unmasked_tokens
    ppl = math.exp(nll)
    return nll, ppl


def compute_kl_div(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    # move the target logits to the same device as the input logits
    target_logits = target_logits.to(device)
    input_logits = input_logits.to(device)
    mask = mask.to(device)

    # compute the KL divergence between the predicted logits and the target logits
    kl_div = F.kl_div(
        F.log_softmax(input_logits, dim=-1),
        F.softmax(target_logits, dim=-1),
        reduction="none",
    ).sum(dim=-1)

    assert (
        kl_div.size() == mask.size()
    ), f"KL Div size {kl_div.size()} does not match mask size {mask.size()}"

    # consider only non padded tokens
    kl_div = (kl_div * mask).sum()

    return kl_div


def compute_js_div(
    input_logits: torch.Tensor,
    target_logits: torch.Tensor,
    mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:

    # move the target logits to the same device as the input logits
    input_logits = input_logits.to(device)
    target_logits = target_logits.to(device)
    mask = mask.to(device)

    input_prob = F.softmax(input_logits, dim=-1)
    target_prob = F.softmax(target_logits, dim=-1)

    midpoint = (input_prob + target_prob) / 2
    midpoint = midpoint.log()

    input_log = F.log_softmax(input_logits, dim=-1)
    target_log = F.log_softmax(target_logits, dim=-1)

    # compute the KL divergence between the predicted logits and the target logits
    input_div = F.kl_div(
        input=midpoint,
        target=input_log,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)

    target_div = F.kl_div(
        input=midpoint,
        target=target_log,
        log_target=True,
        reduction="none",
    ).sum(dim=-1)

    js_div = (input_div + target_div) / 2

    assert (
        js_div.size() == mask.size()
    ), f"Jensen-Shannon Div size {js_div.size()} does not match mask size {mask.size()}"

    # consider only non padded tokens
    js_div = (js_div * mask).sum()

    return js_div


def compute_aux_loss(
    input_logits: torch.Tensor,
    mask: torch.Tensor,
    input_ids: torch.Tensor,
    pretrained_model: Optional[PreTrainedModel],
    last_epoch_model: Optional[Union[PreTrainedModel, PeftModel]],
    args: Namespace,
) -> torch.Tensor:
    device = input_logits.device
    dtype = input_logits.dtype

    # initialize the KL Div losses with tensors with size []
    pre_trained_div: torch.Tensor = torch.tensor(0.0, dtype=dtype, device=device)
    last_epoch_kl_div: torch.Tensor = torch.tensor(0.0, dtype=dtype, device=device)

    if args.add_kl in ["pre-trained", "both"]:
        assert (
            pretrained_model is not None
        ), f"Pre-trained model is required for KL Div {args.add_kl}"

        pretrained_model.eval()
        input_ids = input_ids.to(pretrained_model.device)
        with torch.no_grad():
            target_logits: torch.Tensor = pretrained_model(input_ids).logits

        if args.use_jensen_shannon:
            pre_trained_div = compute_js_div(
                input_logits=input_logits,
                target_logits=target_logits,
                mask=mask,
                device=input_logits.device,
            )
        else:
            pre_trained_div = compute_kl_div(
                input_logits=input_logits,
                target_logits=target_logits,
                mask=mask,
                device=input_logits.device,
            )

    if args.add_kl in ["last-epoch", "best", "both"]:
        assert (
            last_epoch_model is not None
        ), f"Last epoch model is required for KL Div {args.add_kl}"

        last_epoch_model.eval()
        input_ids = input_ids.to(last_epoch_model.device)  # type: ignore
        with torch.no_grad():
            target_logits: torch.Tensor = last_epoch_model(input_ids).logits

        if args.use_jensen_shannon:
            last_epoch_kl_div = compute_js_div(
                input_logits=input_logits,
                target_logits=target_logits,
                mask=mask,
                device=input_logits.device,
            )
        else:
            last_epoch_kl_div = compute_kl_div(
                input_logits=input_logits,
                target_logits=target_logits,
                mask=mask,
                device=input_logits.device,
            )

    assert (
        pre_trained_div.size() == last_epoch_kl_div.size()
    ), f"Pre-trained KL Div size {pre_trained_div.size()} does not match Last Epoch KL Div size {last_epoch_kl_div.size()}"

    return pre_trained_div + last_epoch_kl_div


def create_train_loader(
    train_ds: EFTDataset, args: Namespace, pad_token_id: int
) -> DataLoader:
    return DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=False,  # shuffle is done in the dataset
        num_workers=4,
        pin_memory=True,
        collate_fn=EFTCollator(pad_token_id=pad_token_id),
        worker_init_fn=seed_worker,
        generator=torch.Generator().manual_seed(args.seed),
    )


def tar_filter(tarinfo):
    """Exclude specific folders such as __pycache__."""
    excluded_dirs = {
        "__pycache__",
        ".git",
        ".venv",
        ".env",
    }  # Add other unwanted folders here
    if any(excluded_dir in tarinfo.name for excluded_dir in excluded_dirs):
        return None  # Exclude this file/folder
    return tarinfo  # Include all others


def create_tar_gz(archive_name, paths):
    """
    Create a .tar.gz archive from multiple files and folders, excluding certain directories.

    :param archive_name: Name of the output archive file (e.g., 'backup.tar.gz')
    :param paths: List of file and folder paths to include in the archive
    """
    with tarfile.open(archive_name, "w:gz") as tar:
        for path in paths:
            arcname = os.path.basename(path)  # Store without absolute paths
            tar.add(path, arcname=arcname, filter=tar_filter)


def save_training_params(args: Namespace, output_folder: str) -> None:
    with open(os.path.join(output_folder, "training_params.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    paths = ["utils", "subparsers", "main.py"]
    create_tar_gz(os.path.join(output_folder, "code.tar.gz"), paths)


def load_training_params(output_folder: str) -> Namespace:
    with open(os.path.join(output_folder, "training_params.json"), "r") as f:
        return Namespace(**json.load(f))


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.counter = 0
        self.best_ppl = float("inf")
        self.stopped = False

    def should_stop(self, current_ppl: float) -> bool:
        if current_ppl < self.best_ppl:
            self.best_ppl = current_ppl
            self.counter = 0
        else:
            self.counter += 1

        # save the counter condition for the checkpointer
        self.stopped = self.counter >= self.patience
        return self.stopped


MODELS_BASELINE = {
    "llama3-i": 36,
    "llama3-i-1b": 12,
    "olmo-i": 17,
}


class Checkpoint:
    def __init__(self, args: Namespace):
        self.epoch = 0
        self.step = 0
        self.optimizer: Optional[dict] = None
        self.early_stopping = EarlyStopping(args.max_patience)
        self.train_stats = []
        self.losses = []
        self.unmasked_tokens = 0
        self.correct_entities: Set[str] = set()
        self.kl_div_losses = []
        self.total_time: timedelta = timedelta(0)
        self.best_epoch: Optional[int] = None
        self.most_correct_entities: int = MODELS_BASELINE[args.model_name]


class Checkpointer:
    def __init__(self, args: Namespace):
        self.checkpoint = Checkpoint(args)

    def update_checkpoint(
        self,
        model: PeftModel,
        optimizer: torch.optim.Optimizer,  # type: ignore
        step: int,
        losses: List[float],
        unmasked_tokens: int,
        output_folder: str,
        total_time: timedelta,
        train_stats: Optional[List[dict]] = None,
        early_stopping: Optional[EarlyStopping] = None,
        epoch: Optional[int] = None,
        correct_entities: Optional[Set[str]] = None,
        kl_div_losses: Optional[List[float]] = None,
        best_epoch: Optional[int] = None,
        most_correct_entities: Optional[int] = None,
    ):
        model.save_pretrained(os.path.join(output_folder, "checkpoint"))
        self.checkpoint.optimizer = optimizer.state_dict()
        if epoch is not None:
            self.checkpoint.epoch = epoch
        self.checkpoint.step = step
        self.checkpoint.losses = losses
        self.checkpoint.unmasked_tokens = unmasked_tokens
        self.checkpoint.total_time = total_time
        if train_stats is not None:
            self.checkpoint.train_stats = train_stats
        if early_stopping is not None:
            self.checkpoint.early_stopping = early_stopping
        if correct_entities is not None:
            self.checkpoint.correct_entities = correct_entities
        if kl_div_losses is not None:
            self.checkpoint.kl_div_losses = kl_div_losses
        if best_epoch is not None:
            self.checkpoint.best_epoch = best_epoch
        if most_correct_entities is not None:
            self.checkpoint.most_correct_entities = most_correct_entities

        torch.save(
            self.checkpoint, os.path.join(output_folder, "checkpoint", "checkpoint.pt")
        )

    def load_checkpoint(
        self, model: AutoModelForCausalLM, output_folder: str
    ) -> Tuple[Checkpoint, PeftModel]:
        model = PeftModel.from_pretrained(
            model, os.path.join(output_folder, "checkpoint"), is_trainable=True  # type: ignore
        )
        self.checkpoint = torch.load(
            os.path.join(output_folder, "checkpoint", "checkpoint.pt"),
            weights_only=False,  # torch 2.6 behavior
        )
        return self.checkpoint, model  # type: ignore


def resume_training(epoch: int, args: Namespace, early_stopping: EarlyStopping) -> bool:
    if epoch >= args.epochs:
        print(f"Reached maximum number of epochs {epoch}.")
        return False
    if early_stopping.stopped:
        print(f"Early stopping at epoch {epoch}")
        return False
    return True


def train_one_epoch(
    args,
    model: PeftModel,
    optimizer: torch.optim.Optimizer,  # type: ignore
    train_iterator: Iterator,
    dataloader: DataLoader,
    criterion: CrossEntropyLoss,
    start_step: int,
    steps_so_far: int,
    checkpointer: Checkpointer,
    output_folder: str,
    pretrained_model: Optional[PreTrainedModel],
    last_epoch_model: Optional[Union[PreTrainedModel, PeftModel]],
) -> Tuple[Tuple[float, float], float, int, timedelta]:

    model.train()  # type: ignore
    losses = checkpointer.checkpoint.losses
    kl_div_losses = checkpointer.checkpoint.kl_div_losses
    unmasked_tokens = checkpointer.checkpoint.unmasked_tokens
    total_time = checkpointer.checkpoint.total_time

    # Resume training for the current step
    for _ in range(start_step):
        next(train_iterator)

    with tqdm(dataloader, desc="Training") as pbar:
        pbar.total = len(dataloader)
        pbar.n = start_step
        pbar.refresh()
        for step, (input_ids, labels, *_) in enumerate(
            train_iterator, start=start_step
        ):
            start = datetime.now()

            input_ids = input_ids.to(args.device)
            labels = labels.to(args.device)

            optimizer.zero_grad()
            logits: torch.Tensor = model(input_ids).logits
            outputs = logits.permute(0, 2, 1)

            # compute the loss as sum
            loss = criterion(outputs, labels)
            # store the original loss
            losses.append(loss.item())
            # compute the number of unmasked tokens for this batch
            mask = labels != -100
            batch_unmasked_tokens = (mask).sum().item()
            # normalize the loss
            loss /= batch_unmasked_tokens

            if args.add_kl:
                kl_div_loss = compute_aux_loss(
                    input_logits=logits,
                    mask=mask,
                    input_ids=input_ids,
                    pretrained_model=pretrained_model,
                    last_epoch_model=last_epoch_model,
                    args=args,
                )
                # multiply the kl_div_loss by the lambda_kl parameter
                kl_div_loss *= args.lambda_kl
                # store the kl_div_loss
                kl_div_losses.append(kl_div_loss.item())
                # normalize the kl_div_loss
                kl_div_loss /= batch_unmasked_tokens

                assert (
                    loss.size() == kl_div_loss.size()
                ), f"Loss size {loss.size()} does not match KL Div size {kl_div_loss.size()}"

                # add the kl_div_loss to the original loss
                loss += kl_div_loss

            loss.backward()
            optimizer.step()

            unmasked_tokens += batch_unmasked_tokens
            _, ppl = compute_nll_and_ppl(losses, unmasked_tokens)

            postfix = {"Train PPL": ppl}
            if args.add_kl:
                postfix["Train KL Div"] = sum(kl_div_losses) / unmasked_tokens

            pbar.set_postfix(postfix)
            pbar.update(1)
            steps_so_far += 1

            end = datetime.now()
            batch_time = end - start
            total_time += batch_time

            if steps_so_far % args.save_every == 0:
                checkpointer.update_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    step=step + 1,  # Increment the current step
                    losses=losses,
                    kl_div_losses=kl_div_losses,
                    unmasked_tokens=unmasked_tokens,
                    output_folder=output_folder,
                    total_time=total_time,
                )

    return (
        compute_nll_and_ppl(losses, unmasked_tokens),
        sum(kl_div_losses) / unmasked_tokens,
        steps_so_far,
        total_time,
    )


def train(
    args: Namespace,
    model: PeftModel,
    tokenizer: PreTrainedTokenizer,
    train_ds: EFTDataset,
    validation_probes_loader: DataLoader,
    criterion: CrossEntropyLoss,
    optimizer: torch.optim.Optimizer,  # type: ignore
    output_folder: str,
    checkpointer: Checkpointer,
    pad_token_id: int,
    kl_models: List[Optional[PreTrainedModel]],
) -> None:

    start_epoch = checkpointer.checkpoint.epoch
    start_step = checkpointer.checkpoint.step
    train_stats = checkpointer.checkpoint.train_stats
    early_stopping = checkpointer.checkpoint.early_stopping
    correct_entities = checkpointer.checkpoint.correct_entities
    best_epoch = checkpointer.checkpoint.best_epoch
    most_correct_entities = checkpointer.checkpoint.most_correct_entities

    pretrained_model = kl_models[0]
    last_epoch_model = kl_models[1]

    steps_so_far = math.ceil(len(train_ds) / args.batch_size) * start_epoch + start_step

    # Resume training for the current epoch
    for _ in range(start_epoch):
        train_ds.update_documents(correct_entities)

    train_loader = create_train_loader(train_ds, args, pad_token_id)

    peft_last_epoch_model = last_epoch_model
    if last_epoch_model is not None:
        if start_epoch > 0 and args.add_kl in ["last-epoch", "both"]:
            peft_last_epoch_model = PeftModel.from_pretrained(
                last_epoch_model,
                os.path.join(output_folder, f"epoch_{start_epoch}"),
                is_trainable=False,
            )
        elif best_epoch is not None and args.add_kl == "best":
            peft_last_epoch_model = PeftModel.from_pretrained(
                last_epoch_model,
                os.path.join(output_folder, f"epoch_{best_epoch}"),
                is_trainable=False,
            )

    for epoch in trange(
        start_epoch,
        args.epochs,
        desc="Epochs",
        initial=start_epoch,
        total=args.epochs,
    ):
        train_iterator = iter(train_loader)
        (train_nnl, train_ppl), train_kl_div_loss, steps_so_far, total_time = (
            train_one_epoch(
                args,
                model,
                optimizer,
                train_iterator,
                train_loader,
                criterion,
                start_step,
                steps_so_far,
                checkpointer,
                output_folder,
                pretrained_model,
                peft_last_epoch_model,
            )
        )

        model.save_pretrained(os.path.join(output_folder, f"epoch_{epoch+1}"))

        validation_probes_results = None
        if args.eft:
            # 1. Evaluate the Model on Validation Probes
            validation_probes_results = evaluate_validation_probes(
                valid_probes_loader=validation_probes_loader,
                model=model,
                tokenizer=tokenizer,
                output_folder=os.path.join(output_folder, f"epoch_{epoch+1}"),
                args=args,
            )

            # 2. Update the correct entities based on the Validation Probes
            correct_entities = {
                entity
                for entity, scores in validation_probes_results["entities"].items()
                if isinstance(scores, dict)
                and scores["recall"] >= args.learning_threshold
            }

            if len(correct_entities) > most_correct_entities:
                most_correct_entities = len(correct_entities)
                best_epoch = epoch + 1

        if last_epoch_model is not None:
            if args.add_kl in ["last-epoch", "both"]:
                print(f"Loading model at Epoch {epoch+1}")
                # load the last epoch model for the next epoch
                peft_last_epoch_model = PeftModel.from_pretrained(
                    last_epoch_model,
                    os.path.join(output_folder, f"epoch_{epoch+1}"),
                    is_trainable=False,
                )
            elif best_epoch is not None and args.add_kl == "best":
                print(f"Loading model at Epoch {best_epoch}")
                # or load the best model for the next epoch
                peft_last_epoch_model = PeftModel.from_pretrained(
                    last_epoch_model,
                    os.path.join(output_folder, f"epoch_{best_epoch}"),
                    is_trainable=False,
                )

        # 3. Update the dataset with the new correct entities
        train_ds.update_documents(correct_entities)
        # 4. Create the new dataloader with the updated dataset
        train_loader = create_train_loader(train_ds, args, pad_token_id)

        # 5. Keep track of the changes in the training stats
        train_stats.append(
            {
                "Epoch": epoch + 1,
                "Best Epoch": best_epoch if args.eft else None,
                "Time": str(total_time),
                "Train NLL": train_nnl,
                "Train PPL": train_ppl,
                "Train KL Div": train_kl_div_loss if args.add_kl else None,
                "Patience": early_stopping.patience - early_stopping.counter,
                "# Correct Entities": len(correct_entities),
                "# Remaining Documents": len(train_ds.training_docs),
                "# Remaining Samples": len(train_ds),
                "Correct Entities": list(correct_entities),
                "Remaining Documents": train_ds.training_docs,
                "Validation Probes Results": validation_probes_results,
            }
        )

        # Update the checkpointer
        checkpointer.update_checkpoint(
            model=model,
            optimizer=optimizer,
            step=0,  # Reset the step
            losses=[],  # Reset the losses
            unmasked_tokens=0,  # Reset the unmasked tokens
            output_folder=output_folder,
            total_time=timedelta(0),  # Reset the total time
            train_stats=train_stats,
            early_stopping=early_stopping,
            epoch=epoch + 1,  # Increment the epoch
            correct_entities=correct_entities,  # 6. Save the correct entities to the checkpoint
            kl_div_losses=[],  # Reset the kl_div_losses
            best_epoch=best_epoch,
            most_correct_entities=most_correct_entities,
        )

        start_step = 0

        with open(os.path.join(output_folder, "train_stats.json"), "w") as f:
            json.dump(train_stats, f, indent=4)


def evaluate_validation_probes(
    valid_probes_loader: DataLoader,
    model: Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizer,
    output_folder: str,
    args: Namespace,
) -> Dict[str, Union[Dict[str, float], Dict[str, Dict[str, float]]]]:

    if args.add_nes:
        generation_file = os.path.join(output_folder, "validation_probes_nes.json")
    else:
        generation_file = os.path.join(output_folder, "validation_probes.json")

    if os.path.isfile(generation_file) and not args.override:
        with open(generation_file, "r") as f:
            generation_results = json.load(f)
    else:
        model.eval()
        with torch.no_grad():
            generation_results = []
            for (
                input_ids,
                attention_mask,
                labels,
                entities,
                ne_labels,
                ne_frequencies,
                question_types,
                original_questions,
            ) in tqdm(
                valid_probes_loader,
                desc=f"Generating Validation Probes answers",
            ):
                input_ids = input_ids.to(model.device)
                attention_mask = attention_mask.to(model.device)

                input_texts = tokenizer.batch_decode(
                    input_ids, skip_special_tokens=True
                )

                output = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k,
                    do_sample=args.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                )

                # get only the generated tokens
                output = output[:, input_ids.size(1) :]
                output_texts = tokenizer.batch_decode(output, skip_special_tokens=True)

                for in_t, out_t, l, ent, ne_l, ne_freq, q_type, original_q in zip(
                    input_texts,
                    output_texts,
                    labels,
                    entities,
                    ne_labels,
                    ne_frequencies,
                    question_types,
                    original_questions,
                ):
                    generation_results.append(
                        {
                            "input": in_t,
                            "prediction": out_t,
                            "reference": l,
                            "entity": ent,
                            "ne_label": ne_l,
                            "ne_frequency": ne_freq,
                            "question_type": q_type,
                            "original_question": original_q,
                        }
                    )

        with open(generation_file, "w") as f:
            json.dump(generation_results, f, indent=4)

    low_preds, low_refs = [], []
    high_preds, high_refs = [], []
    preds_per_entity = {}
    refs_per_entity = {}
    for res in generation_results:
        entity = res["entity"]
        if entity not in preds_per_entity:
            preds_per_entity[entity] = []
            refs_per_entity[entity] = []
        preds_per_entity[entity].append(res["prediction"])
        refs_per_entity[entity].append(res["reference"])

        if res["ne_frequency"] == "low":
            low_preds.append(res["prediction"])
            low_refs.append(res["reference"])
        elif res["ne_frequency"] == "high":
            high_preds.append(res["prediction"])
            high_refs.append(res["reference"])
        else:
            raise ValueError(f"Invalid NE frequency: {res['ne_frequency']}")

    scorer = RougeScorer(["rouge1"], use_stemmer=False)

    validation_probes_results: Dict[
        str, Union[Dict[str, float], Dict[str, Dict[str, float]]]
    ] = {}
    for freq, predictions, references in [
        ("low", low_preds, low_refs),
        ("high", high_preds, high_refs),
        ("avg", low_preds + high_preds, low_refs + high_refs),
    ]:
        scores = compute_rouge(scorer, predictions, references)
        validation_probes_results[freq] = scores

    key = f">={args.learning_threshold}"
    validation_probes_results["entities"] = {key: 0}
    for entity in refs_per_entity:
        refs = refs_per_entity[entity]
        preds = preds_per_entity[entity]
        scores = compute_rouge(scorer, preds, refs)
        validation_probes_results["entities"][entity] = scores  # type: ignore
        if scores["recall"] >= args.learning_threshold:  # type: ignore
            validation_probes_results["entities"][key] += 1

    if args.add_nes:
        results_file = os.path.join(output_folder, "validation_probes_nes_results.json")
    else:
        results_file = os.path.join(output_folder, "validation_probes_results.json")
    with open(results_file, "w") as f:
        json.dump(validation_probes_results, f, indent=4)

    return validation_probes_results


def evaluate_control_probes(
    generation_results: list,
    control_probes_loader: DataLoader,
    model: Union[PreTrainedModel, PeftModel],
    tokenizer: PreTrainedTokenizer,
    output_folder: str,
    args: Namespace,
) -> None:

    model.eval()
    with torch.no_grad():
        for (
            input_ids,
            attention_mask,
            labels,
            question_ids,
            question_types,
            original_datasets,
        ) in tqdm(
            control_probes_loader,
            desc=f"Generating Control Probes answers",
        ):
            input_ids = input_ids.to(model.device)
            attention_mask = attention_mask.to(model.device)

            input_text = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

            output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                do_sample=args.do_sample,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            # get only the generated tokens
            output = output[:, input_ids.size(1) :]
            output_text = tokenizer.batch_decode(output, skip_special_tokens=True)

            for in_t, out_t, l, q_id, q_type, original_dataset in zip(
                input_text,
                output_text,
                labels,
                question_ids,
                question_types,
                original_datasets,
            ):
                generation_results.append(
                    {
                        # "input": in_t,
                        "prediction": out_t,
                        "reference": l,
                        "question_id": q_id,
                        "question_type": q_type,
                        "original_dataset": original_dataset,
                    }
                )

        with open(os.path.join(output_folder, "control_probes.json"), "w") as f:
            json.dump(generation_results, f, indent=4)

        control_probes_results = {}
        # metric for control probes
        em = evaluate.load("exact_match")

        # math
        references = []
        predictions = []
        for res in generation_results:
            if res["original_dataset"] == "math":
                references.append(res["reference"])
                pred = res["prediction"]
                if re.search(r"[0-9]+", pred) is not None:
                    pred = re.search(r"[0-9]+", pred).group()  # type: ignore
                predictions.append(pred)

        control_probes_results["math"] = em.compute(
            predictions=predictions, references=references
        )

        # MMLU
        references = []
        predictions = []
        for res in generation_results:
            if res["original_dataset"] == "MMLU":
                references.append(res["reference"])
                pred = res["prediction"]
                if re.search(r"[ABCD]", pred) is not None:
                    pred = str(ord(re.search(r"[ABCD]", pred).group()) - ord("A"))  # type: ignore
                predictions.append(pred)

        control_probes_results["MMLU"] = em.compute(
            predictions=predictions, references=references
        )

        references = []
        predictions = []
        for res in generation_results:
            if res["original_dataset"] == "SocialIQA":
                references.append(res["reference"])
                pred = res["prediction"]
                if re.search(r"[ABC]", pred) is not None:
                    pred = str(ord(re.search(r"[ABC]", pred).group()) - ord("A") + 1)  # type: ignore
                predictions.append(pred)

        control_probes_results["SocialIQA"] = em.compute(
            predictions=predictions, references=references
        )

        with open(os.path.join(output_folder, "control_probes_results.json"), "w") as f:
            json.dump(control_probes_results, f, indent=4)
