import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformer_lens import HookedTransformer

from utils import shuffle_lists_together

TEMPLATE = "The name of the current head of state of {} is"

CLEAN_SUBJECTS = [
    "USA",
    "England",
    "France",
    "Germany",
    "Austria",
    "Canada",
    "India",
    "Japan",
    "Italy",
    "China",
    "Russia",
    "Spain",
    "Poland",
    "Ireland",
    "Scotland",
    "Iran",
]

CORRUPTED_SUBJECTS = CLEAN_SUBJECTS[1:] + [CLEAN_SUBJECTS[0]]

CORRECT_LABELS = [
    " Donald Trump",
    " Charles III",
    " Emmanuel Macron",
    " Frank-Walter Steinmeier",
    " Alexander Van der Bellen",
    " Charles III",
    " Droupadi Murmu",
    " Naruhito",
    " Sergio Mattarella",
    " Xi Jinping",
    " Vladimir Putin",
    " Felipe VI of Spain",
    " Karol Nawrocki",
    " Catherine Connolly",
    " Charles III",
    " Masoud Pezeshkian",
]

INCORRECT_LABELS = CORRECT_LABELS[1:] + [CORRECT_LABELS[0]]


def create_data(
    model: HookedTransformer,
    clean_subjects=CLEAN_SUBJECTS,
    corrupted_subjects=CORRUPTED_SUBJECTS,
    correct_labels=CORRECT_LABELS,
    incorrect_labels=INCORRECT_LABELS,
    template=TEMPLATE,
):
    for i in range(len(clean_subjects)):
        clean_subject = clean_subjects[i]
        corrupted_subject = corrupted_subjects[i]
        clean_tok = model.to_str_tokens(template.format(clean_subject))
        corrupted_tok = model.to_str_tokens(template.format(corrupted_subject))
        assert len(clean_tok) == len(
            corrupted_tok
        ), f"Length mismatch for {clean_tok} and {corrupted_tok}"

    random.seed(42)

    correct_idxs = []
    incorrect_idxs = []
    clean = []
    corrupted = []

    for i in range(len(correct_labels)):
        correct_idx = model.tokenizer(correct_labels[i], add_special_tokens=False).input_ids[0]  # type: ignore
        correct_idxs.append(correct_idx)
        incorrect_idx = model.tokenizer(incorrect_labels[i], add_special_tokens=False).input_ids[  # type: ignore
            0
        ]
        incorrect_idxs.append(incorrect_idx)
        clean.append(template.format(clean_subjects[i]))
        corrupted.append(template.format(corrupted_subjects[i]))

    clean, correct_idxs, corrupted, incorrect_idxs = shuffle_lists_together(
        clean, correct_idxs, corrupted, incorrect_idxs
    )

    return clean, correct_idxs, corrupted, incorrect_idxs


def save_data(clean, correct_idxs, corrupted, incorrect_idxs, filepath: str):
    dataset = {k: [] for k in ["clean", "correct_idx", "corrupted", "incorrect_idx"]}
    for k, v in zip(
        ["clean", "correct_idx", "corrupted", "incorrect_idx"],
        [clean, correct_idxs, corrupted, incorrect_idxs],
    ):
        dataset[k].extend(v)
    df2 = pd.DataFrame.from_dict(dataset)
    df2.to_csv(filepath, index=False)


def collate_EAP(xs):
    clean, corrupted, labels = zip(*xs)
    clean = list(clean)
    corrupted = list(corrupted)
    labels = torch.tensor(labels)
    return clean, corrupted, labels


class EAPDataset(Dataset):
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath)

    def __len__(self):
        return len(self.df)

    def shuffle(self):
        self.df = self.df.sample(frac=1)

    def head(self, n: int):
        self.df = self.df.head(n)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return (
            row["clean"],
            row["corrupted"],
            [row["correct_idx"], row["incorrect_idx"]],
        )

    def to_dataloader(self, batch_size: int, shuffle: bool = False) -> DataLoader:
        return DataLoader(
            self, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_EAP
        )
