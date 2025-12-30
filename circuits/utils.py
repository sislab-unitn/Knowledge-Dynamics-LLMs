import random

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


MAP_MODELS = {
    "llama3-i": "meta-llama/Llama-3.1-8B-Instruct",
    "llama3-i-1b": "meta-llama/Llama-3.2-1B-Instruct",
}


def shuffle_lists_together(*lists):
    """
    Shuffle multiple lists while keeping the same order for each.
    All lists must have the same length.
    """
    if not lists:
        return lists

    # Check all lists have the same length
    length = len(lists[0])
    if not all(len(lst) == length for lst in lists):
        raise ValueError("All lists must have the same length")

    # Create indices and shuffle them
    indices = list(range(length))
    random.shuffle(indices)

    # Apply the same shuffled order to all lists
    shuffled_lists = tuple([lst[i] for i in indices] for lst in lists)

    return shuffled_lists


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

    def to_dataloader(self, batch_size: int):
        return DataLoader(self, batch_size=batch_size, collate_fn=collate_EAP)
