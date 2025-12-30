import argparse
import glob
import io
import json
import os
import re
import subprocess
from argparse import Namespace
from collections import Counter
from functools import partial
from multiprocessing import Pool, Manager
from multiprocessing.managers import DictProxy
from typing import List

import spacy
from tqdm import tqdm

# Global variable to hold the spaCy model
nlp = None


def init_model():
    """Initialize the global spaCy model."""
    global nlp
    nlp = spacy.load("en_core_web_sm")  # Load the model once
    nlp.max_length = 10000000  # Increase the max length of the model


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python utils/process_wiki_ne.py",
        description="Main module.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--data-folder",
        metavar="DATA_FOLDER",
        type=str,
        default="Dolma",
        help="Path to the folder containing the original data.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="data",
        help="Path to the output directory.",
    )
    parser.add_argument(
        "--n-processes",
        metavar="N_PROCESSES",
        type=int,
        default=20,
        help="Number of processes to use.",
    )

    return parser.parse_args()


def process_document(
    line: str,
    relevant_documents: DictProxy,
    file_name: str,
):
    global nlp
    id = re.search(r'"id":.*?(,|{)', line)
    assert id is not None, f"ID not found in {line}"
    id = id.group()[6:-2]

    url = re.search(r'"url":.*?(,|})', line)
    assert url is not None, f"URL not found in {line}"
    url = url.group()[7:-2]

    # extract the content of the document
    text = re.search(r'"text":.*?"(,"|})', line)
    assert text is not None, f"Text not found in {line}"
    text = text.group()[8:-3] if text.group().endswith('"') else text.group()[8:-2]

    # search for the title, to make an exact match
    title = re.search(r".*?\\n\\n?", text, flags=re.M)
    assert title is not None, f"Title not found in {line}"
    title = title.group()[:-4]
    text = text.replace("\\\\", "\\")
    text = text.replace("\\n", "\n")
    text = text.replace('\\"', '"')
    relevant_documents[f"{file_name}-{id}"] = {
        "url": url,
        "title": title,
        "named_entities": Counter(
            [
                f"{(entity.text, entity.label_)}"
                for entity in nlp(text[nlp.max_length]).ents  # type: ignore
            ]
        ),
    }


def process_gz(
    files: List[str],
    n_processes: int,
    out_dir: str,
):
    os.makedirs(out_dir, exist_ok=True)
    with Manager() as manager:
        relevant_documents = manager.dict()

        for file in tqdm(files, desc="Processing GZ Files"):
            p = subprocess.Popen(
                ["zcat", file], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
            )
            lines = io.TextIOWrapper(p.stdout, encoding="utf-8")  # type: ignore

            lines = [line for line in lines]

            with Pool(n_processes, initializer=init_model) as p:
                r = list(
                    tqdm(
                        p.imap(
                            partial(
                                process_document,
                                relevant_documents=relevant_documents,
                                file_name=file.split("/")[-1],
                            ),
                            lines,
                        ),
                        total=len(lines),
                        desc="Processing JSON Lines",
                        # leave=False,
                    )
                )

        with open(os.path.join(out_dir, "ne.json"), "w") as f:
            json.dump(relevant_documents._getvalue(), f, indent=2)


def main(args: Namespace):
    process_gz(
        files=glob.glob(f"{args.data_folder}/*.json.gz"),
        n_processes=args.n_processes,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    args = get_args()
    main(args)
