import argparse
import json
import os
import requests
import time
from argparse import Namespace
from pathlib import Path
from multiprocessing import Pool, Manager
from functools import partial
from typing import Tuple
from multiprocessing.managers import ListProxy

from tqdm import tqdm


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python utils/collect_up_to_date_docs.py",
        description="Program that will download the HTML pages for the list of accepted documents for each entity.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--input-file",
        metavar="INPUT_FILE",
        type=str,
        default="analysis/accepted_docs_per_entity.json",
        help="JSON file containing the accepted docs for each entity.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="docs",
        help="Path to the output directory.",
    )

    return parser.parse_args()


def download_html(doc_info: Tuple[str, str], failed_to_download: ListProxy) -> None:
    doc_id, url = doc_info
    try:
        response = requests.get(url)
        if response.status_code == 200:
            text = response.text
            with open(f"{args.out_dir}/{doc_id}.html", "w") as file:
                file.write(text)
        else:
            failed_to_download.append(doc_info)
    except Exception as e:
        failed_to_download.append(doc_info)
    time.sleep(1)


def main(args):
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.input_file, "r") as file:
        accepted_docs = json.load(file)

    if Path(f"{args.out_dir}/to_download.json").exists():
        with open(f"{args.out_dir}/to_download.json", "r") as file:
            to_download = json.load(file)
            to_download = set((doc_id, url) for doc_id, url in to_download)
    else:
        to_download = set()
        for docs in accepted_docs.values():
            for doc_id, info in docs:
                to_download.add((doc_id, info["url"]))

    n_processes = 50
    with Manager() as manager:
        failed_to_download = manager.list()
        with Pool(n_processes) as p:
            r = list(
                tqdm(
                    p.imap(
                        partial(
                            download_html,
                            failed_to_download=failed_to_download,
                        ),
                        to_download,
                    ),
                    total=len(to_download),
                    desc="Downloading HTML pages",
                )
            )

        with open(f"{args.out_dir}/to_download.json", "w") as file:
            json.dump(failed_to_download._getvalue(), file, indent=4)


if __name__ == "__main__":
    args = get_args()
    main(args)
