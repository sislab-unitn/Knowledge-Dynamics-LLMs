import argparse
import gzip
import json
from argparse import Namespace
from pathlib import Path

import spacy
from tqdm.rich import tqdm


def get_args() -> Namespace:
    """
    Parse command line arguments.

    Returns
    -------
    parsed_args: Namespace instance
        Parsed arguments passed through command line.
    """

    parser = argparse.ArgumentParser(
        prog="python utils/extract_ne.py",
        description="Opens the documents, parse them with SpaCy to extract NEs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--doc-dir",
        metavar="DOC_DIR",
        type=str,
        default="data/docs",
        help="Path to the directory containing the .txt documents.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="data",
        help="Path to the output directory.",
    )

    return parser.parse_args()


def main(args):
    input_dir = Path(args.doc_dir)
    output_dir = Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    spacy.require_gpu()  # type: ignore
    nlp = spacy.load("en_core_web_trf")
    nlp.max_length = 10000000  # Increase the max length of the model

    with gzip.open("data/train.json.gz", "r") as f:
        data = json.load(f)

    ents_per_doc = {}
    files = list(input_dir.glob("*.txt"))
    for file_path in tqdm(files, desc="Processing documents"):
        with open(file_path, "r", encoding="utf-8") as file:
            text = file.read()
            assert (
                text == data[file_path.stem]
            ), "The text from the original document is not the one in the compressed data."

            ents = []
            modified_text = text
            for ent in nlp(text).ents[::-1]:
                ents.append((ent.start_char, ent.end_char, str(ent.label_)))
                modified_text = (
                    modified_text[: ent.start_char]
                    + f"<{str(ent.label_)}>"
                    + str(ent)
                    + "</>"
                    + modified_text[ent.end_char :]
                )

            if len(ents) > 0:
                new_text = data[file_path.stem]

                for start, end, ent_label in ents:
                    new_text = (
                        new_text[:start]
                        + f"<{ent_label}>"
                        + new_text[start:end]
                        + f"</>"
                        + new_text[end:]
                    )

                assert (
                    new_text == modified_text
                ), "Something is wrong with the technique to copy the entities."

            assert file_path.stem not in ents_per_doc, "Document already processed."
            ents_per_doc[file_path.stem] = ents[::-1]

    # Save the data
    with gzip.open(output_dir / "ents_per_doc.json.gz", "wt") as f:
        json.dump(ents_per_doc, f, indent=2)


if __name__ == "__main__":
    args = get_args()
    main(args)
