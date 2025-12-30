import argparse
import re
from argparse import Namespace
from pathlib import Path

from bs4 import BeautifulSoup
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
        prog="python utils/clean_docs.py",
        description="Program that will extract the textual content of the HTML documents.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # arguments of the parser
    parser.add_argument(
        "--doc-dir",
        metavar="DOC_DIR",
        type=str,
        default="docs",
        help="Path to the directory containing the HTML documents.",
    )
    parser.add_argument(
        "--out-dir",
        metavar="OUT_DIR",
        type=str,
        default="docs",
        help="Path to the output directory.",
    )

    return parser.parse_args()


def process_documents(input_dir, output_dir):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list(input_dir.glob("*.html"))
    for file_path in tqdm(files):
        output_file_path = f"{output_dir / file_path.stem}.txt"

        with open(file_path, "r") as file:
            soup = BeautifulSoup(file, "html.parser")
            title = soup.find(class_="firstHeading mw-first-heading")
            assert title is not None, f"Title not found in {file_path}"
            title = title.text
            body = [
                text
                for text in soup.find("div", class_="mw-body-content").find_all(  # type: ignore
                    ["p", re.compile(r"^h[1-6]$")]
                )
                if not text.find_parent("table")
            ]
            # remove empty headers
            valid_p = [body[0]]
            for i in range(1, len(body)):
                if body[i].name in ["h1", "h2", "h3", "h4", "h5", "h6"] and body[
                    i - 1
                ].name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                    valid_p.pop()
                valid_p.append(body[i])

            while valid_p[-1].name in ["h1", "h2", "h3", "h4", "h5", "h6"]:
                valid_p.pop()

            # extract the title and the text and join them using a new line
            text = "\n".join([title] + [p.text for p in valid_p])  # type: ignore
            # remove multiple spaces but keep new lines
            text = re.sub(r"[\r\t\f\v ]+", " ", text)
            # keep only one new line
            text = re.sub(r"\n+", "\n", text)
            # remove references
            text = re.sub(r"\[.+?\]", "", text)
            # remove leading and trailing white spaces
            text = text.strip()

        with open(output_file_path, "w") as output_file:
            output_file.write(text)


def main(args):
    process_documents(args.doc_dir, args.out_dir)


if __name__ == "__main__":
    args = get_args()
    main(args)
