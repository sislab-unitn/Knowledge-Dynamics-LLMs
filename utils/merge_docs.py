import gzip
import json
from pathlib import Path

from tqdm.rich import tqdm

txt_files = list(Path("data/docs").glob("*.txt"))

training_docs = {}
for txt_file in tqdm(txt_files, desc="Compressing docs into a single file"):
    with open(txt_file, "r") as f:
        doc = f.read()
    entity = txt_file.stem
    assert entity not in training_docs, f"Duplicate entity {entity}"
    training_docs[entity] = doc

with gzip.open("data/training_docs.json.gz", "wt", encoding="utf-8") as f:
    json.dump(training_docs, f)
