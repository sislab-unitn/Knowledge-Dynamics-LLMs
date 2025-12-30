import json
import gzip
import os
import re
from typing import Dict, List, Tuple

from tqdm.rich import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizer


VALID_LABELS = [
    "CARDINAL",
    "DATE",
    "EVENT",
    "FAC",
    "GPE",
    "LANGUAGE",
    "LAW",
    "LOC",
    "MONEY",
    "NORP",
    "ORDINAL",
    "ORG",
    "PERCENT",
    "PERSON",
    "PRODUCT",
    "QUANTITY",
    "TIME",
    "WORK_OF_ART",
]
SUBSET_LABELS = ["PERSON", "ORG", "NORP", "GPE"]


def create_samples(
    tokenizer: PreTrainedTokenizer,
    sample: List[int],
    valid_labels: List[str],
    subset_labels: List[str],
) -> Tuple[str, str]:
    assert len(sample) <= max_tokens, "The sample is too long"
    tagged_sent_full = tokenizer.decode(sample)

    tagged_sent_sub = ""
    original_sent = ""
    sent = tagged_sent_full
    while re.search(r"<[A-Z_]*?> .*? </>", sent) is not None:
        matched = re.search(r"<[A-Z_]*?> .*? </>", sent)
        label_tag = re.search(r"<[A-Z_]*?> ", matched.group())  # type: ignore
        label = label_tag.group()[1:-2]  # type: ignore
        start, end = matched.span()  # type: ignore
        terminator = re.search(r" </>", matched.group())  # type: ignore
        _, end_l = label_tag.span()  # type: ignore
        start_t, _ = terminator.span()  # type: ignore

        if label in subset_labels:
            # if the label is relevant, we keep the text with the tags
            to_add = sent[:end]
        else:
            # otherwise we keep only the text
            to_add = sent[:start] + matched.group()[end_l:start_t]  # type: ignore
        tagged_sent_sub += to_add

        if label in valid_labels:
            # if the label was added in this scenario, we discard the tags and keep the text
            original_sent += sent[:start] + matched.group()[end_l:start_t]  # type: ignore
        else:
            # otherwise we keep the whole thing because it was in the original text
            original_sent += sent[:end]

        sent = sent[end:]

    tagged_sent_sub += sent
    sub_ne_sample = tokenizer.encode(tagged_sent_sub, add_special_tokens=False)
    assert len(sub_ne_sample) <= max_tokens, "The sub_ne_sample is too long"

    original_sent += sent
    no_ne_sample = tokenizer.encode(original_sent, add_special_tokens=False)
    assert len(no_ne_sample) <= max_tokens, "The no_ne_sample is too long"

    return tagged_sent_sub, original_sent


def main(
    max_tokens: int,
    tokenizer: PreTrainedTokenizer,
    data_folder: str,
    valid_labels: List[str] = VALID_LABELS,
    subset_labels: List[str] = SUBSET_LABELS,
):
    with gzip.open(os.path.join(data_folder, "docs.json.gz"), "r") as f:
        docs = json.load(f)

    with gzip.open(os.path.join(data_folder, "nes_per_doc.json.gz"), "r") as f:
        nes_per_doc = json.load(f)

    # set containing no named entities
    no_ne: Dict[str, List[str]] = {}
    # set containing all named entities
    all_ne: Dict[str, List[str]] = {}
    # set containing only "PERSON", "ORG", "NORP", "GPE" named entities
    sub_ne: Dict[str, List[str]] = {}
    for doc_id, doc in tqdm(docs.items(), desc="Processing documents"):
        ents = nes_per_doc[doc_id]

        for start, end, ent_label in ents[::-1]:
            doc = (
                doc[:start] + f"<{ent_label}> " + doc[start:end] + f"  </>" + doc[end:]
            )

        doc = re.sub(" +", " ", doc)
        words = doc.split(" ")
        words = [words[0]] + [f" {w}" for w in words[1:]]

        # further split the words to make sure special tokens are consider individually
        valid_words = []
        for w in words:
            matched = re.search(r"<.*?>", w)
            while matched is not None:
                start, end = matched.span()
                valid_words += [w[:start], w[start:end]]
                w = w[end:]
                matched = re.search(r"<.*?>", w)
            valid_words.append(w)
        words = [w for w in valid_words if w != ""]

        assert "".join(words) == doc, "The split is not correct"

        # initialize the lists
        all_ne[doc_id] = []
        sub_ne[doc_id] = []
        no_ne[doc_id] = []

        sample = []
        i = 0
        while i < len(words):
            w = words[i]
            matched = re.search(r"<[A-Z_]*?>", w)

            # named entity, we need to encode the whole entity
            if matched is not None and matched.group()[1:-1] in valid_labels:
                encoded_word = []
                # encode until the end of the entity
                while re.search(r"</>", w) is None:
                    encoded_word += tokenizer.encode(w, add_special_tokens=False)
                    i += 1
                    w = words[i]

                # add the terminator
                encoded_word += tokenizer.encode(w, add_special_tokens=False)
            else:
                # add a word at a time, checkinkg if it fits
                encoded_word = tokenizer.encode(w, add_special_tokens=False)

            if len(sample) + len(encoded_word) > max_tokens:
                sub_ne_sent, no_ne_sent = create_samples(
                    tokenizer,
                    sample,
                    valid_labels=valid_labels,
                    subset_labels=subset_labels,
                )
                all_ne[doc_id].append(tokenizer.decode(sample))
                sub_ne[doc_id].append(sub_ne_sent)
                no_ne[doc_id].append(no_ne_sent)

                # reset the sample
                sample = []

            sample += encoded_word
            i += 1

        if len(sample) > 0:
            sub_ne_sent, no_ne_sent = create_samples(
                tokenizer,
                sample,
                valid_labels=valid_labels,
                subset_labels=subset_labels,
            )
            all_ne[doc_id].append(tokenizer.decode(sample))
            sub_ne[doc_id].append(sub_ne_sent)
            no_ne[doc_id].append(no_ne_sent)

    with gzip.open(os.path.join(data_folder, "train_all_ne.json.gz"), "wt") as f:
        json.dump(all_ne, f, indent=4)

    with gzip.open(os.path.join(data_folder, "train_sub_ne.json.gz"), "wt") as f:
        json.dump(sub_ne, f, indent=4)

    with gzip.open(os.path.join(data_folder, "train_no_ne.json.gz"), "wt") as f:
        json.dump(no_ne, f, indent=4)


if __name__ == "__main__":

    tok = AutoTokenizer.from_pretrained("allenai/OLMo-7B-0724-Instruct-hf")
    max_tokens = 2048
    main(
        max_tokens=max_tokens,
        tokenizer=tok,  # type: ignore
        data_folder="data",
        valid_labels=VALID_LABELS,
        subset_labels=SUBSET_LABELS,
    )
