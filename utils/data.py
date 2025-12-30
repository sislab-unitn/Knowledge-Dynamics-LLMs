import random
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from datasets import load_dataset
import torch
from transformers import PreTrainedTokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

# unavailable from wikipedia
UNAVAILABLE_DOCS = [
    "en_simple_wiki_v0-0001.json.gz-56175878",
    "en_simple_wiki_v0-0000.json.gz-32808961",
    "en_simple_wiki_v0-0001.json.gz-72498864",
    "en_simple_wiki_v0-0001.json.gz-73213510",
    "en_simple_wiki_v0-0001.json.gz-45577729",
    "en_simple_wiki_v0-0001.json.gz-11277190",
    "en_simple_wiki_v0-0001.json.gz-930826",
    "en_simple_wiki_v0-0000.json.gz-20006714",
    "en_simple_wiki_v0-0000.json.gz-16401475",
    "en_simple_wiki_v0-0000.json.gz-9934421",
    "en_simple_wiki_v0-0000.json.gz-30575720",
    "en_simple_wiki_v0-0000.json.gz-7728160",
    "en_simple_wiki_v0-0001.json.gz-62034570",
    "en_simple_wiki_v0-0001.json.gz-37567473",
    "en_simple_wiki_v0-0000.json.gz-25416724",
]


class ValidationProbesDataset(Dataset):
    def __init__(
        self,
        data: Dict[str, Dict[str, Union[str, List[Dict[str, str]]]]],
    ):

        self.questions: List[str] = []
        self.labels: List[str] = []
        self.entities: List[str] = []
        self.ne_labels: List[str] = []
        self.ne_frequencies: List[str] = []
        self.question_types: List[str] = []
        self.original_questions: List[Union[None, Dict[str, str]]] = []

        for entity, question_types in data.items():
            ne_label = question_types["ne_label"]
            ne_frequency = question_types["ne_frequency"]

            for question_type in [
                "Factual Question",
                "Time-Sensitive Question",
                "Temporal Understanding Question",
                "Entity Linking Question",
                "Rephrased Questions",
            ]:
                if question_type in question_types:
                    questions = question_types[question_type]
                    for q in questions:
                        self.questions.append(q["question"])  # type: ignore
                        self.labels.append(q["answer"])  # type: ignore
                        self.entities.append(entity)
                        self.ne_labels.append(ne_label)  # type: ignore
                        self.ne_frequencies.append(ne_frequency)  # type: ignore
                        self.question_types.append(question_type)
                        if question_type == "Rephrased Questions":
                            self.original_questions.append(
                                {
                                    "question": q["original_question"],  # type: ignore
                                    "question_type": q["question_type"],  # type: ignore
                                    "question_id": q["question_id"],  # type: ignore
                                }
                            )
                        else:
                            self.original_questions.append(None)

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(
        self, idx
    ) -> Tuple[str, str, str, str, str, str, Union[None, Dict[str, str]]]:
        return (
            self.questions[idx],
            self.labels[idx],
            self.entities[idx],
            self.ne_labels[idx],
            self.ne_frequencies[idx],
            self.question_types[idx],
            self.original_questions[idx],
        )


class ValidationProbesGenerationCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        instruction: str = "",
        max_len: int = 4096,  # olmo max length
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.instruction = instruction
        self.max_len = max_len

    def __call__(
        self,
        batch: List[Tuple[str, str, str, str, str, str, Union[None, Dict[str, str]]]],
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        List[str],
        List[str],
        List[str],
        List[str],
        List[str],
        List[Union[None, Dict[str, str]]],
    ]:
        input_ids = []
        labels = []
        entities = []
        ne_labels = []
        ne_frequencies = []
        question_types = []
        original_questions = []

        self.tokenizer.padding_side = "left"

        for (
            question,
            label,
            entity,
            ne_label,
            ne_frequency,
            question_type,
            original_question,
        ) in batch:
            if self.model_name in [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
            ]:
                chat = [
                    {"role": "system", "content": self.instruction},
                    {"role": "user", "content": question},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            elif self.model_name == "allenai/OLMo-7B-0724-Instruct-hf":
                chat = [
                    {"role": "user", "content": f"{self.instruction} {question}"},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            elif self.model_name in [
                "allenai/OLMo-7B-0724-hf",
                "allenai/OLMo-1B-0724-hf",
                "meta-llama/Llama-3.1-8B",
            ]:
                prompt = question
            else:
                raise ValueError(f"{self.model_name} not supported")

            inputs: torch.Tensor = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")  # type: ignore
            inputs = inputs.squeeze()
            assert inputs.size(0) <= self.max_len, "Input length exceeds maximum length"

            input_ids.append(inputs)
            labels.append(label)
            entities.append(entity)
            ne_labels.append(ne_label)
            ne_frequencies.append(ne_frequency)
            question_types.append(question_type)
            original_questions.append(original_question)

        reversed_input_ids = [i.flip([0]) for i in input_ids]
        attention_mask = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=-100
        )
        attention_mask = (attention_mask != -100).long()
        attention_mask = attention_mask.flip([1])
        reversed_input_ids = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        input_ids = reversed_input_ids.flip([1])

        self.tokenizer.padding_side = "right"

        return (
            input_ids,
            attention_mask,
            labels,
            entities,
            ne_labels,
            ne_frequencies,
            question_types,
            original_questions,
        )


class EFTDataset(Dataset):
    def __init__(
        self,
        correct_entities: Set[str],
        docs_per_entity: Dict[str, List[str]],
        docs: Dict[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        valid_doc_ids: Optional[Set[str]],
        add_chat_template: bool = False,
        max_len: int = 4096,  # olmo max length
    ):
        self.docs_per_entity = docs_per_entity
        self.valid_doc_ids: Optional[Set[str]] = valid_doc_ids

        self.tokenized_docs: Dict[str, List[Dict[str, torch.Tensor]]] = {}
        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.doc_ids: List[str] = []
        self.training_docs: List[str] = []

        # documents are tokenized at the beginning and stored in memory
        self._tokenize_documents(
            docs, tokenizer, model_name, add_chat_template, max_len
        )
        self.update_documents(correct_entities)

    def _tokenize_documents(
        self,
        docs: Dict[str, List[str]],
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        add_chat_template: bool,
        max_len: int,
    ):
        for doc_id, samples in tqdm(
            docs.items(), desc="Tokenizing documents", leave=False
        ):
            self.tokenized_docs[doc_id] = []
            for text in samples:
                if add_chat_template:
                    chat = [
                        {"role": "system", "content": text},
                    ]
                    prompt = tokenizer.apply_chat_template(
                        chat, tokenize=False, add_generation_prompt=True
                    )
                    input_id: torch.Tensor = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt").squeeze()  # type: ignore
                else:
                    input_id: torch.Tensor = tokenizer.encode(text, add_special_tokens=True, return_tensors="pt").squeeze()  # type: ignore

                label = input_id[1:]  # remove first token
                input_id = input_id[:-1]  # remove last token

                assert len(label) == len(
                    input_id
                ), "Input and labels have different lengths"

                if "molmo" in model_name.lower():
                    assert (
                        input_id.size(0) <= max_len
                    ), "Input length exceeds maximum length"

                assert torch.equal(
                    input_id[1:], label[:-1]
                ), "Input and labels are not the same"

                self.tokenized_docs[doc_id].append(
                    {"input_ids": input_id, "labels": label}
                )

    def update_documents(self, correct_entities: Set[str]):
        # reset the dataset
        self.input_ids: List[torch.Tensor] = []
        self.labels: List[torch.Tensor] = []
        self.doc_ids: List[str] = []
        self.training_docs: List[str] = []

        docs_set: Set[str] = set()

        # consider all documents for entities that the model has not learned
        for entity, doc_ids in self.docs_per_entity.items():
            if entity not in correct_entities:
                docs_set.update(
                    [doc_id for doc_id in doc_ids if doc_id not in UNAVAILABLE_DOCS]
                )

        doc_ids = list(docs_set)
        self.training_docs = doc_ids

        # shuffle the documents, so that the model does not learn the order
        random.shuffle(doc_ids)

        # prepare the actual dataset
        for doc_id in doc_ids:
            # skip documents that are not valid (i.e., main documents when using --main-docs)
            if self.valid_doc_ids is not None and doc_id not in self.valid_doc_ids:
                continue

            for sample in self.tokenized_docs[doc_id]:
                self.input_ids.append(sample["input_ids"])
                self.labels.append(sample["labels"])
                self.doc_ids.append(doc_id)

    def __len__(self) -> int:
        return len(self.input_ids)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor, str]:
        return self.input_ids[idx], self.labels[idx], self.doc_ids[idx]


class EFTCollator:
    def __init__(
        self,
        pad_token_id: int,
    ):
        self.pad_token_id = pad_token_id

    def __call__(
        self, batch: List[Tuple[torch.Tensor, torch.Tensor, List[str]]]
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        doc_ids = []
        input_ids = []
        labels = []

        for input_id, label, doc_id in batch:
            doc_ids.append(doc_id)
            input_ids.append(input_id)
            labels.append(label)

        input_ids = pad_sequence(
            input_ids, batch_first=True, padding_value=self.pad_token_id
        )
        labels = pad_sequence(labels, batch_first=True, padding_value=-100)

        assert input_ids.size(1) == labels.size(
            1
        ), "Input and labels have different lengths"

        return input_ids, labels, doc_ids


class ControlProbesDataset(Dataset):
    def __init__(
        self,
        math_probes: Dict[str, Dict[str, Any]],
        reasoning_probes: List[Dict[str, str]],
        already_generated_answers: Set[str],
    ):

        self.questions: List[str] = []
        self.answers: List[str] = []
        self.question_types: List[str] = []
        self.question_ids: List[str] = []
        self.original_datasets: List[str] = []

        for q_id, question in math_probes.items():
            q_id = f"math_{q_id}"
            if q_id not in already_generated_answers:
                self.questions.append(question["question"])
                self.answers.append(str(question["answer"]))
                self.question_types.append(question["type"])
                self.question_ids.append(q_id)
                self.original_datasets.append("math")

        mmlu = load_dataset("cais/mmlu", "all", split="test")
        for q_id, sample in enumerate(mmlu.map(self._format_question)):
            q_id = f"MMLU_{q_id}"
            if q_id not in already_generated_answers:
                self.questions.append(sample["formatted_question"])
                self.answers.append(str(sample["answer"]))
                self.question_types.append(sample["subject"])
                self.question_ids.append(q_id)
                self.original_datasets.append("MMLU")

        for q_id, sample in enumerate(reasoning_probes):
            q_id = f"SocialIQA_{q_id}"
            if q_id not in already_generated_answers:
                choices = [sample["answerA"], sample["answerB"], sample["answerC"]]
                question = f"Context: {sample['context']} Q: {sample['question']}\n"
                question += "\n".join(
                    [
                        f"{chr(ord('A') + i)}. {choice}"
                        for i, choice in enumerate(choices)
                    ]
                )
                self.questions.append(question)
                self.answers.append(sample["answer"])
                self.question_types.append(sample["type"])
                self.question_ids.append(q_id)
                self.original_datasets.append("SocialIQA")

    def _format_question(self, example):
        # Create a formatted question with answer choices
        formatted_q = f"Q: {example['question']}\n"
        formatted_q += "\n".join(
            [
                f"{chr(ord('A') + i)}. {choice}"
                for i, choice in enumerate(example["choices"])
            ]
        )

        # Return the new field
        return {"formatted_question": formatted_q}

    def __len__(self) -> int:
        return len(self.questions)

    def __getitem__(self, idx) -> Tuple[str, str, str, str, str]:
        return (
            self.questions[idx],
            self.answers[idx],
            self.question_ids[idx],
            self.question_types[idx],
            self.original_datasets[idx],
        )


class ControlProbesGenerationCollator:
    def __init__(
        self,
        tokenizer: PreTrainedTokenizer,
        model_name: str,
        max_len: int = 4096,  # olmo max length
    ):
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.max_len = max_len

    def __call__(
        self,
        batch: List[Tuple[str, str, str, str, str]],
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[str], List[str], List[str]]:
        input_ids = []
        labels = []
        question_ids = []
        question_types = []
        original_datasets = []

        self.tokenizer.padding_side = "left"

        for q, ans, q_id, q_type, original_dataset in batch:
            if original_dataset == "math":
                instruction = "Be brief. Do not add any explanation. Only provide the final result without the equation."
            elif original_dataset == "MMLU":
                instruction = "Be brief. Do not add any explanation. Only answer with 'A', 'B', 'C', or 'D'."
            elif original_dataset == "SocialIQA":
                instruction = "Be brief. Do not add any explanation. Only answer with 'A', 'B', or 'C'."
            else:
                raise ValueError(f"{original_dataset} not supported")

            if self.model_name in [
                "meta-llama/Llama-3.1-8B-Instruct",
                "meta-llama/Llama-3.2-1B-Instruct",
            ]:
                chat = [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": q},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            elif self.model_name == "allenai/OLMo-7B-0724-Instruct-hf":
                chat = [
                    {"role": "user", "content": f"{instruction} {q}"},
                ]
                prompt = self.tokenizer.apply_chat_template(
                    chat, tokenize=False, add_generation_prompt=True
                )
            elif self.model_name in [
                "allenai/OLMo-1B-0724-hf",
                "allenai/OLMo-7B-0724-hf",
                "meta-llama/Llama-3.1-8B",
            ]:
                prompt = q
            else:
                raise ValueError(f"{self.model_name} not supported")

            inputs: torch.Tensor = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")  # type: ignore
            inputs = inputs.squeeze()
            assert inputs.size(0) <= self.max_len, "Input length exceeds maximum length"

            input_ids.append(inputs)
            labels.append(ans)
            question_ids.append(q_id)
            question_types.append(q_type)
            original_datasets.append(original_dataset)

        reversed_input_ids = [i.flip([0]) for i in input_ids]
        attention_mask = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=-100
        )
        attention_mask = (attention_mask != -100).long()
        attention_mask = attention_mask.flip([1])
        reversed_input_ids = pad_sequence(
            reversed_input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id  # type: ignore
        )
        input_ids = reversed_input_ids.flip([1])

        self.tokenizer.padding_side = "right"

        return (
            input_ids,
            attention_mask,
            labels,
            question_ids,
            question_types,
            original_datasets,
        )
