from typing import Optional

import datasets
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from transformers import AutoTokenizer


class SuperGLUEDataModule(LightningDataModule):

    task_text_field_map = {
        "boolq": ("question", "passage"),
        "cb": ("premise", "hypothesis"),
        "rte": ("premise", "hypothesis"),
        "wic": ("sentence1", "sentence2", "word"),
        "wsc": ("text", "span1_text", "span2_text"),
        "copa": ("premise", "question", "choice1", "choice2"),
        "multirc": ("paragraph", "question", "answer"),
        "record": ("passage", "query", "entities"),
    }

    super_glue_task_types = {
        "boolq": "classification",
        "cb": "classification",
        "rte": "classification",
        "wic": "classification",
        "wsc": "_",
        "copa": "multiple_choice",
        "multirc": "classification",
    }

    super_glue_tasks_num_labels = {
        "boolq": 2,
        "cb": 3,
        "rte": 2,
        "wic": 2,
        "wsc": 2,
        "copa": 2,
        "multirc": 2,
        "record": 0,
    }
    
    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'labels'
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "boolq",
        max_seq_length: Optional[int] = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 2,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers

        self.task_type = self.super_glue_task_types[task_name]
        self.text_fields = self.task_text_field_map[task_name]
        self.num_labels = self.super_glue_tasks_num_labels[task_name]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = datasets.load_dataset("super_glue", self.task_name)

        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=["label"],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("super_glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]
    
    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]
    
    def convert_to_features(self, example_batch, indices=None):
        if self.task_name == "copa":
            text = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]], example_batch[self.text_fields[2]], example_batch[self.text_fields[3]]))
        elif self.task_name == "wic":
            text = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]], example_batch[self.text_fields[2]]))
        else:
            text = list(zip(example_batch[self.text_fields[0]], example_batch[self.text_fields[1]]))

        features = self.tokenizer.batch_encode_plus(text, padding='max_length', truncation=True)
        features["labels"] = example_batch["label"]
        return features

class SuperGLUEDataModule_record(LightningDataModule):

    task_text_field_map = {
        "record": ("passage", "query", "entities"),
    }

    loader_columns = [
        'datasets_idx',
        'input_ids',
        'token_type_ids',
        'attention_mask',
        'labels',
    ]

    def __init__(
        self,
        model_name_or_path: str,
        task_name: str = "record",
        max_seq_length: Optional[int] = 512,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        num_workers: int = 8,
        **kwargs,
    ):
        super().__init__()
        self.model_name_or_path = model_name_or_path
        self.task_name = task_name
        self.max_seq_length = max_seq_length
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.num_labels = 2
        self.text_fields = self.task_text_field_map[task_name]

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def setup(self, stage: str):
        self.dataset = {}
        self.dataset['train'] = datasets.load_dataset("super_glue", self.task_name, split="train[:1%]")
        self.dataset['validation'] = datasets.load_dataset("super_glue", self.task_name, split="validation[:5%]")
        self.dataset['test'] = datasets.load_dataset("super_glue", self.task_name, split="test[:5%]")
        for split in self.dataset.keys():
            self.dataset[split] = self.dataset[split].map(
                self.unbatch,
                batched=True,
                remove_columns=["entity_spans"],
            )
            self.dataset[split] = self.dataset[split].map(
                self.convert_to_features,
                batched=True,
                remove_columns=[],
            )
            self.columns = [c for c in self.dataset[split].column_names if c in self.loader_columns]
            self.dataset[split].set_format(type="torch", columns=self.columns)
        
        self.eval_splits = [x for x in self.dataset.keys() if "validation" in x]

    def prepare_data(self):
        datasets.load_dataset("super_glue", self.task_name)
        AutoTokenizer.from_pretrained(self.model_name_or_path, use_fast=True)

    def train_dataloader(self):
        return DataLoader(self.dataset["train"], batch_size=self.train_batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["validation"], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]
    
    def test_dataloader(self):
        if len(self.eval_splits) == 1:
            return DataLoader(self.dataset["test"], batch_size=self.eval_batch_size, num_workers=self.num_workers)
        elif len(self.eval_splits) > 1:
            return [DataLoader(self.dataset[x], batch_size=self.eval_batch_size, num_workers=self.num_workers) for x in self.eval_splits]

    def convert_to_features(self, example_batch, indices=None):
        features = self.tokenizer.batch_encode_plus(example_batch['text'], padding='max_length', truncation=True)
        features["labels"] = example_batch["labels"]
        return features
    
    def unbatch(self, example_batch, indices=None):
        import collections
        import re
        new_batch = collections.defaultdict(list)
        keys = example_batch.keys()
        for values in zip(*example_batch.values()):
            ex = {k: v for k,v in zip(keys, values)}
            idx = ex['idx']
            passage = ex['passage']
            passage = re.sub(r'(\.|\?|\!|\"|\')\n\n', r'\1 ', passage)
            query = ex['query']
            answers = ex['answers']
            inputs = f"{passage} {query}"
            for i, entity in enumerate(ex["entities"]):
                new_text = re.sub('@placeholder', f"{entity} ", inputs)
                new_batch['text'].append(new_text)
                new_batch['labels'].append(1 if entity in answers else 0)
                new_batch['entities'].append([ex["entities"]])
                new_batch['entity'].append(entity)
                new_batch['passage'].append(passage)
                new_batch['query'].append(query)
                new_batch['answers'].append(answers)
                new_batch["idx"].append(idx)
        return new_batch


if __name__ == "__main__":
    dm = SuperGLUEDataModule_record("roberta-base", "record")
    dm.setup("fit")
    import ipdb; ipdb.set_trace()
    batch = next(iter(dm.train_dataloader()))
    print(batch)