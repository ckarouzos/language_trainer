from dataclasses import dataclass, field
from typing import Optional

@dataclass
class DataArguments:
    """ Arguments pertaining to what data we are going to input our model for training and eval. """

    task_name: str = field(
        metadata={"help": "The name of the task to train on: record, rte, wic, wsc, cb, copa, multirc, boolq..."}
    )
    max_seq_length: int = field(
        default=128,
        metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."}
    )
    train_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/CPU for training."}
    )
    eval_batch_size: int = field(
        default=8,
        metadata={"help": "Batch size per GPU/CPU for evaluation."}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "Number of workers for the dataloaders."}
    )


@dataclass
class ModelArguments:
    """ Arguments pertaining to which model/config/tokenizer we are going to fine-tune from. """
    
    seed: int = field(
        default=42,
        metadata={"help": "Seed for training"}
    )
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}    
    )
    config_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    learning_rate: float = field(
        default=5e-5,
        metadata={"help": "The initial learning rate for Adam."}
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Epsilon for Adam optimizer."}
    )
    weight_decay: float = field(
        default=0.01,
        metadata={"help": "Weight decay if we apply some."}
    )
    warmup_ratio: int = field(
        default=0.06,
        metadata={"help": "Linear warmup over warmup_ratio * total_steps."}
    )
    max_epochs: int = field(
        default=10,
        metadata={"help": "Total number of training epochs to perform."}
    )
    method: str = field(
        default="finetune",
        metadata={"help": "Method to train the model: finetune, mlm, clm"}
    )