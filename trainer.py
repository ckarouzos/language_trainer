from datetime import datetime
from typing import Optional

import datasets
import torch
from pytorch_lightning import LightningDataModule, LightningModule, Trainer, seed_everything
from torch.utils.data import DataLoader
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_scheduler,
)

from datamodules.gluedatamodule import GLUEDataModule
from modules.gluetransformer import GLUETransformer

if __name__ =="__main__":
    seed_everything(42)
    dm = GLUEDataModule("distilbert-base-uncased")
    dm.prepare_data()
    dm.setup("fit")

    model = GLUETransformer("distilbert-base-uncased",
                            num_labels=dm.num_labels,
                            eval_splits=dm.eval_splits,
                            task_name=dm.task_name,
    )

    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
    )
    import ipdb; ipdb.set_trace()