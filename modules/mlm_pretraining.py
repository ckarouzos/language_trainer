from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from modules.roberta_lm import RobertaForLM
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForPreTraining,
    get_linear_schedule_with_warmup,
)

class Transformer_MLM(LightningModule):

    def __init__(
        self,
        model_name_or_path: str,
        learning_rate: float = 6e-4,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.01,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        method: str = "mlm",
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path)
        if method == "mlm":
            self.model = AutoModelForPreTraining.from_pretrained(model_name_or_path, config=self.config)
        elif method == "clm":
            self.model = RobertaForLM.from_pretrained(model_name_or_path, config=self.config)

    def forward(self, input_ids, attention_mask):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        outputs = self(input_ids=input_ids, attention_mask=attention_mask)
        val_loss = outputs[0]
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        x = {"loss": val_loss}
        return x

    def validation_epoch_end(self, outputs):
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.log("avg_val_loss", loss, prog_bar=True)

    def configure_optimizers(self):
        model = self.model
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            }
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.hparams.learning_rate, eps=self.hparams.adam_epsilon)

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_steps,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return [optimizer], [scheduler]
