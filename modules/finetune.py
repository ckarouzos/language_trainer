from typing import Optional

import torch
from pytorch_lightning import LightningModule
from torch.nn import ModuleDict
from torchmetrics import Accuracy, MatthewsCorrCoef, F1Score, ExactMatch
from transformers import (
    AdamW,
    AutoConfig,
    AutoModelForSequenceClassification,
    get_linear_schedule_with_warmup,
)

class superGLUE_Transformer(LightningModule):

    super_glue_tasks_metrics = {
        "boolq": ["binary_accuracy"],
        "cb": ["multiclass_f1", "multiclass_accuracy"],
        "copa": ["binary_accuracy"],
        "multirc": ["f1", "exact_match"],
        "rte": ["binary_accuracy"],
        "axg": ["binary_accuracy"],
        "axb": ["matthews_corrcoef"],
        "wic": ["binary_accuracy"],
    }

    def __init__(
        self,
        model_name_or_path: str,
        num_labels: int,
        task_name: str = "boolq",
        learning_rate: float = 2e-5,
        adam_epsilon: float = 1e-8,
        warmup_steps: int = 0,
        weight_decay: float = 0.0,
        train_batch_size: int = 32,
        eval_batch_size: int = 32,
        eval_splits: Optional[list] = None,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        self.config = AutoConfig.from_pretrained(model_name_or_path, num_labels=num_labels)

        self.mn_metric = ModuleDict({
            "binary_accuracy": Accuracy(task='binary'),
            "multiclass_accuracy": Accuracy(task='multiclass', num_classes=3),
            "multiclass_f1": F1Score(task='multiclass', num_classes=3),
            "f1": F1Score(task='binary'),
            "matthews_corrcoef": MatthewsCorrCoef(task='binary'),
            "exact_match": ExactMatch(task='multiclass', num_classes=2),
        })

        self.model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, config=self.config)
        self.metrics = ModuleDict({mn: self.mn_metric[mn] for mn in self.super_glue_tasks_metrics[task_name]})

    def forward(self, **inputs):
        return self.model(**inputs)

    def training_step(self, batch, batch_idx):
        outputs = self(**batch)
        loss = outputs[0]
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        for mn in self.metrics:
            metric = self.metrics[mn]
            ms = metric(torch.argmax(outputs[1], axis=1), batch["labels"])
            self.log("train_"+mn, ms, prog_bar=False, logger=True)
        return loss
    
    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        outputs = self(**batch)
        val_loss, logits = outputs[:2]
        if self.hparams.num_labels > 1:
            preds = torch.argmax(logits, axis=1)
        elif self.hparams.num_labels == 1:
            preds = logits.squeeze()
        labels = batch["labels"]
        self.log("val_loss", val_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        x = {"loss": val_loss, "preds": preds, "labels": labels}
        for mn in self.metrics:
            metric = self.metrics[mn]
            ms = metric(torch.argmax(outputs[1], axis=1), batch["labels"])
            self.log("val_"+mn, ms, prog_bar=True, logger=True)
            x["val_"+mn] = ms
        return x

    def validation_epoch_end(self, outputs):
        preds = torch.cat([x["preds"] for x in outputs]).detach().cpu().numpy()
        labels = torch.cat([x["labels"] for x in outputs]).detach().cpu().numpy()
        loss = torch.stack([x["loss"] for x in outputs]).mean()
        for mn in self.metrics:
            avg_sc = torch.stack([x["val_"+mn] for x in outputs]).mean()
        self.log("avg_val_loss", loss, prog_bar=True)
        self.log("avg_val_"+mn, avg_sc, prog_bar=True)

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