import argparse
import torch
from datamodules.supergluedatamodule import SuperGLUEDataModule, SuperGLUEDataModule_record
from modules.finetune import superGLUE_Transformer, superGLUE_Transformer_record
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="wic") 
    parser.add_argument("--model_name_or_path", type=str, default="../models/roberta-base")
    parser.add_argument("--max_epochs", type=int, default=10)
    args = parser.parse_args()
    task_name = args.task_name
    model_name_or_path = args.model_name_or_path
    max_epochs = args.max_epochs
    seed_everything(42)
    if task_name == "record":
        dm = SuperGLUEDataModule_record(model_name_or_path, task_name=task_name)
    else:
        dm = SuperGLUEDataModule(model_name_or_path, task_name=task_name)
    dm.prepare_data()
    dm.setup("fit")
    if task_name == "record":
        model = superGLUE_Transformer_record(model_name_or_path, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)
    else:
        model = superGLUE_Transformer(model_name_or_path, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./", log_graph=True)
    trainer = Trainer(
        max_epochs=max_epochs,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=tb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=50)]
    )
    trainer.fit(model, dm)