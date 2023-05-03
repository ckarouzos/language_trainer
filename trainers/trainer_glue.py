import argparse
import torch
from datamodules.gluedatamodule import GLUEDataModule
from modules.finetune import GLUE_Transformer
from modules.mlm_pretraining import Transformer_MLM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="cola") 
    parser.add_argument("--model_name_or_path", type=str, default="../models/roberta-base")
    parser.add_argument("--max_epochs", type=int, default=1)
    parser.add_argument("--mlm", default=False, action="store_true")
    args = parser.parse_args()
    task_name = args.task_name
    model_name_or_path = args.model_name_or_path
    max_epochs = args.max_epochs
    mlm = args.mlm
    seed_everything(42)
    dm = GLUEDataModule(model_name_or_path, task_name=task_name)
    dm.prepare_data()
    dm.setup("fit")
    if mlm:
        model = Transformer_MLM(model_name_or_path, eval_splits=dm.eval_splits,)
    else:
        model = GLUE_Transformer(model_name_or_path, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./", log_graph=True)
    trainer = Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1 if torch.cuda.is_available() else None,
        logger=tb_logger,
        callbacks=[TQDMProgressBar(refresh_rate=50)]
    )
    trainer.fit(model, dm)

    if mlm:
        model = GLUE_Transformer.load_from_checkpoint(trainer.checkpoint_callback.best_model_path, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="./", log_graph=True)
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            logger=tb_logger,
            callbacks=[TQDMProgressBar(refresh_rate=50)]
        )
    trainer.fit(model, dm)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print("Testing best model...")
        trainer.test(model=model, ckpt_path=best_model_path, datamodule=dm, verbose=True)
