import argparse
import torch
from transformers import AutoModelForSequenceClassification, AutoModel
from datamodules.supergluedatamodule import SuperGLUEDataModule, SuperGLUEDataModule_record
from modules.finetune import superGLUE_Transformer, superGLUE_Transformer_record
from modules.mlm_pretraining import Transformer_MLM
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar


if __name__ =="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_name", type=str, default="wic") 
    parser.add_argument("--model_name_or_path", type=str, default="../models/roberta-base")
    parser.add_argument("--max_epochs", type=int, default=10)
    parser.add_argument("--mlm", default=False, action="store_true")
    args = parser.parse_args()
    task_name = args.task_name
    model_name_or_path = args.model_name_or_path
    max_epochs = args.max_epochs
    mlm = args.mlm
    seed_everything(42)
    if task_name == "record":
        dm = SuperGLUEDataModule_record(model_name_or_path, task_name=task_name)
    else:
        dm = SuperGLUEDataModule(model_name_or_path, task_name=task_name)
    dm.prepare_data()
    dm.setup("fit")
    if mlm:
        model = Transformer_MLM(model_name_or_path, eval_splits=dm.eval_splits,)
    else:
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
        callbacks=[TQDMProgressBar(refresh_rate=10)]
    )
    trainer.fit(model, dm)

    if mlm:
        checkpoint = torch.load(trainer.checkpoint_callback.best_model_path)
        state_dict = checkpoint['state_dict']
        modified_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('model.roberta.'):
                modified_k = k[14:]
            else:
                modified_k = k
            modified_state_dict[modified_k] = v
        model = AutoModel.from_pretrained(model_name_or_path, state_dict=modified_state_dict)
        output_dir = "../mlm_model"
        model.save_pretrained(output_dir)
        if task_name == "record":
            model = superGLUE_Transformer_record(output_dir, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)
        else:
            model = superGLUE_Transformer(output_dir, num_labels=dm.num_labels, eval_splits=dm.eval_splits, task_name=dm.task_name,)
        tb_logger = pl_loggers.TensorBoardLogger(save_dir="./", log_graph=True)
        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1 if torch.cuda.is_available() else None,
            logger=tb_logger,
            callbacks=[TQDMProgressBar(refresh_rate=10)]
        )
    trainer.fit(model, dm)

    best_model_path = trainer.checkpoint_callback.best_model_path
    if best_model_path:
        print("Testing best model...")
        #trainer.test(model=model, ckpt_path=best_model_path, datamodule=dm, verbose=True)
