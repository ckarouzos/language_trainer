import os
import sys
import torch
from transformers import HfArgumentParser, AutoModel
from datamodules.supergluedatamodule import SuperGLUEDataModule, SuperGLUEDataModule_record
from modules.finetune import superGLUE_Transformer, superGLUE_Transformer_record
from modules.mlm_pretraining import Transformer_MLM
from utils.parser_dataclasses import ModelArguments, DataArguments
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar


if __name__ == "__main__":

    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    elif len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        model_args, data_args = parser.parse_yaml_file(yaml_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()


    seed_everything(model_args.seed)
    if data_args.task_name == "record":
        dm = SuperGLUEDataModule_record(
            model_name_or_path=model_args.model_name_or_path, 
            task_name=data_args.task_name, 
            max_seq_length=data_args.max_seq_lenght, 
            train_batch_size=data_args.train_batch_size, 
            eval_batch_size=data_args.eval_batch_size, 
            num_workers=data_args.num_workers,)
    else:
        dm = SuperGLUEDataModule(
            model_name_or_path=model_args.model_name_or_path, 
            task_name=data_args.task_name, 
            max_seq_length=data_args.max_seq_lenght, 
            train_batch_size=data_args.train_batch_size, 
            eval_batch_size=data_args.eval_batch_size, 
            num_workers=data_args.num_workers,)
    dm.prepare_data()
    dm.setup("fit")

    if data_args.task_name == "record":
        model = superGLUE_Transformer_record(
            model_name_or_path=model_args.model_name_or_path,
            task_name=data_args.task_name,
            num_labels=dm.num_labels,
            learning_rate=model_args.learning_rate,
            adam_epsilon=model_args.adam_epsilon,
            warmup_steps=model_args.warmup_steps,
            weight_decay=model_args.weight_decay,
            train_batch_size=data_args.train_batch_size,
            eval_batch_size=data_args.eval_batch_size,
            eval_splits=dm.eval_splits,)
    else:
        model = superGLUE_Transformer(
            model_name_or_path=model_args.model_name_or_path,
            task_name=data_args.task_name,
            num_labels=dm.num_labels,
            learning_rate=model_args.learning_rate,
            adam_epsilon=model_args.adam_epsilon,
            warmup_steps=model_args.warmup_steps,
            weight_decay=model_args.weight_decay,
            train_batch_size=data_args.train_batch_size,
            eval_batch_size=data_args.eval_batch_size,
            eval_splits=dm.eval_splits,)

    tb_logger = pl_loggers.TensorBoardLogger(save_dir="./", log_graph=True)
    trainer = Trainer(
        max_epochs=model_args.max_epochs,
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
