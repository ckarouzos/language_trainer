
for lr in 1e-5 2e-5 3e-5 5e-5; do
    python ../trainers/trainer.py \
        --task_name  \
        --model_name_or_path  \
        --train_batch_size 32 \
        --eval_batch_size 32 \
        --num_workers 4 \
        --learning_rate ${lr} \
        --max_epochs 20 \
