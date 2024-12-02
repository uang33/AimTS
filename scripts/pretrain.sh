
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128
python -u train.py \
    --run_name AimTS \
    --loader monash \
    --epochs 60 \
    --batch-size 16 \
    --repr-dims 512 \
    --max-threads 8 \
    --seed 3407  \
    --save-every 1 \
    --finetune_lr 0.002 \
    --lr 0.0007 \
    --gpu 3 \
    --method CI | tee -a log/AimTS.log