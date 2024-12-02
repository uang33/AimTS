    list=(  
            'ACSF1' 
    )

for data_name in ${list[@]}
    do
        python -u fine_tune.py \
        --data $data_name \
        --loader UCR \
        --batch-size 8 \
        --epochs 36 \
        --repr-dims 512 \
        --max-threads 8 \
        --seed 3407   \
        --finetune 1 \
        --lr 0.003 \
        --ftlr 0.001 | tee -a log/$data_name.txt
    done

