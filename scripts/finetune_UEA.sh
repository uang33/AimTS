for data_name in 'AtrialFibrillation' 
    do
        python -u fine_tune.py \
            --data $data_name \
            --run_name UEA \
            --loader UEA \
            --epochs 15 \
            --batch-size 8 \
            --repr-dims 512 \
            --max-threads 8 \
            --seed 42  \
            --save-every 5 \
            --finetune 1 \
            --lr 0.003 \
            --ftlr 0.001 \
            --method CI | tee -a log/$data_name.txt
    done 
