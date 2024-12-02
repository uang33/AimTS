import argparse
import os
import matplotlib.pyplot as plt
from AimTS import AimTS
import datautils
from utils import init_dl_program
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data', help='The dataset name')
    parser.add_argument('--run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=2, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The encoder learning rate (defaults to 0.001)')
    parser.add_argument('--ftlr', type=float, default=0.001, help='The head learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=100, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--finetune', type=float, default=None, help='data percent')
    parser.add_argument('--method', type=str, default='CI', help='how to solve multivariate data')

    args = parser.parse_args()
    
    print("Dataset:", args.data)
    print("Arguments:", str(args))


    s = '***************************************************************\nDataset:{}, Arguments:{}\n'.format(args.data,str(args))
    with open(os.path.join('./res','save_result.txt'), 'a', encoding = 'utf-8') as f:   
        f.write(s)

    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    if args.loader == 'UCR':
        task_type = 'classification'
        test_train_data, test_train_labels, test_test_data, test_test_labels = datautils.load_UCR(args.data)
        
    elif args.loader == 'UEA':
        task_type = 'classification'
        test_train_data, test_train_labels, test_test_data, test_test_labels = datautils.load_UEA(args.data)

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        max_train_length=args.max_train_length,
        method=args.method,
    )
    
    model = AimTS(
        input_dims=test_train_data.shape[2],
        device=device,

        **config
    )

    model.load('./checkpoints/model.pkl')
    
    loss_ft, metrics_dict = model.finetune_fit(test_train_data, test_train_labels, test_test_data, test_test_labels, epochs=args.epochs,
                                               finetune_data=args.finetune, batch_size=args.batch_size, encoding_window = 'full_series', finetune_lr=args.ftlr, method=args.method)
    
    
    with open(os.path.join('./res','save_result.txt'), 'a', encoding = 'utf-8') as f:   
       for key, value in metrics_dict.items():
            line = f"{key}: {value}\n"
            f.write(line)

    
    print('--------- end -------------------')
    print('\n')