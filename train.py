import argparse
import os
import time
import datetime
import matplotlib.pyplot as plt
import numpy as np
from AimTS import AimTS
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout
from data_provider.data_factory import train_data_provider

def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def data_read(dir_path):
    with open(dir_path, 'r') as f:
        raw_data = f.read()
        data = raw_data[1: -1].split(", ")

    return np.asanyarray(data, float)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default=None, help='The dataset name')
    parser.add_argument('--run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--loader', type=str, required=True, help='The data loader used to load the experimental data. This can be set to UCR, UEA, forecast_csv, forecast_csv_univar, anomaly, or anomaly_coldstart')
    parser.add_argument('--gpu', type=int, default=2, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.0025, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--finetune_lr', type=float, default=0.002, help='The finetune learning rate')
    parser.add_argument('--method', type=str, default='CI', help='how to solve multivariate data')

    args = parser.parse_args()
    
    print("Dataset:", args.data)
    print("Arguments:", str(args))
    s = '***************************************************************\nDataset:{}, Arguments:{}\n'.format(args.data,str(args))

    with open(os.path.join('./res','save_result.txt'), 'a', encoding = 'utf-8') as f:   
                f.write(s)
                f.write('\n')
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)
    print(device)
    print('Loading data... \n')
    concat_dataset, data_loader = train_data_provider(args, flag='TRAIN')
    
    print('done')

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        output_dims=args.repr_dims,
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = 'checkpoints/' + name_with_datetime(args.run_name)
    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()

    model = AimTS(
         device=device,
         **config
    )

    loss_log = model.fit(
        concat_dataset,
        data_loader=data_loader,
        method=args.method,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True,
    )
    print('Saving model ...\n', end='done')
    model.save(f'{run_dir}/model.pkl')
    t = time.time() - t
    train_time = datetime.timedelta(seconds=t)
    print(f"\nTraining time: {train_time}\n")
    
    train_loss_path = r"./pic/pretrain/ts2img_train_loss.txt"
    y_train_loss = data_read(train_loss_path)
    x_train_loss = range(len(y_train_loss))

    plt.figure()

    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('epoch')
    plt.ylabel('loss')

    plt.plot(x_train_loss, y_train_loss, linewidth=1, linestyle='solid', label='train_loss')
    plt.legend()
    plt.title('Loss curve')
    plt.savefig('./pic/pretrain/ts2img_Loss.jpg')

    print('--------- end -------------------')
    print('\n')
