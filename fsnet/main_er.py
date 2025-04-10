import argparse
import os
import torch
import random
import numpy as np
import uuid
import datetime
import importlib

#from exp.exp_online import Exp_TS2VecSupervised

from ray import tune
from ray.tune.search.hyperopt import HyperOptSearch


def init_dl_program(
        device_name,
        seed=None,
        use_cudnn=True,
        deterministic=False,
        benchmark=False,
        use_tf32=False,
        max_threads=None
):
    import torch
    if max_threads is not None:
        torch.set_num_threads(max_threads)  # intraop
        if torch.get_num_interop_threads() != max_threads:
            torch.set_num_interop_threads(max_threads)  # interop
        try:
            import mkl
        except:
            pass
        else:
            mkl.set_num_threads(max_threads)

    if seed is not None:
        random.seed(seed)
        seed += 1
        np.random.seed(seed)
        seed += 1
        torch.manual_seed(seed)

    if isinstance(device_name, (str, int)):
        device_name = [device_name]

    devices = []
    for t in reversed(device_name):
        t_device = torch.device(t)
        devices.append(t_device)
        if t_device.type == 'cuda':
            assert torch.cuda.is_available()
            torch.cuda.set_device(t_device)
            if seed is not None:
                seed += 1
                torch.cuda.manual_seed(seed)
    devices.reverse()
    torch.backends.cudnn.enabled = use_cudnn
    torch.backends.cudnn.deterministic = deterministic
    torch.backends.cudnn.benchmark = benchmark

    if hasattr(torch.backends.cudnn, 'allow_tf32'):
        torch.backends.cudnn.allow_tf32 = use_tf32
        torch.backends.cuda.matmul.allow_tf32 = use_tf32

    return devices if len(devices) > 1 else devices[0]

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')


# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]
parser.add_argument('--gamma', type=float, default=0.9)

# data args
parser.add_argument('--data', type=str, required=False, default='WTH', help='data')
parser.add_argument('--test_bsz', type=int, default=1)
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')

parser.add_argument('--root_path', type=str, default='/Users/platypus/Desktop/OCL_TS/fsnet/data/WTH', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='WTH.csv', help='data file')    
parser.add_argument('--seq_len', type=int, default=60, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=0, help='start token length of Informer decoder')
parser.add_argument('--target', type=str, default='WetBulbCelsius', help='target feature in S or MS task')
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
#
# data loader args
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
#
# optimizer args
parser.add_argument('--learning_rate', type=float, default=0.001, help='optimizer learning rate')
#
# net args
parser.add_argument('--enc_in', type=int, default=12, help='encoder input size')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
parser.add_argument('--c_out', type=int, default=12, help='output size')
#
# exp_basics args
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')
#
# Exp specific args
parser.add_argument('--online_learning', type=str, default='full')
parser.add_argument('--opt', type=str, default='adam')
parser.add_argument('--n_inner', type=int, default=1)

parser.add_argument('--finetune', action='store_true', default=False)
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--finetune_model_seed', type=int)
#
# train args
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
#
# lr scheduler args
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
#
# main args
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--method', type=str, default='er_clever')
#



args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
args.test_bsz = args.batch_size if args.test_bsz == -1 else args.test_bsz
if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
    'Toy': {'data': 'Toy.csv', 'T':'Value', 'S':[1,1,1]},
    'ToyG': {'data': 'ToyG.csv', 'T':'Value', 'S':[1,1,1]},
    'Exchange': {'data': 'exchange_rate.csv', 'T':'OT', 'M':[8,8,8]},
    'Illness': {'data': 'national_illness.csv', 'T':'OT', 'M':[7,7,7]},
    'Traffic': {'data': 'traffic.csv', 'T':'OT', 'M':[862,862,862]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.detail_freq = args.freq
args.freq = args.freq[-1:]

    

def objective(config):
    #for buffer_size, n_replay, er_loss_weight, test_lr in combinations:
    args.buffer_size = config["buffer_size"]
    args.n_replay = config["n_replay"]
    args.er_loss_weight = config["er_loss_weight"]
    args.test_lr = config["test_lr"]

    print('Args in experiment:')
    print(args)

    #Exp = Exp_TS2VecSupervised
    Exp = getattr(importlib.import_module('exp.exp_{}'.format(args.method)), 'Exp_TS2VecSupervised')

    metrics ,preds, true, mae, mse = [], [], [], [], []

    for ii in range(args.itr):
        print('\n ====== Run {} ====='.format(ii))
        # setting record of experiments
        #method_name = 'ts2vec_finetune' if args.finetune else 'ts2vec_supervised'
        method_name = args.method
        uid = uuid.uuid4().hex[:4]
        suffix = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M") + "_" + uid
        setting = '{}_{}_pl{}_ol{}_opt{}_tb{}_{}'.format(method_name, args.data, args.pred_len,args.online_learning, args.opt, args.test_bsz, suffix)

        init_dl_program(args.gpu, seed=ii)
        args.finetune_model_seed = ii
        exp = Exp(args) # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.train(setting)
        
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        m, mae_, mse_, p, t = exp.test(setting)
        metrics.append(m)
        if str(args.data) == 'ECL' or str(args.data) == 'Traffic':
            preds=[0]
            true=[0]
        else:
            preds.append(p)
            true.append(t)

        mae.append(mae_)
        mse.append(mse_)

        tune.report({"mae": mae_, "mse": mse_})

        #torch.cuda.empty_cache()

    """
    #folder_path = './results/' + setting + '/'
    folder_path = './results{}/{}/'.format(args.n_inner, setting)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    np.save(folder_path + 'metrics.npy', np.array(metrics))
    np.save(folder_path + 'preds.npy', np.array(preds))
    np.save(folder_path + 'trues.npy', np.array(true))
    np.save(folder_path + 'mae.npy', np.array(mae))
    np.save(folder_path + 'mse.npy', np.array(mse))
    """


search_space = {
    "buffer_size": tune.choice([500]),
    "test_lr": tune.loguniform(1e-5, 1e-2),
    "n_replay": tune.choice([2, 4, 8, 16]),
    "er_loss_weight": tune.loguniform(1e-2, 1)
    }
search_alg = HyperOptSearch(metric="mae", mode="min")
tuner = tune.Tuner(
    objective,
    param_space=search_space,
    tune_config=tune.TuneConfig(
        metric="mae",
        mode="min",
        search_alg=search_alg,
        num_samples=20,
        max_concurrent_trials=5
    ),
    run_config=tune.RunConfig()
)

results = tuner.fit()

#print("Best hyperparameters found were: ", results.get_best_result().metrics)

import pickle
with open('results.pkl', 'wb') as f:
    pickle.dump(results, f)
    