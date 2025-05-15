from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.encoder import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from tqdm import tqdm
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg
from utils.buffer import Buffer
import pdb
import numpy as np
from einops import rearrange
from collections import OrderedDict
import time
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from collections import defaultdict
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, train_test_split
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder, mask):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, input):
        return self.encoder(input, mask=self.mask)[:, -1]

class net(nn.Module):
    def __init__(self, args, device):
        super().__init__()
        self.device = device
        encoder = TSEncoder(input_dims=args.enc_in + 7,
                             output_dims=320,  # standard ts2vec backbone value
                             hidden_dims=64, # standard ts2vec backbone value
                             depth=10) 
        self.encoder = TS2VecEncoderWrapper(encoder, mask='all_true').to(self.device)
        self.dim = args.c_out * args.pred_len
        
        #self.regressor = nn.Sequential(nn.Linear(320, 320), nn.ReLU(), nn.Linear(320, self.dim)).to(self.device)
        self.regressor = nn.Linear(320, self.dim).to(self.device)
        
    def forward(self, x):
        rep = self.encoder(x)
        y = self.regressor(rep)
        return y

class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        self.model = net(args, device = self.device)
 
        self.buffer = Buffer(500, self.device)       
        self.count = 0
        if args.finetune:
            inp_var = 'univar' if args.features == 'S' else 'multivar'
            model_dir = str([path for path in Path(f'/export/home/TS_SSL/ts2vec/training/ts2vec/{args.data}/')
                .rglob(f'forecast_{inp_var}_*')][args.finetune_model_seed])
            state_dict = torch.load(os.path.join(model_dir, 'model.pkl'))
            for name in list(state_dict.keys()):
                if name != 'n_averaged':
                    state_dict[name[len('module.'):]] = state_dict[name]
                del state_dict[name]
            self.model[0].encoder.load_state_dict(state_dict)
        
        ########################
        # ATTRIBUTES FOR OCAR
        self.tau = 0
        self.representation = PMatKFAC
        self.variant = 'regression'
        self.regul = self.args.OCAR_regul
        self.regul_last = self.args.OCAR_regul_last
        self.lambda_ = 0.2
        self.F_ema = None
        self.F_ema_inv = None
        self.alpha_ema = self.args.OCAR_alpha_ema
        self.alpha_ema_last = self.alpha_ema
        self.iterations = 0
        self.output_size = None
        self.freq = 1
        ########################

    def _get_data(self, flag):
        args = self.args

        data_dict_ = {
            'ETTh1': Dataset_ETT_hour,
            'ETTh2': Dataset_ETT_hour,
            'ETTm1': Dataset_ETT_minute,
            'ETTm2': Dataset_ETT_minute,
            'WTH': Dataset_Custom,
            'ECL': Dataset_Custom,
            'Solar': Dataset_Custom,
            'custom': Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.args.data]
        timeenc = 2

        if flag  == 'test':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.test_bsz;
            freq = args.freq
        elif flag == 'val':
            shuffle_flag = False;
            drop_last = False;
            batch_size = args.batch_size;
            freq = args.detail_freq
        elif flag == 'pred':
            shuffle_flag = False;
            drop_last = False;
            batch_size = 1;
            freq = args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True;
            drop_last = True;
            batch_size = args.batch_size;
            freq = args.freq

        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate)
        return self.opt

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        self.opt = self._select_optimizer()
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1

                self.opt.zero_grad()
                pred, true, x = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    # mettere qua OCAR
                    ###################
                    #self.before_update(x, pred, true)
                    ###################
                    self.opt.step()
                
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            #test_loss = self.vali(test_data, test_loader, criterion)
            test_loss = 0.
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model, best_model_path

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, _ = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='vali')
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def test(self, setting, data='test'):
        test_data, test_loader = self._get_data(flag=data)

        #reset optimizer to SGD using online_lr
        self.opt = optim.SGD(self.model.parameters(), lr=self.args.online_lr)

        self.model.eval()
        if self.online == 'regressor':
            for p in self.model.encoder.parameters():
                p.requires_grad = False 
        elif self.online == 'none':
            for p in self.model.parameters():
                p.requires_grad = False
        
        preds = []
        trues = []
        start = time.time()
        maes,mses,rmses,mapes,mspes = [],[],[],[],[]

        #for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(tqdm(test_loader)):
            pred, true = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='test')
            preds.append(pred.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(pred.detach().cpu().numpy(), true.detach().cpu().numpy())
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        print('test shape:', preds.shape, trues.shape)
        MAE, MSE, RMSE, MAPE, MSPE = cumavg(maes), cumavg(mses), cumavg(rmses), cumavg(mapes), cumavg(mspes)
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        #mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}, time:{}'.format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark, mode='train'):
        if mode =='test' and self.online != 'none':
            return self._ol_one_batch(dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark)

        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float()
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = self.model(x)
        else:
            outputs = self.model(x)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)'), x   # return x to use them in the fisher
    
    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)
        batch_y = batch_y.float()
        for _ in range(self.n_inner):
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(x)
            else:
                outputs = self.model(x)

            loss = criterion(outputs, true)
            if not self.buffer.is_empty():
                buff_x, buff_y, idx = self.buffer.get_data(8)
                out = self.model(buff_x)
                loss += 0.2* criterion(out, buff_y)
            loss = loss /1.2
            loss.backward()
            # mettere qua OCAR
            ###################
            # concatenate curr data and buffer data
            if not self.buffer.is_empty():
                mb_x = torch.cat([x, buff_x], dim=0)
                mb_output = torch.cat([outputs, out], dim=0)
                mb_y = torch.cat([true, buff_y], dim=0)
            else:
                mb_x = x
                mb_output = outputs
                mb_y = true
            self.before_update(mb_x, mb_output, mb_y)
            ###################
            self.opt.step()       
            
            self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        idx = self.count +  torch.arange(batch_y.size(0)).to(self.device)
        self.count += batch_y.size(0)
        self.buffer.add_data(examples = x, labels = true, logits = idx)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')






    def before_update(self, mb_x, mb_output, mb_y):


        #Compute the weights for the FIM to compensate for different classes frequencies
        batch_size = int(mb_x.size(0))

        #Create a temporary dataloader to compute the FIM
        temp_dataset = torch.utils.data.TensorDataset(mb_x, mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataset, batch_size=mb_x.size(0), shuffle=False)


        #if mb_output.size(1) != self.output_size:
        #    self.iterations = 0
        
        if self.iterations % self.freq == 0:
            #Last layer FIM
            F = compute_FIM_last_layer(self.model, 
                                       temp_dataloader, 
                                       self.model.regressor, 
                                       mb_output.size(1),
                                        self.device,
                                        self.lambda_,
                                        new_idxs = [0])

            #Update the EMA of the FIM
            if self.F_ema is None or self.alpha_ema == 1.0:
                self.F_ema = F
            else:
                self.F_ema = self.EMA_FIM(self.F_ema, F)

        self.iterations += 1

        #original_last_known = torch.norm(strategy.model.linear.classifier.weight.grad[list(self.known_classes), :].flatten())
        #original_last_new = torch.norm(strategy.model.linear.classifier.weight.grad[list(new_classes), :].flatten())

        #Grads last layer 
        grads = []
        for p in self.model.regressor.parameters():
            if p.grad is not None:
                grads.append(p.grad.view(-1))
        if grads:
            grad_vector = torch.cat(grads)
        
            reg_F = self.F_ema + self.regul_last 
            reg_grads = grad_vector / reg_F

            #replace the last layer gradients with the new ones
            start_idx = 0
            for p in self.model.regressor.parameters():
                numel = p.numel()
                p.grad.copy_(reg_grads[start_idx:start_idx + numel].view_as(p))
                start_idx += numel


        



    def EMA_FIM(self, mat_old, mat_new):
        mat_new = mat_old * (1 - self.alpha_ema) + mat_new * self.alpha_ema
        return mat_new

    
def compute_FIM_last_layer(model, loader, last_layer, n_output, device, lambda_, new_idxs):
    
    model.eval()
    n_examples = len(loader.sampler)
    params = list(last_layer.parameters())
    num_params = sum(p.numel() for p in params)

    FIM = torch.zeros(num_params, device=device)
    

    for batch in loader:
        inputs = batch[0]
        inputs = inputs.to(device)
        inputs.require_grad = True
        batch_size = inputs.size(0)
        
        # Forward 
        inputs, _ = batch
        output = model(inputs).view(batch_size, n_output)  
        lambda_ = torch.ones_like(output)
        lambda_[new_idxs] = 1
        output = lambda_ * output    
        
        for i in range(batch_size):
            for d in range(n_output):
                grad_tuple = torch.autograd.grad(
                    outputs=output[i, d],
                    inputs=params,
                    retain_graph=True,
                    create_graph=False,
                    allow_unused=True
                )

                grad_vector = torch.cat([
                    g.view(-1) if g is not None else torch.zeros(p.numel(), device=device)
                    for p, g in zip(params, grad_tuple)
                ])

                FIM += grad_vector ** 2
    
    FIM /= n_examples
    return FIM
