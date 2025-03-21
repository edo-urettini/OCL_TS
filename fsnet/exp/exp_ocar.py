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
        self.regul = 1e-8
        self.lambda_ = 1
        self.F_ema = None
        self.F_ema_inv = None
        self.alpha_ema = 0.1
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
        self.opt = optim.SGD(self.model.parameters(), lr=self.args.learning_rate)
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
                    self.before_update(x, pred, true)
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

        return self.model

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

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')

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
        '''This should not be needed
        # Check if new classes are observed
        curr_classes = set(strategy.experience.classes_in_this_experience)
        new_classes = curr_classes - self.known_classes    
        if new_classes:
            self.known_classes.update(curr_classes)
        '''


        #Compute the weights for the FIM to compensate for different classes frequencies
        batch_size = int(mb_x.size(0))
        weights = torch.ones(batch_size, device=self.device)
        '''This should not be needed
        n_known = len(self.known_classes)
        if len(self.buffer_idx) == len(weights):
            #######WARNING, MODIFIED FOR CLEAR########
            if self.iterations % strategy.train_epochs == 0:
                weights[self.buffer_idx] = self.n_new 
                self.n_new = self.n_new + strategy.train_epochs * self.regul
        '''

        if self.tau == 0:
            self.tau = self.opt.param_groups[0]['lr']
        else:
            self.tau = self.tau + self.regul

        #Create a temporary dataloader to compute the FIM
        temp_dataset = torch.utils.data.TensorDataset(mb_x, mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataset, batch_size=mb_x.size(0), shuffle=False)

        if self.representation == PMatEKFAC and self.F_ema is not None:
            old_diag = self.F_ema.data[1]
        else:
            old_diag = None

        #if mb_output.size(1) != self.output_size:
        #    self.iterations = 0
        
        if self.iterations % self.freq == 0:
            #Compute and update the FIM
            # FIM must not compute the gradients
            #with torch.no_grad():
            F = FIM(model=self.model,
                    loader=temp_dataloader,
                    representation=self.representation,
                    n_output=mb_output.size(1),
                    variant=self.variant, 
                    device=self.device,
                    lambda_=self.lambda_)

            #Update the EMA of the FIM
            if self.F_ema is None or (self.alpha_ema == 1.0 and self.alpha_ema_last == 1.0):
                self.F_ema = F
            else:
                self.F_ema = self.EMA_kfac(self.F_ema, F)
            self.F_ema_inv = self.F_ema.inverse(regul = 10*self.opt.param_groups[0]['lr'])

        self.iterations += 1

        if self.representation == PMatEKFAC:
            self.F_ema.update_diag(temp_dataloader)
            if old_diag is not None:
                self.F_ema = self.EMA_diag(old_diag, self.F_ema)

        #original_last_known = torch.norm(strategy.model.linear.classifier.weight.grad[list(self.known_classes), :].flatten())
        #original_last_new = torch.norm(strategy.model.linear.classifier.weight.grad[list(new_classes), :].flatten())

        #Size of the output layer
        self.output_size = mb_output.size(1)


        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(self.model)
        regularized_grad = self.F_ema_inv.mv(original_grad_vec)
        regularized_grad.to_model_grad(self.model)

    def EMA_kfac(self, mat_old, mat_new):
        """
        Compute the exponential moving average of two PMatKFAC matrices.

        :param mat_old: The previous PMatKFAC matrix.
        :param mat_new: The new PMatKFAC matrix.
        :return: A new PMatKFAC matrix representing the EMA.
        """
        if self.representation == PMatEKFAC:
            old = mat_old.data[0]
            new = mat_new.data[0]
        else:
            old = mat_old.data
            new = mat_new.data

        last_old_layer = list(old.keys())[-1]
        last_new_layer = list(new.keys())[-1]
        shared_keys = old.keys() & new.keys()
        
        for layer_id in shared_keys:
            a_old, g_old = old[layer_id]
            a_new, g_new = new[layer_id]

            ema_a = (1 - self.alpha_ema) * a_old + self.alpha_ema * a_new
            ema_g = (1 - self.alpha_ema) * g_old + self.alpha_ema * g_new

            new[layer_id] = (ema_a, ema_g)
        
       
        if last_old_layer != last_new_layer and self.alpha_ema_last < 1.0:
            a_old_last, g_old_last = old[last_old_layer]
            a_new_last, g_new_last = new[last_new_layer]

            ema_a_last = (1 - self.alpha_ema_last) * a_old_last + self.alpha_ema_last * a_new_last
            
            #g_new_last = g_new_last * self.alpha_ema_last + (1 - self.alpha_ema_last) 

            new[last_new_layer] = (ema_a_last, g_new_last)

        if self.representation == PMatEKFAC:
            mat_new.data = (new, mat_new.data[1])
        else:
            mat_new.data = new
 
         # Create a new PMatKFAC instance with the EMA data
        return mat_new


    def EMA_diag(self, diag_old, mat_new):
        #Compute the EMA of the diagonal of the FIM when using PMatEkfac representation
        old = diag_old
        new = mat_new.data[1]

        shared_keys = old.keys() & new.keys()
        last_old_layer = list(old.keys())[-1]
        last_new_layer = list(new.keys())[-1]

        for layer_id in shared_keys:
            old_diag = old[layer_id]
            new_diag = new[layer_id]

            ema_diag = (1 - self.alpha_ema) * old_diag + self.alpha_ema * new_diag
            new[layer_id] = ema_diag

        mat_new.data = (mat_new.data[0], new)

        return mat_new