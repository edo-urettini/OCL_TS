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
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector
from nngeometry.layercollection import LayerCollection
from utils.stats import StudentTLoss
from scipy.stats import norm

from nngeometry.metrics import FIM_MonteCarlo, FIM

import os, csv
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
        self.regul = self.args.OCAR_regul
        self.regul_last = self.args.OCAR_regul_last
        self.lambda_ = 0.2
        self.F_ema = None
        self.F_ema_inv = None
        self.alpha_ema = self.args.OCAR_alpha_ema
        self.alpha_ema_last = self.alpha_ema
        self.iterations = 0
        self.freq = 100
        self.deg_f = self.args.deg_f
        self.ng_only_last = self.args.ng_only_last
        self.scale = torch.ones(args.c_out * args.pred_len, requires_grad=False).to(self.device)
        self.loss_mean = 0.0
        self.loss_sq_mean= 0.0
        self.z = norm.ppf(0.99)
        self.loss = 0.0
        self.grad_EMA = None
        self.delta_t = 1
        self.score_lr = self.args.OCAR_score_lr
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
        if self.deg_f>=100:
            criterion = nn.MSELoss()
            self.variant = 'regression'
        else:            
            criterion = StudentTLoss(nu=self.deg_f, reduction='mean')
            self.variant = 'student_t'
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
                if self.variant == 'student_t':
                    loss = criterion(pred, true, self.scale.detach())
                else:
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
            if self.variant == 'student_t':
                loss = criterion(pred.detach().cpu(), true.detach().cpu(), self.scale.detach().cpu())
            else:
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

            if self.variant == 'student_t':
                loss = criterion(outputs, true, self.scale.detach())
            else:
                loss = criterion(outputs, true)
            if not self.buffer.is_empty():
                buff_x, buff_y, idx = self.buffer.get_data(8)
                out = self.model(buff_x)
                if self.variant == 'student_t':
                    loss += 0.2 * criterion(out, buff_y, self.scale.detach())
                else:
                    loss += 0.2* criterion(out, buff_y)
            loss = loss / 1.2
            loss.backward()
            # mettere qua OCAR
            ###################
            self.loss = loss.item()
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

        #Create layer collection
        lc = None
        if self.ng_only_last:
            lc = LayerCollection()
            lc.add_layer_from_model(self.model, self.model.regressor)

        #Update FIM condition (trigger if current loss is worst p%)
        loss_a = 0.01
        if self.loss_mean == 0.0:
            self.loss_mean = self.loss
            self.loss_sq_mean = self.loss**2
        self.loss_mean = (1 - loss_a) * self.loss_mean + loss_a * self.loss
        self.loss_sq_mean = (1 - loss_a) * self.loss_sq_mean + loss_a * self.loss**2            
        loss_std = np.sqrt(self.loss_sq_mean - self.loss_mean**2)
        if self.loss > self.loss_mean + self.z * loss_std or self.iterations % self.freq == 0:
            update_fim = True
            self.delta_t = 1
            self.tau = self.regul
        else:
            self.delta_t += 1
            update_fim = False   
            self.tau += (1-self.regul)/self.freq
        

        #Create a temporary dataloader to compute the FIM
        temp_dataset = torch.utils.data.TensorDataset(mb_x, mb_y)
        temp_dataloader = torch.utils.data.DataLoader(temp_dataset, batch_size=mb_x.size(0), shuffle=False)

        if self.representation == PMatEKFAC and self.F_ema is not None:
            old_diag = self.F_ema.data[1]
        else:
            old_diag = None

        #if mb_output.size(1) != self.output_size:
        #    self.iterations = 0
        
        if update_fim:
            #Compute and update the FIM
            # FIM must not compute the gradients
            #with torch.no_grad():
            '''''
            F = FIM(model=self.model,
                    loader=temp_dataloader,
                    representation=self.representation,
                    variant=self.variant, 
                    device=self.device,
                    lambda_=self.lambda_,
                    new_idxs=[0],
                    deg_f = self.deg_f,
                    scale = self.scale,
                    n_output=mb_output.size(1))
            '''''

            
            F = FIM_MonteCarlo(model=self.model,
                    loader=temp_dataloader,
                    representation=self.representation,
                    variant=self.variant, 
                    device=self.device,
                    trials=100,
                    lambda_=self.lambda_,
                    new_idxs=[0],
                    deg_f = self.deg_f,
                    scale = self.scale.detach(),
                    n_output=mb_output.size(1),
                    layer_collection=lc)
             
            #Update the EMA of the FIM
            if self.F_ema is None or (self.alpha_ema == 1.0 and self.alpha_ema_last == 1.0):
                self.F_ema = F
            else:
                self.F_ema = self.EMA_kfac(self.F_ema, F, delta_t=self.delta_t)
            id_last = list(self.F_ema.data.keys())[-1]
            self.F_ema_inv = self.F_ema.inverse(regul = self.tau)

        self.iterations += 1

        if self.representation == PMatEKFAC:
            self.F_ema.update_diag(temp_dataloader)
            if old_diag is not None:
                self.F_ema = self.EMA_diag(old_diag, self.F_ema)

        #Update scale parameter
        err = mb_y - mb_output
        score_scale = (self.deg_f * self.scale * (err**2 - self.scale)) / (self.deg_f * self.scale + err**2)
        score_scale = score_scale.mean(dim=0)
        self.scale = self.scale + self.score_lr * score_scale
        self.scale = torch.clamp(self.scale, min=1e-2, max=1e3)

        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(self.model, layer_collection=lc)
        if self.grad_EMA is None:
            self.grad_EMA = original_grad_vec
        else:
            self.grad_EMA = self.EMA_grad(self.grad_EMA, original_grad_vec)
        regularized_grad = self.F_ema_inv.mv(self.grad_EMA)
        regularized_grad.to_model_grad(self.model)

    def EMA_kfac(self, mat_old, mat_new, delta_t=1):
        """
        Compute the exponential moving average of two PMatKFAC matrices.

        :param mat_old: The previous PMatKFAC matrix.
        :param mat_new: The new PMatKFAC matrix.
        :return: A new PMatKFAC matrix representing the EMA.
        """
        alpha = 1 - (1 - self.alpha_ema) ** delta_t
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

            ema_a = (1 - alpha) * a_old + alpha * a_new
            ema_g = (1 - alpha) * g_old + alpha * g_new

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
    
    def EMA_grad(self, grad_old, grad_new):
        old = grad_old.to_flat()
        new = grad_new.to_flat()
        ema_flat = (1 - self.alpha_ema) * old + self.alpha_ema * new
        return PVector(grad_old.layer_collection, vector_repr=ema_flat)
    

    def compare_FIMs(self, F1, F2):
        """
        compute Frobenius norm of the difference between two FIMs
        """
        F1 = F1.data
        F2 = F2.data
        total_diff_norm = 0.0
        total_F1_norm = 0.0
        for key in F1.keys():
            a1, g1 = F1[key]
            a2, g2 = F2[key]
            diff_norm = torch.norm(a1 - a2)**2 + torch.norm(g1 - g2)**2
            F1_norm = torch.norm(a1)**2 + torch.norm(g1)**2
            total_diff_norm += diff_norm 
            total_F1_norm += F1_norm
        total_norm = total_diff_norm / total_F1_norm

        return np.sqrt(total_norm.item())
            
        
        

    def before_update_temp(self, mb_x, mb_output, mb_y):

        #Try different scales, reguls and deg_f
        for regul in [0.1, 1]:
                    for i in range(1, 1000):
                        try:
                            self.regul = regul
                            self.scale = 1.0
                            #Randomize the input and the target keeping the same shape
                            def sample_k_loguniform(n, k_min=1e-2, k_max=1e4):
                                log_k = np.random.uniform(np.log(k_min), np.log(k_max), size=n)
                                return np.exp(log_k)
                            k, k1 = sample_k_loguniform(2)
                            mb_x = k*torch.randn_like(mb_x)
                            mb_y = k1*torch.randn_like(mb_y)

                            #initialize the weights with xavier
                            for m in self.model.modules():
                                if isinstance(m, nn.Linear):
                                    nn.init.xavier_uniform_(m.weight)
                                    if m.bias is not None:
                                        nn.init.zeros_(m.bias)
                            #Set the model to training mode
                            self.model.train()
                            #Set the model to the device
                            self.model.to(self.device)


                            #Forward and backward pass
                            self.model.zero_grad()
                            mb_output = self.model(mb_x)
                            criterion = StudentTLoss(nu=self.deg_f, scale= self.scale, reduction='mean')
                            loss = criterion(mb_output, mb_y)
                            loss.backward()


                            #Create layer collection
                            lc = None
                            if self.ng_only_last:
                                lc = LayerCollection()
                                lc.add_layer_from_model(self.model, self.model.regressor)


                            #Create a temporary dataloader to compute the FIM
                            temp_dataset = torch.utils.data.TensorDataset(mb_x, mb_y)
                            temp_dataloader = torch.utils.data.DataLoader(temp_dataset, batch_size=mb_x.size(0), shuffle=False)

                            
                            if self.iterations % self.freq == 0:
                                F = FIM(model=self.model,
                                        loader=temp_dataloader,
                                        representation=self.representation,
                                        n_output=mb_output.size(1),
                                        variant=self.variant, 
                                        device=self.device,
                                        lambda_=self.lambda_,
                                        new_idxs=[0],
                                        deg_f = self.deg_f,
                                        layer_collection=lc,
                                        scale = self.scale)

                                self.F_ema = F
                                id_last = list(self.F_ema.data.keys())[-1]
                                self.F_ema_inv = self.F_ema.inverse(id_last=id_last, regul = self.regul, regul_last=self.regul)

                            self.iterations += 1

                            #Size of the output layer
                            self.output_size = mb_output.size(1)


                            #Compute the regularized gradient
                            original_grad_vec = PVector.from_model_grad(self.model, layer_collection=lc)
                            regularized_grad = self.F_ema_inv.mv(original_grad_vec)
                            regularized_grad.to_model_grad(self.model)

                            
                            #Compute the norms 
                            orig_norm = torch.norm(original_grad_vec.get_flat_representation()).item()
                            reg_norm = torch.norm(regularized_grad.get_flat_representation()).item()
                            target_norm = torch.norm(mb_y).item()
                            input_norm = torch.norm(mb_x).item()
                            new_input_norm = torch.norm(mb_x[0]).item()
                            old_input_norm = torch.norm(mb_x[1:]).item()
                            m   = self.output_size
                            v   = self.deg_f
                            tau = self.regul
                            bound_norm = (np.sqrt(m) / 4.0) * np.sqrt((v + 1) * (v + 3) / (v * tau))
                            #Store the original gradient and the regularized gradient norm history in a csv file
                            history_path = './grad_norm_history.csv'
                            file_exists  = os.path.isfile(history_path)

                            # Write header once, then append each row: iteration, orig, reg
                            with open(history_path, 'a', newline='') as fp:
                                writer = csv.writer(fp)
                                if not file_exists:
                                    writer.writerow(['iteration', 'original_grad_norm', 'regularized_grad_norm', 'target_norm', 'input_norm', 'new_input_norm', 'old_input_norm', 'bound_norm', 'scale', 'regul', 'deg_f'])
                                writer.writerow([self.iterations, orig_norm, reg_norm, target_norm, input_norm, new_input_norm, old_input_norm, bound_norm, self.scale, self.regul, self.deg_f])
                        except Exception as e:
                            print("Error: ", e)
        #Kill the process with an error
        assert False, "Error: Process killed to avoid infinite loop"