from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.ts2vec.encoder import TSEncoder, GlobalLocalMultiscaleTSEncoder
from models.ts2vec.losses import hierarchical_contrastive_loss
from models.model import Informer
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

import os
import time
from pathlib import Path

import warnings
warnings.filterwarnings('ignore')


class Exp_TS2VecSupervised(Exp_Basic):
    def __init__(self, args):
        self.args = args
        self.device = self._acquire_device()
        self.online = args.online_learning
        assert self.online in ['none', 'full', 'regressor']
        self.n_inner = args.n_inner
        self.opt_str = args.opt
        #self.model = net(args, device = self.device)
        self.model = Informer(args.enc_in, args.dec_in, args.c_out, args.seq_len, args.label_len, args.pred_len)
        self.model.to(self.device)

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
        # ATTRIBUTES FOR NatSR
        self.tau = 0
        self.representation = PMatKFAC
        self.regul = self.args.NatSR_regul
        self.regul_last = self.args.NatSR_regul_last
        self.lambda_ = 0.2
        self.F_ema = None
        self.F_ema_inv = None
        self.alpha_ema = self.args.NatSR_alpha_ema
        self.alpha_ema_last = self.alpha_ema
        self.iterations = 0
        self.freq = 100
        self.deg_f = self.args.deg_f
        self.ng_only_last = self.args.ng_only_last
        self.scale = torch.ones(1).to(self.device)
        self.loss_mean = 0.0
        self.loss_sq_mean= 0.0
        self.z = norm.ppf(0.99)
        self.loss = 0.0
        self.grad_EMA = None
        self.delta_t = 1
        self.score_lr = 0.1
        self.alpha_ema_grad = self.args.NatSR_alpha_ema_grad
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
        timeenc = 1

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
        if self.deg_f>1000:
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
                pred, true = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                if self.variant == 'student_t':
                    #loss = criterion(pred.detach().cpu(), true.detach().cpu(), self.scale.detach().cpu())
                    loss = criterion(pred, true, self.scale)
                else:
                    loss = criterion(pred, true)

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(self.opt)
                    scaler.update()
                else:
                    loss.backward()
                    self.opt.step()
                
                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()                
                
                train_loss.append(loss.item())
                
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
            pred, true = self._process_one_batch(
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

        #reset optimizer using online_lr
        self.opt = optim.AdamW(self.model.parameters(), lr=self.args.online_lr)


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
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        else:
            if self.args.output_attention:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
            else:
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
        if self.args.inverse:
            outputs = dataset_object.inverse_transform(outputs)
        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)


        return rearrange(outputs, 'b t d -> b (t d)'), rearrange(batch_y, 'b t d -> b (t d)')


    def _ol_one_batch(self,dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        true = rearrange(batch_y, 'b t d -> b (t d)').float().to(self.device)
        criterion = self._select_criterion()
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float().to(self.device)

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        x = torch.cat([batch_x.float(), batch_x_mark.float()], dim=-1).to(self.device)

        for _ in range(self.n_inner):
            # decoder input
            if self.args.padding==0:
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
            elif self.args.padding==1:
                dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float().to(self.device)
            dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
            # encoder - decoder
            if self.args.use_amp:
                with torch.cuda.amp.autocast():
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            else:
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if self.args.inverse:
                outputs = dataset_object.inverse_transform(outputs)
            outputs = outputs.reshape((outputs.shape[0], -1))
            if self.variant == 'student_t':
                loss = criterion(outputs, true, self.scale)
            else:
                loss = criterion(outputs, true)
            # replay
            if not self.buffer.is_empty():
                buff_x, buff_y, idx = self.buffer.get_data(8)
                buff_x, buff_x_mark = buff_x[:, :, :self.args.c_out], buff_x[:, :, self.args.c_out:]
                buff_y, buff_y_mark = buff_y[:, :, :self.args.c_out], buff_y[:, :, self.args.c_out:]
                buff_true = rearrange(buff_y, 'b t d -> b (t d)').float().to(self.device)

                # decoder input
                if self.args.padding==0:
                    dec_inp = torch.zeros([buff_y.shape[0], self.args.pred_len, buff_y.shape[-1]]).float().to(self.device)
                elif self.args.padding==1:
                    dec_inp = torch.ones([buff_y.shape[0], self.args.pred_len, buff_y.shape[-1]]).float().to(self.device)
                dec_inp = torch.cat([buff_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            buff_out = self.model(buff_x, buff_x_mark, dec_inp, buff_y_mark)[0]
                        else:
                            buff_out = self.model(buff_x, buff_x_mark, dec_inp, buff_y_mark)
                else:
                    if self.args.output_attention:
                        buff_out = self.model(buff_x, buff_x_mark, dec_inp, buff_y_mark)[0]
                    else:
                        buff_out = self.model(buff_x, buff_x_mark, dec_inp, buff_y_mark)
                if self.args.inverse:
                    buff_out = dataset_object.inverse_transform(buff_out)
                buff_out = buff_out.reshape((buff_out.shape[0], -1))
                if self.variant == 'student_t':
                    loss = criterion(buff_out, buff_true, self.scale)
                else:
                    loss = criterion(buff_out, buff_true)
                loss += 0.2 * loss
            loss.backward()
            # mettere qua NatSR
            ###################
            self.loss = loss.item()
            # concatenate curr data and buffer data
            if not self.buffer.is_empty():
                mb_x = torch.cat([batch_x, buff_x], dim=0)
                mb_x_mark = torch.cat([batch_x_mark, buff_x_mark])
                mb_y = torch.cat([batch_y, buff_y], dim=0)  #torch.cat([true, buff_true], dim=0)
                mb_y_mark = torch.cat([batch_y_mark, buff_y_mark], dim=0)
                mb_out = torch.cat([outputs, buff_out], dim=0)
            else:
                mb_x = batch_x
                mb_x_mark = batch_x_mark
                mb_y = batch_y
                mb_y_mark = batch_y_mark
                mb_out = outputs

            if self.args.padding==0:
                mb_dec_inp = torch.zeros([mb_y.shape[0], self.args.pred_len, mb_y.shape[-1]]).float().to(self.device)
            elif self.args.padding==1:
                mb_dec_inp = torch.ones([mb_y.shape[0], self.args.pred_len, mb_y.shape[-1]]).float().to(self.device)
            mb_dec_inp = torch.cat([mb_y[:,:self.args.label_len,:], mb_dec_inp], dim=1).float().to(self.device)
            self.before_update(mb_x, mb_x_mark, mb_dec_inp, mb_y_mark, mb_out, mb_y.to(self.device))
            ###################
            self.opt.step()       
            
            self.opt.zero_grad()

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        idx = self.count +  torch.arange(batch_y.size(0)).to(self.device)
        self.count += batch_y.size(0)
        self.buffer.add_data(examples = torch.cat([batch_x, batch_x_mark], dim=2), labels = torch.cat([batch_y, batch_y_mark], dim=2), logits = idx)
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)
        return outputs, rearrange(batch_y, 'b t d -> b (t d)')


    def before_update(self, mb_x, mb_x_mark, mb_dec_inp, mb_y_mark, mb_out, mb_y):

        #Create layer collection
        lc = None
        if self.ng_only_last:
            lc = LayerCollection()
            lc.add_layer_from_model(self.model, self.model.projection)
        else:
            lc = LayerCollection()
            known_modules = {
                "Linear",
                "Conv2d",
                "BatchNorm1d",
                "BatchNorm2d",
                "GroupNorm",
                "WeightNorm1d",
                "WeightNorm2d",
                "Cosine1d",
                "Affine1d",
                "ConvTranspose2d",
                "Conv1d",
                "LayerNorm",
                "Embedding",
            }
            for layer, mod in self.model.named_modules():
                mod_class = mod.__class__.__name__
                if mod_class in known_modules:
                    lc.add_layer(layer, LayerCollection._module_to_layer(mod))
            return lc

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
            #self.delta_t = 1
            #self.tau = self.regul
        else:
            #self.delta_t += 1
            update_fim = False   
            #self.tau += (1-self.regul)/self.freq
        
        #self.tau =  0.9/ (1 + self.scale.item()**2) + (0.1/self.scale.item()**2)
        self.tau =  self.regul
        #Create a temporary dataloader to compute the FIM
        temp_dataset = torch.utils.data.TensorDataset(mb_x, mb_x_mark, mb_dec_inp, mb_y_mark, mb_dec_inp)
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
                    trials=10,
                    lambda_=self.lambda_,
                    new_idxs=[0],
                    deg_f = self.deg_f,
                    scale = self.scale.detach(),
                    n_output=mb_dec_inp.size(1),
                    layer_collection=lc)
             
            #Update the EMA of the FIM
            if self.F_ema is None or (self.alpha_ema == 1.0 and self.alpha_ema_last == 1.0):
                self.F_ema = F
            else:
                self.F_ema = self.EMA_kfac(self.F_ema, F)
            id_last = list(self.F_ema.data.keys())[-1]
            self.F_ema_inv = self.F_ema.inverse(regul = self.tau)

        self.iterations += 1

        if self.representation == PMatEKFAC:
            self.F_ema.update_diag(temp_dataloader)
            if old_diag is not None:
                self.F_ema = self.EMA_diag(old_diag, self.F_ema)

        #Update scale parameter
        if self.variant == 'student_t':
            with torch.no_grad():
                err = mb_y - mb_out.reshape(mb_y.shape)
                s_2 = self.scale**2
                score_scale = (self.deg_f * s_2 * (err**2 - s_2)) / (self.deg_f * s_2 + err**2)
                score_scale = score_scale.mean()
                s_2 = s_2 + self.score_lr * score_scale
                self.scale = torch.clamp(s_2, min=1e-2, max=1e2)
                self.scale = torch.sqrt(torch.clamp(s_2, min=1e-2, max=1e2))
    

        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(self.model, layer_collection=lc)
        regularized_grad = self.F_ema_inv.mv(original_grad_vec)
        if self.grad_EMA is None:
            self.grad_EMA = regularized_grad
        else:
            self.grad_EMA = self.EMA_grad(self.grad_EMA, regularized_grad)
        self.grad_EMA.to_model_grad(self.model)


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
        ema_flat = (1 - self.alpha_ema_grad) * old + self.alpha_ema_grad * new
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
