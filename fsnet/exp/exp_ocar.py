import time

from tqdm import tqdm

import torch

from einops import rearrange
from utils.metrics import metric, cumavg
from utils.buffer import Buffer

import numpy as np
from nngeometry.metrics import FIM
from nngeometry.object import PMatDiag, PMatBlockDiag, PMatKFAC, PMatEKFAC, PMatDense, PMatQuasiDiag, PVector
from nngeometry.layercollection import LayerCollection
from scipy.stats import norm
import torch.nn as nn
from utils.stats import StudentTLoss
import os, csv


from .exp_basic import OCLTSExp


class OCARExp(OCLTSExp):
    def __init__(self, 
                 tau=0,
                 representation='PMatKFAC',
                 variant='student_t',
                 regul=0.1,
                 regul_last=0.1,
                 lambda_=0.2,
                 alpha_ema=0.5,
                 iterations=0,
                 ocar_freq=100,
                 deg_f=5,
                 ng_only_last=False,
                 scale=1.0,
                 loss_mean=0.0,
                 loss_sq_mean=0.0,
                 loss=0.0,
                 **kwargs):
        super(OCARExp, self).__init__(**kwargs)
        self.tau = tau
        if representation == 'PMatKFAC':
            self.representation = PMatKFAC
        else:
            raise ValueError("Representation not supported: {}".format(representation))

        self.variant = variant
        self.regul = regul
        self.regul_last = regul_last
        self.lambda_ = lambda_
        self.F_ema = None
        self.F_ema_inv = None
        self.alpha_ema = alpha_ema
        self.alpha_ema_last = self.alpha_ema
        self.iterations = iterations
        self.ocar_freq = ocar_freq
        self.deg_f = deg_f
        self.ng_only_last = ng_only_last
        self.scale = scale
        self.loss_mean = loss_mean
        self.loss_sq_mean= loss_sq_mean
        self.z = norm.ppf(0.99)
        self.loss = loss


    def _init_buffer(self, buffer_size, device, mode="reservoir"):
        self.buffer = Buffer(buffer_size, device, mode)
        self.count = 0


    def _online(self, config: dict, data_loader, use_tqdm: bool = True):
        """
        Online training loop.
        :param config: configuration dictionary
        :param data: dataset object
        :param data_loader: data loader
        :param use_tqdm: whether to use tqdm for progress bar
        :return:
        [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues
        """

        preds = []
        trues = []
        start = time.time()
        maes, mses, rmses, mapes, mspes = [], [], [], [], []

        self.opt = self._select_optimizer(lr=config["learning_rate"])

        self._init_buffer(config["buffer_size"], self.device, config["buffer_type"])

        iterable_ = enumerate(tqdm(data_loader)) if use_tqdm else enumerate(data_loader)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in iterable_:

            true = rearrange(batch_y, "b t d -> b (t d)").float().to(self.device)

            for _ in range(config["n_inner"]):
                self.opt.zero_grad()
                # current data
                outputs = self.model(batch_x, batch_x_mark)
                # current loss
                loss = self.criterion(outputs, true)
                # replay data and loss
                if not self.buffer.is_empty():
                    buff_x, buff_y, idx = self.buffer.get_data(config["n_replay"])
                    out = self.model(buff_x)
                    loss += config["er_loss_weight"] * self.criterion(out, buff_y)
                # loss normalization
                loss /= 1 + config["er_loss_weight"]
                # backward and update
                loss.backward()

                ### OCAR specific step
                self.loss = loss.item()
                # remember the model is called on the batch_x and batch_x_mark
                x = torch.cat([batch_x, batch_x_mark], dim=-1).to(self.device)
                if not self.buffer.is_empty():
                    
                    mb_x = torch.cat([x, buff_x], dim=0)
                    mb_output = torch.cat([outputs, out], dim=0)
                    mb_y = torch.cat([true, buff_y], dim=0)
                else:
                    mb_x = x
                    mb_output = outputs
                    mb_y = true
                self.before_update(mb_x, mb_output, mb_y)
                ###

                self.opt.step()

            f_dim = -1 if self.data_config["features"] == "MS" else 0
            batch_y = (
                batch_y[:, -self.data_config["pred_len"] :, f_dim:]
                .float()
                .to(self.device)
            )
            idx = self.count + torch.arange(batch_y.size(0)).to(self.device)
            self.count += batch_y.size(0)
            # in the buffer, we will save the concatenated data
            x = torch.cat([batch_x, batch_x_mark], dim=-1).float().to(self.device)
            self.buffer.add_data(examples=x, labels=true, logits=idx)

            preds.append(outputs.detach().cpu())
            trues.append(true.detach().cpu())
            mae, mse, rmse, mape, mspe = metric(
                outputs.detach().cpu().numpy(), true.detach().cpu().numpy()
            )
            maes.append(mae)
            mses.append(mse)
            rmses.append(rmse)
            mapes.append(mape)
            mspes.append(mspe)

        preds = torch.cat(preds, dim=0).numpy()
        trues = torch.cat(trues, dim=0).numpy()
        MAE, MSE, RMSE, MAPE, MSPE = (
            cumavg(maes),
            cumavg(mses),
            cumavg(rmses),
            cumavg(mapes),
            cumavg(mspes),
        )
        mae, mse, rmse, mape, mspe = MAE[-1], MSE[-1], RMSE[-1], MAPE[-1], MSPE[-1]

        end = time.time()
        exp_time = end - start
        # mae, mse, rmse, mape, mspe = metric(preds, trues)
        print("mse:{}, mae:{}, time:{}".format(mse, mae, exp_time))
        return [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues


    def before_update(self, mb_x, mb_output, mb_y):

        #Create layer collection
        lc = None
        if self.ng_only_last:
            lc = LayerCollection()
            lc.add_layer_from_model(self.model, self.model.regressor)

        #Update FIM condition (trigger if current loss is worst 5%)
        loss_a = 0.01
        if self.loss_mean == 0.0:
            self.loss_mean = self.loss
            self.loss_sq_mean = self.loss**2
        self.loss_mean = (1 - loss_a) * self.loss_mean + loss_a * self.loss
        self.loss_sq_mean = (1 - loss_a) * self.loss_sq_mean + loss_a * self.loss**2            
        loss_std = np.sqrt(self.loss_sq_mean - self.loss_mean**2)
        if self.loss > self.loss_mean + self.z * loss_std or self.iterations % self.ocar_freq == 0:
            update_fim = True
            self.tau = self.regul
        else:
            update_fim = False   
            self.tau += (1-self.regul)/self.ocar_freq
        

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
                    scale = self.scale,)

            #Update the EMA of the FIM
            if self.F_ema is None or (self.alpha_ema == 1.0 and self.alpha_ema_last == 1.0):
                self.F_ema = F
            else:
                self.F_ema = self.EMA_kfac(self.F_ema, F)
            id_last = list(self.F_ema.data.keys())[-1]
            self.F_ema_inv = self.F_ema.inverse(id_last=id_last, regul = self.tau, regul_last=self.tau)

        self.iterations += 1

        if self.representation == PMatEKFAC:
            self.F_ema.update_diag(temp_dataloader)
            if old_diag is not None:
                self.F_ema = self.EMA_diag(old_diag, self.F_ema)

        #Compute the regularized gradient
        original_grad_vec = PVector.from_model_grad(self.model, layer_collection=lc)
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

                    
                    if self.iterations % self.ocar_freq == 0:
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