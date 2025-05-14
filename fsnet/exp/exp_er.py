import time

from tqdm import tqdm

import torch

from einops import rearrange
from utils.metrics import metric, cumavg
from utils.buffer import Buffer

from .exp_basic import OCLTSExp


class ExperienceReplayExp(OCLTSExp):
    def __init__(self, **kwargs):
        super(ExperienceReplayExp, self).__init__(**kwargs)

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
