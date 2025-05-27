import time

from tqdm import tqdm

import torch

from einops import rearrange
from utils.metrics import metric, cumavg

from .exp_basic import OCLTSExp


class FSNetExp(OCLTSExp):
    def __init__(
        self,
        model_config: dict,
        data_config: dict,
        run_config: dict,
        warm_up_config: dict,
        tuning_config: dict,
        # online_config:dict,
        checkpoint_path: str = "./checkpoints",
        ):

        super(FSNetExp, self).__init__(
            model_config,
            data_config,
            run_config,
            warm_up_config,
            tuning_config,
            checkpoint_path,
        )

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

        iterable_ = enumerate(tqdm(data_loader)) if use_tqdm else enumerate(data_loader)
        for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in iterable_:

            true = rearrange(batch_y, "b t d -> b (t d)").float().to(self.device)

            for _ in range(config["n_inner"]):
                self.opt.zero_grad()
                outputs = self.model(batch_x, batch_x_mark)
                loss = self.criterion(outputs, true)
                loss.backward()
                self.opt.step()
                self.model.store_grad()

            f_dim = -1 if self.data_config["features"] == "MS" else 0
            batch_y = (
                batch_y[:, -self.data_config["pred_len"] :, f_dim:]
                .float()
                .to(self.device)
            )
            # in the buffer, we will save the concatenated data
            x = torch.cat([batch_x, batch_x_mark], dim=-1).float().to(self.device)

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
