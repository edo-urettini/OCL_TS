import os
import time
from collections import defaultdict

import numpy as np


import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune
import ray

from models.net import Net, FSNet
from utils.tools import EarlyStopping, adjust_learning_rate
from einops import rearrange

from data.data_loader import (
    Dataset_ETT_hour,
    Dataset_ETT_minute,
    Dataset_Custom,
    Dataset_Pred,
)

import warnings

warnings.filterwarnings("ignore")


class OCLTSExp:
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

        self.model_config = model_config
        self.data_config = data_config
        self.run_config = run_config
        self.warm_up_config = warm_up_config
        self.tuning_config = tuning_config
        # self.online_config = online_config

        self.checkpoint_path = checkpoint_path
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
        self.best_model_path = os.path.join(checkpoint_path, "checkpoint.pth")

        self.device = self._acquire_device()

        # initialize the model
        model_name = self.model_config.pop("model")
        if model_name == "fsnet":
            model_class = FSNet
        elif model_name == "simple" or model_name == "naive":
            model_class = Net

        self.model = model_class(
            **model_config, pred_len=data_config["pred_len"], device=self.device
        )
        self.model.to(self.device)

        self.criterion = nn.MSELoss()
        self.opt = self._select_optimizer(lr=self.warm_up_config["learning_rate"])

        self.best_results = None
        self.best_hyperparameter_config = None

    def _acquire_device(self):
        if self.run_config["use_gpu"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = (
                str(self.run_config["gpu"])
                if not self.run_config["use_multi_gpu"]
                else self.run_config["devices"]
            )
            device = torch.device("cuda:{}".format(self.run_config["gpu"]))
            print("Use GPU: cuda:{}".format(self.run_config["gpu"]))
        elif torch.backends.mps.is_available():
            # for MAC
            device = torch.device("mps")
            print("Use MPS")
        else:
            device = torch.device("cpu")
            print("Use CPU")
        return device

    def _select_optimizer(self, lr: float):
        return optim.Adam(self.model.parameters(), lr=lr)

    def _get_data(self, flag: str, seed_worker=None, generator=None):
        args = self.data_config

        data_dict_ = {
            "ETTh1": Dataset_ETT_hour,
            "ETTh2": Dataset_ETT_hour,
            "ETTm1": Dataset_ETT_minute,
            "ETTm2": Dataset_ETT_minute,
            "WTH": Dataset_Custom,
            "ECL": Dataset_Custom,
            "Solar": Dataset_Custom,
            "custom": Dataset_Custom,
        }
        data_dict = defaultdict(lambda: Dataset_Custom, data_dict_)
        Data = data_dict[self.data_config["data"]]

        if flag == "test":
            shuffle_flag = False
            drop_last = False
            batch_size = args["test_bsz"] if "test_bsz" in args else 1
            freq = args["freq"]
        elif flag == "val":
            shuffle_flag = False
            drop_last = False
            batch_size = args["batch_size"]
            freq = args["detail_freq"]
        elif flag == "pred":
            shuffle_flag = False
            drop_last = False
            batch_size = 1
            freq = args["detail_freq"]
            Data = Dataset_Pred
        else:
            shuffle_flag = True
            drop_last = True
            batch_size = args["batch_size"]
            freq = args["freq"]

        data_set = Data(
            root_path=args["root_path"],
            data_path=args["data_path"],
            flag=flag,
            size=[args["seq_len"], args["label_len"], args["pred_len"]],
            features=args["features"],
            target=args["target"],
            inverse=args["inverse"],
            timeenc=args["timeenc"],
            freq=args["freq"],
            cols=args["cols"],
            perc_warm_up=args["perc_warm_up"],
            perc_val=args["perc_val"],
            scale=args["scale"],
        )

        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            # num_workers=args.num_workers,
            drop_last=drop_last,
            worker_init_fn=seed_worker,
            generator=generator,
        )

        return data_set, data_loader

    def warm_up(self, seed_worker=None, generator=None):
        """
        Warm up the model with the given configuration.
        """

        train_data, train_loader = self._get_data(flag="train")
        val_data, val_loader = self._get_data(flag="val")

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.warm_up_config["patience"], verbose=True
        )

        for epoch in range(self.warm_up_config["train_epochs"]):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
                iter_count += 1

                self.opt.zero_grad()
                # forward
                pred = self.model(batch_x, batch_x_mark)
                f_dim = -1 if self.data_config["features"] == "MS" else 0
                batch_y = (
                    batch_y[:, -self.data_config["pred_len"] :, f_dim:]
                    .float()
                    .to(self.device)
                )
                true = rearrange(batch_y, "b t d -> b (t d)")
                # update
                loss = self.criterion(pred, true)
                train_loss.append(loss.item())
                loss.backward()
                self.opt.step()
                # log
                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | loss: {2:.7f}".format(
                            i + 1, epoch + 1, loss.item()
                        )
                    )
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.warm_up_config["train_epochs"] - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)

            # validation
            self.model.eval()
            val_loss = []
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                val_loader
            ):
                pred = self.model(batch_x, batch_x_mark)
                batch_y = (
                    batch_y[:, -self.data_config["pred_len"] :, f_dim:]
                    .float()
                    .to(self.device)
                )
                true = rearrange(batch_y, "b t d -> b (t d)")
                loss = self.criterion(pred.detach().cpu(), true.detach().cpu())
                val_loss.append(loss)
            val_loss = np.average(val_loss)
            # log
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, val_loss
                )
            )
            # early stop
            early_stopping(val_loss, self.model, self.checkpoint_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.opt, epoch + 1, self.warm_up_config)

        self.model.load_state_dict(torch.load(self.best_model_path))

        return

    def _get_search_space(self):
        """
        Get the search space for hyperparameter tuning.
        By design, hyperparameters are structured as follows:
        hyperparameter_name: {type of range: numberic}
        if type of range == 'choice', then the value is a list of values
        if type of range == 'loguniform', then the value is a list of two values [min, max]
        """
        search_space = {}
        for k, v in self.tuning_config.items():
            if type(v) != dict:
                search_space[k] = tune.choice([v])
                continue
            range_type, range_val = list(v.items())[0]  # brutto

            if range_type == "choice":
                search_space[k] = tune.choice(range_val)
            elif range_type == "loguniform":
                search_space[k] = tune.loguniform(range_val[0], range_val[1])
            else:
                raise ValueError(f"Unknown range type: {range_type}")

        return search_space

    def tune_hyperparameters(self, seed_worker=None, generator=None):
        """
        Tune the model with for the best online hyperparameters configuration.
        """

        ray.init()

        search_space = self._get_search_space()

        def obj_function(config):
            """
            Objective function for the hyperparameter tuning.
            """

            # Load the best model from warm up
            self.model.load_state_dict(torch.load(self.best_model_path))

            [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues = (
                self.online(config=config, flag="val", use_tqdm=False)
            )
            mse_rep = {"mse": mse}  # metrics[1]}
            tune.report(mse_rep)
            return

        hyperopt_search = HyperOptSearch(metric="mse", mode="min")

        tuner = tune.Tuner(
            obj_function,
            tune_config=tune.TuneConfig(
                num_samples=self.tuning_config["num_samples"],
                max_concurrent_trials=self.tuning_config["max_concurrent_trials"],
                search_alg=hyperopt_search,
            ),
            param_space=search_space,
            run_config=tune.RunConfig(),
        )

        self.best_results = tuner.fit()
        self.best_hyperparameter_config = self.best_results.get_best_result(
            metric="mse", mode="min"
        ).config

        return

    def online(
        self,
        config: dict | None = None,
        model_path: str | None = None,
        flag: str = "test",
        use_tqdm: bool = True,
        seed_worker=None,
        generator=None,
    ):
        """
        Online learning with the given configuration.
        """
        if not config:
            assert (
                self.best_hyperparameter_config is not None
            ), "Please tune the hyperparameters first, or pass a config."
            config = self.best_hyperparameter_config

        # Load the best model from tuning
        if not model_path:
            assert (
                self.best_model_path is not None
            ), "Please warm up the model first, or pass a model path."
            model_path = self.best_model_path
        self.model.load_state_dict(torch.load(model_path))

        # Load the data
        _, data_loader = self._get_data(flag=flag)

        self.model.train()

        return self._online(config, data_loader, use_tqdm=use_tqdm)

    def _online(self, config: dict, data_loader: DataLoader, use_tqdm: bool = True):
        raise NotImplementedError(
            """This method should be implemented in the child class.
            It should contain the online learning step for the model.
            """
        )
