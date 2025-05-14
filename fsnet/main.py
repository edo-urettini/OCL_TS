import yaml
import os

# from exp.exp_basic import OCLTSExp
from exp.exp_er import ExperienceReplayExp
from exp.exp_fsnet import FSNetExp
from exp.exp_naive import NaiveExp


from fsnet.utils.reproducibility import set_seed, seed_worker, get_dataloader_seed_generator

SEED, SEED_DL = 123, 42
set_seed(SEED)
generator = get_dataloader_seed_generator(SEED_DL)


def prepare_config(experience_strategy: str, main_config_path: str):

    if experience_strategy == "er":
        config_path = os.path.join(main_config_path, "config_er.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["model"]["model"] = "simple"
        exp_class = ExperienceReplayExp
    elif experience_strategy == "naive":
        config_path = os.path.join(main_config_path, "config_naive.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["model"]["model"] = "naive"
        exp_class = NaiveExp
    elif experience_strategy == "fsnet":
        # fsnet
        config_path = os.path.join(main_config_path, "config_fsnet.yaml")
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        config["model"]["model"] = "fsnet"
        exp_class = FSNetExp
    else:
        raise ValueError("Experience strategy not supported")

    data_parser = {
        "ETTh1": {
            "data": "ETTh1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTh2": {
            "data": "ETTh2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm1": {
            "data": "ETTm1.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "ETTm2": {
            "data": "ETTm2.csv",
            "T": "OT",
            "M": [7, 7, 7],
            "S": [1, 1, 1],
            "MS": [7, 7, 1],
        },
        "WTH": {
            "data": "WTH.csv",
            "T": "WetBulbCelsius",
            "M": [12, 12, 12],
            "S": [1, 1, 1],
            "MS": [12, 12, 1],
        },
        "ECL": {
            "data": "ECL.csv",
            "T": "MT_320",
            "M": [321, 321, 321],
            "S": [1, 1, 1],
            "MS": [321, 321, 1],
        },
        "Solar": {
            "data": "solar_AL.csv",
            "T": "POWER_136",
            "M": [137, 137, 137],
            "S": [1, 1, 1],
            "MS": [137, 137, 1],
        },
        "Toy": {"data": "Toy.csv", "T": "Value", "S": [1, 1, 1]},
        "ToyG": {"data": "ToyG.csv", "T": "Value", "S": [1, 1, 1]},
        "Exchange": {"data": "exchange_rate.csv", "T": "OT", "M": [8, 8, 8]},
        "Illness": {"data": "national_illness.csv", "T": "OT", "M": [7, 7, 7]},
        "Traffic": {"data": "traffic.csv", "T": "OT", "M": [862, 862, 862]},
    }
    if config["data"]["data"] in data_parser.keys():
        data_info = data_parser[config["data"]["data"]]
        config["data"]["data_path"] = data_info["data"]
        config["data"]["target"] = data_info["T"]
        # config["model"]["enc_in"], config["model"]["dec_in"], config["model"]["c_out"] = data_info[config["data"]["features"]]
        config["model"]["enc_in"], _, config["model"]["c_out"] = data_info[
            config["data"]["features"]
        ]  # we don't need this

    config["data"]["detail_freq"] = config["data"]["freq"]
    config["data"]["freq"] = config["data"]["freq"][-1:]

    # set the enc_in dimension for the model. We need to consider the temporal features and the ways they are computed
    temp_feat_dim = 4 if config["data"]["timeenc"] in {0, 1} else 7
    config["model"]["enc_in"] += temp_feat_dim
    return exp_class, config


def main():

    EXPERIENCE_STRATEGY = "er"  # alternatives: 'fsnet', 'er', 'naive'
    MAIN_CONFIG_PATH = "/Users/platypus/Desktop/OCL_TS/fsnet"
    exp_class, config = prepare_config(EXPERIENCE_STRATEGY, MAIN_CONFIG_PATH)

    exp = exp_class(
        model_config=config["model"],
        data_config=config["data"],
        run_config=config["run"],
        warm_up_config=config["warm_up"],
        tuning_config=config["tuning"],
        # online_config = config['online'],
        checkpoint_path=config["checkpoint_path"],
    )

    exp.warm_up(seed_worker=seed_worker, generator=generator)
    # here the experiment has a model trained and saved in
    # self.checkpoint_path

    exp.tune_hyperparameters(seed_worker=seed_worker, generator=generator)
    # here the experiment has the results from the tuning in
    # self.best_results and self.best_hyperparameter_config

    [mae, mse, rmse, mape, mspe, exp_time], MAE, MSE, preds, trues = exp.online(
        seed_worker=seed_worker, generator=generator
    )
    # we can also call this passing the best configuration and the best model as parameters


if __name__ == "__main__":
    main()
