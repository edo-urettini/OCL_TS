import numpy as np
import pandas as pd
import torch

from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune
import ray

import os
import yaml
import importlib

#from utils.tools import init_dl_program

def online_hpo(args, exp, setting, best_model_path):
    """
    Run hyperparameter tuning for the online training phase using Ray Tune
    with HyperOptSearch.
    
    Parameters:
        args: Parsed command line arguments.
        exp: An experiment instance already initialized with the pre-trained model.
        setting: A unique identifier for the current experiment.
    
    Returns:
        best_config (dict): The best hyperparameters found.
    """

    best_model_path = os.path.abspath(best_model_path)

    ray.init(num_gpus=1, num_cpus=12)

    # Define the hyperparameter search space.
    search_space = {
        "buffer_size": tune.choice([500]),
        "test_lr": tune.loguniform(1e-5, 1e-2), #tune.choice([0.0005]),    #tune.loguniform(1e-5, 1e-2),
        "n_replay": tune.choice([2, 4, 8, 16]),
        "er_loss_weight": tune.loguniform(1e-2, 10)
    }
    search_space = {
       "buffer_size": tune.choice([500]),
       "test_lr": tune.choice([0.0001]),
       "n_replay": tune.choice([8]),
       "er_loss_weight": tune.choice([2])
    }

    def train_function(config):

        Exp = getattr(importlib.import_module('exp.exp_{}'.format(args.method)), 'Exp_TS2VecSupervised')
        trial_exp = Exp(args)

        trial_exp.model.load_state_dict(torch.load(best_model_path))

        for key, value in config.items():
            setattr(args, key, value)
        

        metrics, _, _, _, _ = trial_exp.test(setting, data='val')
        mse = {"mse": metrics[1]}

        tune.report(mse)


    hyperopt_search = HyperOptSearch(metric="mse", mode="min")


    tuner = tune.Tuner(
        train_function,
        tune_config=tune.TuneConfig(
            num_samples=5,
            max_concurrent_trials=10,
            search_alg=hyperopt_search,
        ),
        param_space=search_space,
        run_config=tune.RunConfig(),
    )

    results = tuner.fit()

    ###
    results.get_dataframe().to_csv("results_fifo_real_ECL_2.csv", index=False)
    #import pickle
    #with open("results_fifo_ECL_2.pkl", "wb") as f:
    #    pickle.dump(results, f)
    ###
   
    best_config = results.get_best_result(metric="mse", mode="min").config

    # Save the best configuration
    best_config_dir = os.path.join("./best_configs", args.method)
    os.makedirs(best_config_dir, exist_ok=True)
    best_config_filename = os.path.join(best_config_dir, "best_config.yaml")
    with open(best_config_filename, 'w') as file:
        yaml.dump(best_config, file)

    #Save all results
    results_df = results.get_dataframe()
    
    # Save the full results to CSV for later analysis.
    results_dir = os.path.join("./tuning_results", args.method)
    os.makedirs(results_dir, exist_ok=True)
    results_csv_filename = os.path.join(results_dir, f"tuning_results_{setting}.csv")
    results_df.to_csv(results_csv_filename, index=False)

    return best_config