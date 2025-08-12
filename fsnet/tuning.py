import numpy as np
import pandas as pd
import torch

from ray.tune.search.hyperopt import HyperOptSearch
from ray import tune
import ray

import os
import yaml
import importlib

from utils.tools import init_dl_program
import copy

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
        "online_lr": tune.loguniform(1e-4, 1e-1),
        "OCAR_regul" : tune.loguniform(1e-2, 1e-0),
        "OCAR_alpha_ema" : tune.uniform(0.0, 1.0),
        "OCAR_score_lr" : tune.loguniform(1e-2, 1),
    }

    def train_function(config, base_args):
        try: 
            trial_args = copy.deepcopy(base_args)
            for key, value in config.items():
                setattr(trial_args, key, value)

            init_dl_program(trial_args.gpu, seed=trial_args.finetune_model_seed)

            Exp = getattr(importlib.import_module('exp.exp_{}'.format(trial_args.method)), 'Exp_TS2VecSupervised')
            trial_exp = Exp(trial_args)

            trial_exp.model.load_state_dict(torch.load(best_model_path)) 

            metrics, _, _, _, _ = trial_exp.test(setting, data='test')
            mse = {"mse": metrics[1]}
            
            if np.isnan(mse["mse"]):
                mse["mse"] = 1e10
        except Exception as e:
            mse = {"mse": 1e20}

        tune.report(mse)  



    hyperopt_search = HyperOptSearch(metric="mse", mode="min")


    tuner = tune.Tuner(
        tune.with_resources(
            tune.with_parameters(train_function, base_args=args),
            {"gpu": 0.15, "num_retries": 0},
        ),
        tune_config=tune.TuneConfig(
            num_samples=100,
            max_concurrent_trials=6,
            search_alg=hyperopt_search,
        ),
        param_space=search_space,
        run_config=tune.RunConfig(storage_path="/data/e.urettini/projects/OCL_TS/ray_results"),
    )

    results = tuner.fit()

   
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