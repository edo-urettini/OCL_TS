import argparse
import copy
from einops import rearrange
import os
import random

import numpy as np
import torch

from exp.old_exp_fsnet import Exp_TS2VecSupervised as ExpOld
from exp.exp_fsnet import FSNetExp as ExpNew

from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric, cumavg



# ------------- CONFIGURATIONS -------------
# MODELS
device = "cpu"
model_config = {
    "enc_in": 12,    # for WTH
    "c_out": 12,   # for WTH
    "output_dims": 320,
    "hidden_dims": 64,
    "depth": 10,
    "regressor_dims": 320,
    "model": "fsnet"
}
# convert model_config to args
model_args = argparse.Namespace(**model_config)
# change enc_in for the model_config because we take into account the temporal features
model_config["enc_in"] = model_config["enc_in"] + 7 

# DATA
data_config = {
    "data": "WTH.csv",  # from prepare config (with ETTh2)
    "target": "WetBulbCelsius",  # from prepare_config
    "M": [12, 12, 12], # from prepare_config
    "S": [1, 1, 1], # from prepare_config
    "MS": [12, 12, 1], # from prepare_config
    "root_path": "/Users/platypus/Desktop/OCL_TS/fsnet/data/WTH",  # ECL, ETT, WTH
    "data_path": "/Users/platypus/Desktop/OCL_TS/fsnet/data/WTH/WTH.csv",  # ECL, ETT, WTH
    "freq": "h",
    "detail_freq": "h",
    "features": "M",  # maybe this indicates if we want to use multiple features or not for input and output
    "seq_len": 60,
    "label_len": 0,
    "pred_len": 24,
    "batch_size": 32,
    "inverse": False,
    "cols": None,
    "perc_warm_up": 0.15,
    "perc_val": 0.35,  # 0.15
    "scale": True,  # data normalization
    "timeenc": 2,  # differences in computing temporal information in timefeatures.py
    "test_bsz": 1,  # batch size of test data   ONLY FOR OLD EXP
    "num_workers": 0,  # data loader num workers    ONLY FOR OLD EXP
}
data_args = argparse.Namespace(**data_config)

# RUN
run_config = {
    "num_workers": 0,
    "gpu": 0,
    "use_gpu": False,
    "use_multi_gpu": False,
    "devices": "0,1,2,3",
}

# WARM UP
warm_up_config = {
    "opt": "adam",
    "learning_rate": 0.001,
    "patience": 3,
    "train_epochs": 6,
    "lradj": "type1",  # learning rate adjustment (for the schedule)
}

# TUNING
tuning_config = {
    "n_inner": 1,  # number of inner loop updates
    "num_samples": 100,  # number of tried configurations
    "max_concurrent_trials": 10,
    # learning_rate:  {loguniform: [0.000001, 0.00001]}
    "learning_rate": {"loguniform": [0.00001, 0.01]},  # {choice: [0.0001]}
}

# COMBINE EVERYTHING
# combine the data and model args
total_args = data_args.__dict__
total_args.update(model_args.__dict__)
total_args.update(warm_up_config)
total_args.update(
    {
        "use_gpu": False,
        "online_learning": "full",
        "n_inner": 1,
        "opt": "adam",
        "finetune": False,
        "method": "fsnet",
        "checkpoints": "checkpoints",
        "use_amp": False,
    }
    )
args = argparse.Namespace(**total_args)

# ------------- TESTS -------------
# initialize the experiment classes
seed = 42
g_new = torch.Generator()
g_new.manual_seed(seed)
g_old = torch.Generator()
g_old.manual_seed(seed)
random.seed(seed)
#seed += 1
np.random.seed(seed)
#seed += 1
torch.manual_seed(seed)

device = "cpu"
old_exp = ExpOld(args)
new_exp = ExpNew(model_config, data_config, run_config, warm_up_config, tuning_config)

# DATA
train_data, train_loader = new_exp._get_data(flag="train", generator=g_new)
vali_data, vali_loader = new_exp._get_data(flag="val")
test_data, test_loader = new_exp._get_data(flag="test")
old_train_data, old_train_loader = old_exp._get_data(flag="train", generator=g_old)
old_vali_data, old_vali_loader = old_exp._get_data(flag="val")
old_test_data, old_test_loader = old_exp._get_data(flag="test")
# check if the data is the same
assert len(train_data) == len(old_train_data)
assert len(vali_data) == len(old_vali_data)
assert len(test_data) == len(old_test_data)
for b_new, b_old in zip(train_loader, old_train_loader):
    batch_x, batch_y, batch_x_mark, batch_y_mark = b_new
    old_batch_x, old_batch_y, old_batch_x_mark, old_batch_y_mark = b_old
    assert torch.equal(batch_x, old_batch_x)  # they are shuffled, but with the same generator
    assert torch.equal(batch_y, old_batch_y)  # they are shuffled, but with the same generator
    assert torch.equal(batch_x_mark, old_batch_x_mark)  # they are shuffled, but with the same generator
    assert torch.equal(batch_y_mark, old_batch_y_mark)  # they are shuffled, but with the same generator
for b_new, b_old in zip(vali_loader, old_vali_loader):
    batch_x, batch_y, batch_x_mark, batch_y_mark = b_new
    old_batch_x, old_batch_y, old_batch_x_mark, old_batch_y_mark = b_old
    assert torch.equal(batch_x, old_batch_x)  # they are not shuffled
    assert torch.equal(batch_y, old_batch_y)  # they are not shuffled
    assert torch.equal(batch_x_mark, old_batch_x_mark)  # they are not shuffled
    assert torch.equal(batch_y_mark, old_batch_y_mark)  # they are not shuffled
for b_new, b_old in zip(test_loader, old_test_loader):
    batch_x, batch_y, batch_x_mark, batch_y_mark = b_new
    old_batch_x, old_batch_y, old_batch_x_mark, old_batch_y_mark = b_old
    assert torch.equal(batch_x, old_batch_x)  # they are not shuffled
    assert torch.equal(batch_y, old_batch_y)  # they are not shuffled
    assert torch.equal(batch_x_mark, old_batch_x_mark)  # they are not shuffled
    assert torch.equal(batch_y_mark, old_batch_y_mark)  # they are not shuffled

# MODELS
# check if the models are the same
assert list(old_exp.model.state_dict().keys()) == list(new_exp.model.state_dict().keys())
for k in old_exp.model.state_dict().keys():
    assert old_exp.model.state_dict()[k].shape == new_exp.model.state_dict()[k].shape
# copy old parameters to a new model
old_exp.model.load_state_dict(copy.deepcopy(new_exp.model.state_dict()))
old_exp.model.eval()
new_exp.model.eval()
old_exp.model.to(device)
new_exp.model.to(device)
# check if the output is the same
x, y, x_mark, y_mark = train_data[0]
x = torch.tensor(x).unsqueeze(0)
x_mark = torch.tensor(x_mark).unsqueeze(0)
new_out = new_exp.model(x, x_mark)
old_out = old_exp.model(torch.cat([x, x_mark], dim=-1).float()) # old model wants a single float tensor
assert torch.equal(new_out, old_out)


# ------------- TRAINING/WARM UP -------------
opt_new = torch.optim.Adam(new_exp.model.parameters(), lr=0.001)
opt_old = torch.optim.Adam(old_exp.model.parameters(), lr=0.001)
criterion = torch.nn.MSELoss()
new_early_stopping = EarlyStopping(patience=warm_up_config["patience"], verbose=True)
old_early_stopping = EarlyStopping(patience=warm_up_config["patience"], verbose=True)
for epoch in range(2):
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    train_loader
                ):
        opt_new.zero_grad()
        opt_old.zero_grad()
        pred_new = new_exp.model(batch_x, batch_x_mark)
        pred_old = old_exp.model(torch.cat([batch_x, batch_x_mark], dim=-1).float())
        assert torch.allclose(pred_new, pred_old, atol=1e-6)
        true = rearrange(batch_y, 'b t d -> b (t d)').float()
        loss_new = criterion(pred_new, true)
        loss_old = criterion(pred_old, true)
        assert torch.allclose(loss_new, loss_old, atol=1e-6)
        loss_new.backward()
        loss_old.backward()
        # check if losses are the same
        assert torch.allclose(loss_new, loss_old, atol=1e-6)
        opt_new.step()
        opt_old.step()
        # check if the parameters are still the same
        for p1, p2 in zip(old_exp.model.parameters(), new_exp.model.parameters()):
            assert torch.allclose(p1, p2, atol=1e-6)
    
    # validation check
    new_exp.model.eval()
    old_exp.model.eval()
    tot_val_loss_new = []
    tot_val_loss_old = []
    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                    vali_loader
                ):
        pred_new = new_exp.model(batch_x, batch_x_mark)
        pred_old = old_exp.model(torch.cat([batch_x, batch_x_mark], dim=-1).float())
        assert torch.allclose(pred_new, pred_old, atol=1e-6)
        true = rearrange(batch_y, 'b t d -> b (t d)').float()
        val_loss_new = criterion(pred_new, true)
        val_loss_old = criterion(pred_old, true)
        assert torch.allclose(val_loss_new, val_loss_old, atol=1e-6)
        tot_val_loss_new.append(val_loss_new.item())
        tot_val_loss_old.append(val_loss_old.item())
    tot_val_loss_new = np.mean(tot_val_loss_new)
    tot_val_loss_old = np.mean(tot_val_loss_old)
    
    path_new = "test_check_new"
    path_old = "test_check_old"
    if not os.path.exists(path_new):
        os.makedirs(path_new)
    if not os.path.exists(path_old):
        os.makedirs(path_old)
    new_early_stopping(tot_val_loss_new, new_exp.model, path_new)
    old_early_stopping(tot_val_loss_old, old_exp.model, path_old)
    assert new_early_stopping.early_stop == old_early_stopping.early_stop
    # adjust learning rate
    adjust_learning_rate(opt_new, epoch+1, {"lradj": "type1", "learning_rate": 0.001})
    adjust_learning_rate(opt_old, epoch+1, {"lradj": "type1", "learning_rate": 0.001})
    assert np.allclose(opt_new.param_groups[0]['lr'], opt_old.param_groups[0]['lr'], atol=1e-6)

new_exp.model.load_state_dict(torch.load(os.path.join(path_new, 'checkpoint.pth')))
old_exp.model.load_state_dict(torch.load(os.path.join(path_old, 'checkpoint.pth')))
for p1, p2 in zip(old_exp.model.parameters(), new_exp.model.parameters()):
    assert torch.allclose(p1, p2, atol=1e-6)

# ------------- TESTING/ONLINE -------------
new_exp.model.eval()
old_exp.model.eval()
for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(
                train_loader
            ):
    opt_new.zero_grad()
    opt_old.zero_grad()
    pred_new = new_exp.model(batch_x, batch_x_mark)
    pred_old = old_exp.model(torch.cat([batch_x, batch_x_mark], dim=-1).float())
    assert torch.allclose(pred_new, pred_old, atol=1e-6)
    true = rearrange(batch_y, 'b t d -> b (t d)').float()
    loss_new = criterion(pred_new, true)
    loss_old = criterion(pred_old, true)
    assert torch.allclose(loss_new, loss_old, atol=1e-6)
    loss_new.backward()
    loss_old.backward()
    # check if losses are the same
    assert torch.allclose(loss_new, loss_old, atol=1e-6)
    opt_new.step()
    opt_old.step()
    # check if the parameters are still the same
    for p1, p2 in zip(old_exp.model.parameters(), new_exp.model.parameters()):
        assert torch.allclose(p1, p2, atol=1e-6)
    new_exp.model.store_grad()
    old_exp.model.store_grad()

    mae_old, mse_old, rmse_old, mape_old, mspe_old = metric(pred_old.detach().cpu().numpy(), true.detach().cpu().numpy())
    mae_new, mse_new, rmse_new, mape_new, mspe_new = metric(pred_new.detach().cpu().numpy(), true.detach().cpu().numpy())
    assert np.allclose(mae_old, mae_new, atol=1e-6)
    assert np.allclose(mse_old, mse_new, atol=1e-6)
    assert np.allclose(rmse_old, rmse_new, atol=1e-6)
    assert np.allclose(mape_old, mape_new, atol=1e-6)
    assert np.allclose(mspe_old, mspe_new, atol=1e-6)

    if i == 1000:
        break