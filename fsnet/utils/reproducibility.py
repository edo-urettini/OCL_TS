import os
import random
import numpy as np
import torch


def set_seed(seed: int = 42):
    """
    Set seed for Python, NumPy, and PyTorch to ensure reproducibility.
    """

    os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # for multi-GPU

    # Force deterministic algorithms in PyTorch
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id):
    """
    DataLoader worker seeding function.
    Ensures each worker has a different but deterministic seed.
    """
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def get_dataloader_seed_generator(seed: int):
    """
    Returns a torch.Generator seeded for DataLoader reproducibility.
    """
    g = torch.Generator()
    g.manual_seed(seed)
    return g
