import os
import random
from typing import Optional

import torch
import numpy as np
import seaborn as sns
from torch.nn import functional as F

# slurm support
NPROC = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
DEFAULT_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

default_seed = 42
resources_dir = "resources"


def seed_everything(seed=default_seed):
    """Seed all the random number generators for reproducibility."""
    global default_seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    default_seed = seed


def init_seaborn():
    sns.set_theme(style="whitegrid", palette="muted")


def get_mask(X: torch.Tensor):
    return X != 0


def masked_sum(X: torch.Tensor, mask: torch.Tensor, dim=0):
    return (X * mask).sum(dim=dim)


def masked_mean(X: torch.Tensor, mask: torch.Tensor, dim=0):
    x_sum = masked_sum(X, mask)
    mask_sum = mask.sum(dim=dim)
    return x_sum / mask_sum


def masked_max(X: torch.Tensor, mask: torch.Tensor, dim=0):
    return (X * mask).max(dim=dim)[0]


def masked_last(X: torch.Tensor, mask: torch.Tensor, dim=0):
    # Q) Why the following line works?
    last_idx = mask.sum(dim=dim) - 1
    return X[last_idx, torch.arange(mask.shape[1], device=X.device)]


def masked_softmax(
        t: torch.Tensor, mask: Optional[torch.Tensor], dim: int = -1):
    if mask is None:
        return F.softmax(t, dim=dim)

    t_masked = t * mask
    t_max = torch.max(t_masked, dim=dim, keepdim=True)[0]
    t_exp = torch.exp(t_masked - t_max)

    t_exp_masked = t_exp * mask
    t_exp_masked_sum = torch.sum(t_exp_masked, dim=dim, keepdim=True)
    is_zero = t_exp_masked_sum == 0
    t_exp_masked_sum += is_zero.float()

    return t_exp_masked / t_exp_masked_sum
