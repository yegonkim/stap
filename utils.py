import torch
import numpy as np
import random

from omegaconf import OmegaConf
import pandas as pd

def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    
def torch_delete(tensor, indices):
    """Delete elements from a tensor in indices."""
    mask = torch.ones(tensor.shape[0], dtype=torch.bool)
    mask[indices] = False
    return tensor[mask]

def torch_expand(tensor, dim, copies):
    """Expand tensor along a dimension using torch.expand() for memory efficiency."""
    shape = list(tensor.shape)
    shape[dim] = shape[dim] * copies
    return tensor.expand(*shape)

def flatten_configdict(
    cfg: OmegaConf,
    sep: str = ".",
):
    cfgdict = OmegaConf.to_container(cfg)
    cfgdict = pd.json_normalize(cfgdict, sep=sep)

    return cfgdict.to_dict(orient="records")[0]

class direct_model(torch.nn.Module):
    def __init__(self, model, unrolling):
        super().__init__()
        self.model = model
        self.unrolling = unrolling
    def forward(self, x):
        for _ in range(self.unrolling):
            x = self.model(x)
        return x
    
class trajectory_model(torch.nn.Module):
    def __init__(self, model, unrolling):
        super().__init__()
        self.model = model
        self.unrolling = unrolling
    def forward(self, x):
        trajectory = []
        for _ in range(self.unrolling):
            x = self.model(x)
            trajectory.append(x)
        return torch.cat(trajectory, dim=1) # [cfg.train.batch_size, unrolling, nx]
    
class split_model(torch.nn.Module):
    def __init__(self, model, batch_size):
        super().__init__()
        self.model = model
        self.batch_size = batch_size
    def forward(self, x):
        y = [self.model(x_split) for x_split in x.split(self.batch_size)]
        return torch.cat(y, dim=0)
    
class ensemble_mean_model(torch.nn.Module):
    def __init__(self, ensemble):
        super().__init__()
        self.ensemble = ensemble
    def forward(self, x):
        y = [model(x) for model in self.ensemble]
        return torch.stack(y).mean(dim=0)