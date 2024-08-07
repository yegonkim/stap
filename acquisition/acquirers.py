import torch
import numpy as np

from . import ACQUISITION_FUNCTIONS
from .feature_functions import get_features_ycov
from utils import torch_delete

from tqdm import tqdm

def select(ensemble, X_train, X_pool, batch_acquire, selection_method='random', acquisition_function='variance', **cfg):
    device = cfg.get('device', 'cpu')
    if selection_method == "random":
        weights = torch.ones(1, X_pool.shape[0])
        new_idxs = torch.multinomial(weights, num_samples=batch_acquire, replacement=False)[0]
    elif selection_method == "greedy":
        acquisition_function = ACQUISITION_FUNCTIONS[acquisition_function]
        
        pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
        new_idxs = torch.tensor([], dtype=torch.int64, device=device)

        for _ in tqdm(range(batch_acquire)):
            logical_new_idxs = torch.zeros(pool_idxs.shape[-1], dtype=torch.bool, device=device)
            logical_new_idxs[new_idxs] = True
            current_pool_idxs = pool_idxs[~logical_new_idxs]

            X = X_pool[new_idxs] # [current_batch, ...]
            cand_X = X_pool[current_pool_idxs] # [num_candidates, ...]
            cand_X = cand_X.unsqueeze(1) # [num_candidates, 1, ...]
            if X.shape[0] > 0:
                cand_X  = torch.cat([torch.stack([X]*cand_X.shape[0], dim=0), cand_X], dim=1) # [num_candidates, current_batch+1, ...]
            with torch.no_grad():
                utilities = acquisition_function(cand_X, ensemble, X_pool=X_pool) # [num_candidates]
            idxs = torch.argmax(utilities)
            new_idxs = torch.cat([new_idxs, current_pool_idxs[idxs:idxs+1]], dim=0)

    # elif selection_method == "kmeanspp":
    #     if cfg.acquisition.mode.split("_")[0] == "p":
    #         X_all = X_pool
    #         rem = torch.arange(X_all.shape[0], device=device, dtype=torch.int64)
    #         sel = torch.tensor([], dtype=torch.int64, device=device)
    #         new_sel = torch.tensor([], dtype=torch.int64, device=device)
    #     elif cfg.acquisition.mode.split("_")[0] == "tp":
    #         X_all = torch.cat([X_pool, self.train_list['X']], dim=0)
    #         rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
    #         sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
    #         new_sel = torch.tensor([], dtype=torch.int64, device=device)
    #     with torch.no_grad():
    #         features_all = get_features_ycov(X_all, ensemble, sim, cfg) # [bs_all, feature_dim]
    #         features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
        
    #     for _ in range(batch_acquire):
    #         if sel.shape[0] == 0:
    #             new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
    #         else:
    #             distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
    #             scores = torch.min(distances, dim=0).values
    #             new = torch.multinomial(scores, 1)
    #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
    #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
    #         rem = torch_delete(rem, new)
    #     X_pool = X_all[rem]
    #     X = X_all[new_sel]
    elif selection_method == "lcmd":
        pool_idxs = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
        new_idxs = torch.tensor([], dtype=torch.int64, device=device)

        X_all = torch.cat([X_pool, X_train], dim=0)
        rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
        sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
        new_sel = torch.tensor([], dtype=torch.int64, device=device)
        
        # if cfg.acquisition.mode.split("_")[0] == "p":
        #     X_all = X_pool
        #     rem = torch.arange(X_all.shape[0], device=device, dtype=torch.int64)
        #     sel = torch.tensor([], dtype=torch.int64, device=device)
        #     new_sel = torch.tensor([], dtype=torch.int64, device=device)
        # elif cfg.acquisition.mode.split("_")[0] == "tp":
        #     X_all = torch.cat([X_pool, self.train_list['X']], dim=0)
        #     rem = torch.arange(X_pool.shape[0], device=device, dtype=torch.int64)
        #     sel = torch.arange(X_pool.shape[0], X_all.shape[0], device=device, dtype=torch.int64)
        #     new_sel = torch.tensor([], dtype=torch.int64, device=device)

        with torch.no_grad():
            features_all = get_features_ycov(X_all, ensemble) # [bs_all, feature_dim]
            features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
        
        for _ in range(batch_acquire):
            if sel.shape[0] == 0:
                new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
            else:
                distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
                centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
                largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel))])) # index in bs_sel
                mask = torch.where(centers == largest_cluster, 0.0, -1e10)
                new = torch.argmax(distances[largest_cluster] + mask) # index in rem
            sel = torch.cat([sel, rem[new:new+1]], dim=0)
            new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
            rem = torch_delete(rem, new)
        new_idxs = new_sel
    
    return new_idxs