import torch
import numpy as np

from generate_data import evolve
from utils import torch_expand
from omegaconf import OmegaConf

from utils import split_model


def extract_consecutive_trues(bool_list, value_list):
    result = []
    current_sublist = []
    for i in range(1, len(bool_list)):
        if bool_list[i]:
            if not current_sublist:
                current_sublist.append(value_list[i-1])
            current_sublist.append(value_list[i])
        else:
            if current_sublist:
                result.append(current_sublist)
                current_sublist = []
    if current_sublist:
        result.append(current_sublist)
    return result

@torch.no_grad()
def Y_from_selected(ensemble: list, selected: dict, pool, L, cfg: OmegaConf):
    # selected: {index: S}
    # pool: Pool
    device = cfg.device
    filter = cfg.filter

    X = []
    S_list = []
    for index, S in selected.items():
        X.append(pool[index])
        S_list.append(S)
    X = torch.stack(X, dim=0) # [datasize, 1, nx]
    S = torch.cat(S_list, dim=0) # [datasize, L]
    bs = X.shape[0]
    assert S.ndim == 2
    assert S.shape[0] == X.shape[0] and S.shape[1] == L

    try:
        # for scale_threhold in [2, 1.5, 1.25, 1.1]:
        preds = [torch_expand(X[:,None], 1, len(ensemble))] # [datasize, ensemble_size, c, nx]
        for t in range(L):
            X_t = preds[-1].clone()
            filter_indices = (X_t.mean(dim=1).flatten(1).abs().max(dim=1).values > filter)
            # print(filter_indices)
            if filter_indices.any():
                S[filter_indices, t:] = False
            # print(X_t[S[:,t]].max(), X_t[S[:,t]].min())
            if (S[:,t] == True).any():
                X_t[S[:,t]] = evolve(X_t[S[:,t], :].mean(dim=1), cfg)[:,-1][:,None] # [datasize, ensemble_size, c, nx]
            if (S[:,t] == False).any():
                X_t[~S[:,t]] = torch.stack([split_model(model, cfg.eval_batch_size)(X_t[~S[:,t], i].to(device)).cpu() for i, model in enumerate(ensemble)], dim=1) # [datasize, ensemble_size, c, nx]
            preds.append(X_t)
        preds = torch.stack(preds, dim=2).mean(dim=1) # [datasize, nt, c, nx]
    except:
        raise ValueError('Simulation instability')
    
    Y = []
    for i in range(len(preds)):
        bool_list = [True] + S[i].cpu().tolist()
        value_list = preds[i].cpu()
        Y += extract_consecutive_trues(bool_list, value_list)
    
    for i, traj in enumerate(Y):
        Y[i] = torch.stack(traj, dim=0) # [nt, c, nx]

    return Y

# @torch.no_grad()
# def Y_from_selected(ensemble: list, selected: dict, pool, L, cfg: OmegaConf):
#     # selected: {index: S}
#     # pool: Pool
#     device = cfg.device
#     filter = cfg.filter

#     X = []
#     S_list = []
#     for index, S in selected.items():
#         X.append(pool[index])
#         S_list.append(S)
#     X = torch.stack(X, dim=0) # [datasize, 1, nx]
#     S = torch.cat(S_list, dim=0) # [datasize, L]
#     assert S.ndim == 2
#     assert S.shape[0] == X.shape[0] and S.shape[1] == L

#     try:
#         # for scale_threhold in [2, 1.5, 1.25, 1.1]:
#         preds = [torch_expand(X[:,None], 1, len(ensemble))] # [datasize, ensemble_size, c, nx]
#         for t in range(L):
#             X_t = preds[-1].clone()
#             #TODO: use scale_threshold to predict simulation instability
#             if (S[:,t] == True).any():
#                 X_t[S[:,t]] = evolve(X_t[S[:,t], :].mean(dim=1), cfg)[:,-1][:,None] # [datasize, ensemble_size, c, nx]
#             if (S[:,t] == False).any():
#                 X_t[~S[:,t]] = torch.stack([split_model(model, cfg.eval_batch_size)(X_t[~S[:,t], i].to(device)).cpu() for i, model in enumerate(ensemble)], dim=1) # [datasize, ensemble_size, c, nx]
#             preds.append(X_t)
#         preds = torch.stack(preds, dim=2).mean(dim=1) # [datasize, nt, c, nx]
#     except:
#         raise ValueError('Simulation instability')
    
#     Y = []
#     for i in range(len(preds)):
#         bool_list = [True] + S[i].cpu().tolist()
#         value_list = preds[i].cpu()
#         Y += extract_consecutive_trues(bool_list, value_list)
    
#     for i, traj in enumerate(Y):
#         Y[i] = torch.stack(traj, dim=0) # [nt, c, nx]

#     return Y

@torch.no_grad()
def Y_from_selected_cheat(ensemble: list, selected: dict, pool_with_traj, L, cfg: OmegaConf):
    # selected: {index: S}
    # pool: Pool
    device = cfg.device

    preds = []
    S_list = []
    for index, S in selected.items():
        preds.append(pool_with_traj[index]) # [nt, c, nx]
        S_list.append(S)
    # X = torch.stack(X, dim=0) # [datasize, 1, nx]
    preds = torch.stack(preds, dim=0) # [datasize, nt, c, nx]
    S = torch.cat(S_list, dim=0) # [datasize, L]
    assert S.ndim == 2
    # assert S.shape[0] == X.shape[0] and S.shape[1] == L

        
    Y = []
    for i in range(len(preds)):
        bool_list = [True] + S[i].cpu().tolist()
        value_list = preds[i].cpu()
        Y += extract_consecutive_trues(bool_list, value_list)

    for i, traj in enumerate(Y):
        Y[i] = torch.stack(traj, dim=0) # [nt, c, nx]
    
    datasize = sum([selected[i].sum() for i in selected])
    return Y