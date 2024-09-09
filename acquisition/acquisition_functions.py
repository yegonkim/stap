# torch
import torch
import torch.nn as nn
import einops

# logger
from omegaconf import OmegaConf

from .feature_functions import get_features_ycov

# Main functions

def constant(X, ensemble, **cfg):
    return torch.zeros(X.shape[0], device=X.device)

def entropy(X, ensemble, **cfg):
    std = cfg.get('std', 1e-2)
    bs, bs_acquire = X.shape[:2]
    features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble) # [bs*bs_acquire, N, dim]
    features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    covariance_like = torch.einsum('bcnd,bcmd->bnm', features, features) # [bs, N, N]
    log_det_cov = torch.logdet(covariance_like + std**2*torch.eye(covariance_like.shape[1], device=features.device).unsqueeze(0)) # [bs]
    score = log_det_cov / 2 + (features.shape[1] + features.shape[3]) / 2 * torch.log(torch.tensor(2 * 3.14159265358979323846, device=features.device))
    return score

def variance(X, ensemble, **cfg):
    bs, bs_acquire = X.shape[:2]
    features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble) # [bs*bs_acquire, N, dim]
    features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    trace = torch.einsum('bcnd,bcnd->b', features, features) # [bs]
    score = trace
    score /= bs_acquire
    return score

def bait(X, ensemble, **cfg):
    X_pool = cfg.get('X_pool', None)
    std = cfg.get('std', 1)

    bs, bs_acquire = X.shape[:2]

    features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble) # [bs*bs_acquire, N, dim]
    features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    features = features.permute(0,2,1,3).reshape(bs, features.shape[2], -1) # [bs, N, bs_acquire*dim]

    features_target = get_features_ycov(X_pool, ensemble) # [bs, N, dim]

    # if not 'dP' in data:
    #     features = get_features_ycov(P.reshape(bs*bs_acquire, *P.shape[2:]), ensemble, sim, cfg) # [bs*bs_acquire, N, dim]
    #     features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    #     features = features.permute(0,2,1,3).reshape(bs, features.shape[2], -1) # [bs, N, bs_acquire*dim]
    # else:
    #     dP = data['dP']
    #     features, features_grad = get_features_ycov_grad(P.reshape(bs*bs_acquire, *P.shape[2:]), dP.reshape(bs*bs_acquire, *dP.shape[2:]), ensemble, sim, cfg)
    #     features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    #     features_grad = features_grad.view(bs, bs_acquire, *features_grad.shape[1:]) # [bs, bs_acquire, N, dim]
    #     features_cat = torch.cat([features, features_grad], dim=-1) # [bs, bs_acquire, N, 2*dim]
    #     features_cat = features_cat.permute(0,2,1,3).reshape(bs, features_cat.shape[2], -1) # [bs, N, bs_acquire*2*dim]
    #     features = features_cat

    # if 'features_target' in kwargs:
    #     features_target = kwargs['features_target'] # [bs_target, N, dim]
    # else:
    #     with torch.no_grad():
    #         target_P = sim.get_params(cfg.acquisition.target_size, mode=cfg.acquisition.target_mode) # [bs_pool, dim]
    #         features_target = get_features_ycov(target_P, ensemble, sim, cfg) # [bs_target, N, dim]
    # # if acquirer.P_list.shape[0] > 0:
    #     features_prev = get_features_ycov(acquirer.P_list, ensemble, sim, cfg) # [bs_prev, N, dim]
    #     features_target = torch.cat([features_prev, features_target], dim=0) # [bs_target, N, dim]
    features_target = features_target.permute(1,0,2).reshape(features_target.shape[1], -1) # [N, bs_target*dim]
    
    features = features.double()
    features_target = features_target.double()

    # trace_K_nn = torch.einsum('ni,ni->...', features_target, features_target) # []
    temp_woodbury = torch.einsum('bni,bmi->bnm', features, features) # [bs, N, N]
    temp_woodbury = temp_woodbury + torch.eye(features.shape[1], device=features.device).unsqueeze(0) * std**2 # [bs, N, N]
    woodbury_inv = torch.cholesky_inverse(torch.linalg.cholesky(temp_woodbury)) # [bs, N, N]
    # woodbury_inv = torch.cholesky_inverse(torch.vmap(robust_cholesky)(temp_woodbury)) # [bs, N, N]
    woodbury_inv = woodbury_inv * std**2 # [bs, N, N]
    trace_subtract = (-1/(std**4) * torch.einsum('bti,bts,bsj,nk,bnj,mk,bmi->b', features, woodbury_inv, features, features_target, features, features_target, features)
                    + 1/(std**2) * torch.einsum('nk,bni,mk,bmi->b', features_target, features, features_target, features)) # [bs]
    # trace_posterior = trace_K_nn - trace_subtract # [bs]
    # score = -1 * trace_posterior
    score = trace_subtract.float()
    
    return score / features_target.shape[0]

# def kmeans_(X, ensemble, **cfg):
#     X_pool = cfg.get('X_pool', None)
#     features_target = get_features_ycov(X_pool, ensemble) # [bs_pool, N, dim]
#     bs, bs_acquire = X.shape[:2]

#     features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble, sim, cfg) # [bs*bs_acquire, N, dim]
#     features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
#     features = features.flatten(start_dim=2)
    
#     features_target = features_target.flatten(start_dim=1) # [bs_pool, N*dim]

#     if cfg.acquisition.mode.split("_")[0] == "p":
#         features_all = features
#     elif cfg.acquisition.mode.split("_")[0] == "tp":
#         X_prev = acquirer.train_list['X'] # [bs_prev, ...]
#         # P_prev = P_prev[torch.randperm(P_prev.shape[0])[:min(cfg.acquisition.kc_params.batch, P_prev.shape[0])]] # [bs_prev, ...]
#         with torch.no_grad():
#             features_prev = get_features_ycov(X_prev, ensemble, sim, cfg) # [bs_prev, N, dim]
#             features_prev = features_prev.flatten(start_dim=1) # [bs_prev, N*dim]
#             features_prev = einops.repeat(features_prev, '... -> b ...', b=bs) # [bs, bs_prev, feature_dim]
#         features_all = torch.cat([features_prev, features], dim=1) # [bs, bs_inner+bs_prev, feature_dim]

#     # if not 'dP' in data:
#     #     features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble, sim, cfg) # [bs*bs_acquire, N, dim]
#     #     features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
#     #     features = features.flatten(start_dim=2)
        
#     #     with torch.no_grad():
#     #         if 'features_target' in kwargs:
#     #             features_target = kwargs['features_target'] # [bs_pool, N, dim]
#     #         else:
#     #             P_target = sim.get_params(cfg.acquisition.target_size, mode=cfg.acquisition.target_mode) # [bs_pool, dim]
#     #             features_target = get_features_ycov(P_target, ensemble, sim, cfg) # [bs_pool, N, dim]
#     #         features_target = features_target.flatten(start_dim=1) # [bs_pool, N*dim]

#     #     if cfg.acquisition.mode.split("_")[0] == "p":
#     #         features_all = features
#     #     elif cfg.acquisition.mode.split("_")[0] == "tp":
#     #         P_prev = acquirer.P_list # [bs_prev, ...]
#     #         # P_prev = P_prev[torch.randperm(P_prev.shape[0])[:min(cfg.acquisition.kc_params.batch, P_prev.shape[0])]] # [bs_prev, ...]
#     #         with torch.no_grad():
#     #             features_prev = get_features_ycov(P_prev, ensemble, sim, cfg) # [bs_prev, N, dim]
#     #             features_prev = features_prev.flatten(start_dim=1) # [bs_prev, N*dim]
#     #             features_prev = einops.repeat(features_prev, '... -> b ...', b=bs) # [bs, bs_prev, feature_dim]
#     #         features_all = torch.cat([features_prev, features], dim=1) # [bs, bs_inner+bs_prev, feature_dim]
#     # else:
#     #     dP = data['dP']
#     #     features, features_grad = get_features_ycov_grad(X.reshape(bs*bs_acquire, *X.shape[2:]), dP.reshape(bs*bs_acquire, *dP.shape[2:]), ensemble, sim, cfg) # [bs*bs_acquire, N, dim]
#     #     features = torch.stack([features, features_grad], dim=-1) # [bs*bs_acquire, N, dim, 2]
#     #     features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim, 2]
#     #     features = features.flatten(start_dim=2) # [bs, bs_acquire, N*dim*2]
        
#     #     with torch.no_grad():
#     #         P_target = sim.get_params(cfg.acquisition.target_size, mode=cfg.acquisition.target_mode) # [bs_pool, dim]
#     #         features_target, features_target_grad = get_features_ycov_grad(P_target, ensemble, sim, cfg) # [bs_pool, N, dim]
#     #         features_target = torch.stack([features_target, features_target_grad], dim=-1) # [bs_pool, N, dim, 2]
#     #         features_target = features_target.flatten(start_dim=1) # [bs_pool, N*dim*2]

#     #     if cfg.acquisition.mode.split("_")[0] == "p":
#     #         features_all = features
#     #     elif cfg.acquisition.mode.split("_")[0] == "tp":
#     #         P_prev = acquirer.P_list # [bs_prev, ...]
#     #         with torch.no_grad():
#     #             features_prev, features_prev_grad = get_features_ycov_grad(P_prev, ensemble, sim, cfg) # [bs_prev, N, dim]
#     #             features_prev = torch.stack([features_prev, features_prev_grad], dim=-1) # [bs_prev, N, dim, 2]
#     #             features_prev = features_prev.flatten(start_dim=1) # [bs_prev, N*dim]
#     #             features_prev = einops.repeat(features_prev, '... -> b ...', b=bs) # [bs, bs_prev, feature_dim]
#     #         features_all = torch.cat([features_prev, features], dim=1) # [bs, bs_inner+bs_prev, feature_dim]

#     distances = torch.cdist(features_all, features_target.unsqueeze(0)).square() # [bs, bs_sel, bs_rem]
#     dist_to_center = torch.min(distances, dim=1).values # [bs, bs_rem]: indices in bs_sel
#     score = -1 * torch.mean(dist_to_center, dim=1)

#     return score

# def kmeans(data, ensemble, **cfg):
#     X = data['X']
#     bs, bs_acquire = X.shape[:2]

#     if bs_acquire == 1:
#         score = torch.zeros(bs, device=X.device)
#     else:
#         score = kmeans_(data, ensemble, **cfg)

#     return score

