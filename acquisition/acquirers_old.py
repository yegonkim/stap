from typing import Any, Dict, List, Tuple, Callable, Union, Optional
from collections.abc import Iterable

# torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# logger
import wandb
from omegaconf import OmegaConf

from utils import torch_delete
from . import ACQUISITION_FUNCTIONS
from .feature_functions import get_features_ycov

OPTIMIZERS = {"SGD": torch.optim.SGD, "Adam": torch.optim.Adam, "LBFGS": torch.optim.LBFGS, "SGHMC": SGHMC}

class Acquirer:
    def __init__(self, cfg: OmegaConf, sim: Sim):
        self.cfg = cfg
        self.sim = sim
        self.acq = ACQUISITION_FUNCTIONS[cfg.acquisition.function]
        self.acq_V = ACQUISITION_FUNCTIONS[cfg.acquisition_V.function]
        self.acq_p = ACQUISITION_FUNCTIONS[cfg.perturbation.function]
        self.batch_acquire = cfg.active.batch_acquire
        if cfg.use_V:
            self.train_list = {'X': torch.Tensor().to(cfg.device), 'Y': torch.Tensor().to(cfg.device),
                           'dX': torch.Tensor().to(cfg.device), 'dY': torch.Tensor().to(cfg.device)}
        else:
            self.train_list = {'X': torch.Tensor().to(cfg.device), 'Y': torch.Tensor().to(cfg.device)}
        self.cost = 0
        # if "pool" in cfg.acquisition.init_method:
    
    def pool_init(self, cfg, sim):
        self.P_pool = sim.get_params(cfg.acquisition.pool_size)
        self.X_pool = sim.query_in(self.P_pool) * cfg.acquisition.pool_scale
        if cfg.pool_add_noise:
            noise = torch.randn_like(self.X_pool)
            norm = torch.linalg.vector_norm(noise, dim=tuple(range(1, noise.dim())), keepdim=True)
            new_norm = torch.clamp(norm, min=cfg.perturbation.eps_l, max=cfg.perturbation.eps_u)
            noise = noise / norm * new_norm
            self.X_pool += noise

    def reset(self):
        if self.cfg.use_V:
            self.train_list = {'X': torch.Tensor().to(self.cfg.device), 'Y': torch.Tensor().to(self.cfg.device),
                           'dX': torch.Tensor().to(self.cfg.device), 'dY': torch.Tensor().to(self.cfg.device)}
        else:
            self.train_list = {'X': torch.Tensor().to(self.cfg.device), 'Y': torch.Tensor().to(self.cfg.device)}
        self.cost = 0
        # if "pool" in self.cfg.acquisition.init_method:
        self.P_pool = self.sim.get_params(self.cfg.acquisition.pool_size)
        self.X_pool = self.sim.query_in(self.P_pool)

    def query(self, **queries): # X: [self.batch_acquire, ...], dX: [self.batch_acquire, ...]
        with torch.no_grad():
            X = queries['X']
            Y = self.sim.query_out(X).to(self.cfg.device)
            self.train_list['X'] = torch.cat([self.train_list['X'], X], dim=0)
            self.train_list['Y'] = torch.cat([self.train_list['Y'], Y], dim=0)
            self.cost += X.shape[0]

        # with torch.no_grad():
        #     if not self.cfg.use_V:
        #         # P = queries['P']
        #         X = queries['X']
        #         Y = self.sim.query_out(X).to(self.cfg.device)
        #         self.train_list['X'] = torch.cat([self.train_list['X'], X], dim=0)
        #         self.train_list['Y'] = torch.cat([self.train_list['Y'], Y], dim=0)
        #         self.cost += 1 * X.shape[0]
        #         # self.P_list = torch.cat([self.P_list, P], dim=0)
        #     else:
        #         X = queries['X']
        #         dX = queries['dX']
        #         # X, dX = self.sim.query_in_and_grad(P, dP)
        #         Y, dY = self.sim.query_out_and_grad(X, dX)
        #         self.train_list['X'] = torch.cat([self.train_list['X'], X], dim=0)
        #         self.train_list['Y'] = torch.cat([self.train_list['Y'], Y], dim=0)
        #         self.train_list['dX'] = torch.cat([self.train_list['dX'], dX], dim=0)
        #         self.train_list['dY'] = torch.cat([self.train_list['dY'], dY], dim=0)
        #         self.cost += 2 * X.shape[0]
        #         # self.P_list = torch.cat([self.P_list, P], dim=0)
        #         # self.dP_list = torch.cat([self.dP_list, dP], dim=0)

    def suggest(self, ensemble:List, trial: int=0):
        for model in ensemble:
            model.eval()
        
        sim = self.sim
        cfg = self.cfg

        def sample_dX(p, X):
            norm = torch.linalg.vector_norm(p, dim=tuple(range(1, p.dim())), keepdim=True) / np.sqrt((p.numel() // p.shape[0]))
            new_norm = torch.clamp(norm, min=cfg.perturbation.eps_l, max=cfg.perturbation.eps_u).detach() + norm - norm.detach()
            dX = p / norm * new_norm
            dX *= torch.linalg.vector_norm(X, dim=tuple(range(1, X.dim())), keepdim=True) / np.sqrt((X.numel() // X.shape[0]))
            return dX
        
        # if cfg.all.gradient_steps>0:
        #     if cfg.all.both:
        #         P = self.sim.get_params(self.batch_acquire)
        #         X = sim.query_in(P)
        #         dX = sample_dX(torch.randn_like(X), X)
        #         optim_class = OPTIMIZERS[cfg.all.gradient_params.name]
        #         Z_1 = P.clone().detach().requires_grad_()
        #         Z_2 = dX.clone().detach().requires_grad_()
        #         optimizer_query = optim_class([Z_1, Z_2], lr=cfg.all.gradient_params.lr)
        #         with torch.no_grad():
        #             features_pool = get_features_ycov(self.X_pool, ensemble, sim, cfg)
        #         def closure():
        #             optimizer_query.zero_grad()
        #             X = sim.query_in(Z_1)
        #             X_ = (X+sample_dX(Z_2, X)).unsqueeze(0)
        #             data = {'X': X_}
        #             loss = -self.acq(data, ensemble, sim, trial, cfg, self, features_target=features_pool)
        #             loss.backward()
        #             return loss
        #         for i in range(1, cfg.all.gradient_steps+1):
        #             optimizer_query.step(closure)
        #         X = sim.query_in(Z_1)
        #         X_p = X + sample_dX(Z_2, X)
        #         self.query(X=X_p)
        #     else:
        #         P = self.sim.get_params(self.batch_acquire)
        #         optim_class = OPTIMIZERS[cfg.all.gradient_params.name]
        #         Z_1 = P.clone().detach().requires_grad_()
        #         optimizer_query = optim_class([Z_1], lr=cfg.all.gradient_params.lr)
        #         with torch.no_grad():
        #             features_pool = get_features_ycov(self.X_pool, ensemble, sim, cfg)
        #         def closure():
        #             optimizer_query.zero_grad()
        #             X_ = (sim.query_in(Z_1)).unsqueeze(0)
        #             data = {'X': X_}
        #             loss = -self.acq(data, ensemble, sim, trial, cfg, self, features_target=features_pool)
        #             loss.backward()
        #             return loss
        #         for i in range(1, cfg.all.gradient_steps+1):
        #             optimizer_query.step(closure)
        #         X_p = sim.query_in(Z_1)
        #         # print(torch.linalg.norm(X_p, dim=tuple(range(1, X_p.dim()))))
        #         self.query(X=X_p)
        #     return None

        # def constrain(P):
        #     return torch.clamp(P, min=sim.lbound, max=sim.ubound)
        # def clip(P):
        #     return torch.clamp(P, min=sim.lbound, max=sim.ubound).detach() + P - P.detach()
        # def sigmoid(logP):
        #     return torch.sigmoid(logP) * (sim.ubound - sim.lbound) + sim.lbound
        # def inv_sigmoid(P):
        #     return torch.logit(torch.clamp((P - sim.lbound) / (sim.ubound - sim.lbound), min=1e-7, max=1-1e-7))
        
        # features_target_remember = None
        with torch.no_grad():
            # if cfg.acquisition.init_method == "random":
            #     P = self.sim.get_params(self.batch_acquire) # [self.batch_acquire, ...]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            #         # dP = dP / torch.norm(dP, dim=tuple(range(1, dP.dim())), keepdim=True) # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "X_random_search":
            #     num_candidates = cfg.acquisition.pool_size
            #     cand_P = self.sim.get_params(num_candidates * self.batch_acquire) # [num_candidates * self.batch_acquire, ...]
            #     cand_P = cand_P.view(num_candidates, self.batch_acquire, *(cand_P.shape[1:])) # [num_candidates, self.batch_acquire, ...]
            #     # cand_X = sim.query_in(cand_P) # [num_candidates, self.batch_acquire, ...]
            #     utilities = self.acq({'P': cand_P}, ensemble, sim, trial, cfg, self) # [num_candidates]
            #     index = torch.argmax(utilities)
            #     P = cand_P[index] # [self.batch_acquire, ...]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            #         # dP = dP / torch.norm(dP, dim=tuple(range(1, dP.dim())), keepdim=True) # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "X_greedy_search":
            #     num_candidates = cfg.acquisition.pool_size
            #     P = torch.Tensor()
            #     for i in range(self.batch_acquire):
            #         cand_P = self.sim.get_params(num_candidates).unsqueeze(1) # [num_candidates, 1, ...]
            #         if P.shape[0] > 0:
            #             cand_P  = torch.cat([torch.stack([P]*num_candidates, dim=0), cand_P], dim=1) # [num_candidates, current_batch+1, ...]
            #         utilities = self.acq({'P': cand_P}, ensemble, sim, trial, cfg, self) # [num_candidates]
            #         index = torch.argmax(utilities)
            #         P = cand_P[index] # [current_batch, ...]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "XV_random_search":
            #     assert cfg.use_V
            #     num_candidates = cfg.acquisition.pool_size
            #     cand_P = self.sim.get_params(num_candidates * self.batch_acquire) # [num_candidates * self.batch_acquire, ...]
            #     cand_P = cand_P.view(num_candidates, self.batch_acquire, *(cand_P.shape[1:])) # [num_candidates, self.batch_acquire, ...]
            #     cand_dP = torch.randn_like(cand_P) # [num_candidates, self.batch_acquire, ...]
            #     # cand_dP = cand_dP / torch.norm(cand_dP, dim=tuple(range(2, cand_dP.dim())), keepdim=True) # [num_candidates, self.batch_acquire, ...]
            #     # cand_X, cand_dX = sim.query_in_and_grad(cand_P, cand_dP) # [num_candidates, self.batch_acquire, ...]
            #     utilities = self.acq_V({'P': cand_P, 'dP': cand_dP}, ensemble, sim, trial, cfg, self) # [num_candidates]
            #     index = torch.argmax(utilities)
            #     P = cand_P[index] # [self.batch_acquire, ...]
            #     dP = cand_dP[index] # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "X_cmaes":
            #     P0 = sim.get_params(self.batch_acquire) # [self.batch_acquire, ...]
            #     popsize = cfg.acquisition.cmaes_params.popsize
            #     es = cma.CMAEvolutionStrategy(P0.flatten().cpu().numpy(), sim.ubound-sim.lbound, {'bounds': [sim.lbound, sim.ubound], 'verbose': -9, 'popsize': popsize})
            #     for i in range(cfg.acquisition.cmaes_params.steps):
            #         es_cand = es.ask()
            #         P = torch.tensor(np.array(es_cand)).float().to(cfg.device) # [100, self.batch_acquire*fid]
            #         P = P.view(popsize, *P0.shape) # [100, self.batch_acquire, ...]
            #         P = constrain(P)
            #         # X = sim.query_in(P) # [100, self.batch_acquire, ...]
            #         utilities = self.acq({'P': P}, ensemble, sim, trial, cfg, self) # [100]
            #         losses = -utilities
            #         es.tell(es_cand, losses.detach().cpu().numpy())
            #     P = torch.tensor(es.result.xbest).float().to(cfg.device) # [self.batch_acquire*fid]
            #     P = P.view(*P0.shape) # [self.batch_acquire, ...]
            #     P = constrain(P)
            #     if cfg.use_V:
            #         dP = torch.randn_like(P)
            # elif cfg.acquisition.init_method == "XV_cmaes":
            #     assert cfg.use_V
            #     raise NotImplementedError
            # elif cfg.acquisition.init_method == "X_coreset":
            #     if cfg.acquisition.mode.split("_")[0] == "p":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = P_pool
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     elif cfg.acquisition.mode.split("_")[0] == "tp":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = torch.cat([P_pool, self.P_list], dim=0)
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.arange(P_pool.shape[0], P_all.shape[0], device=cfg.device, dtype=torch.int64)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     get_features = FEATURE_FUNC[cfg.acquisition.mode.split("_")[-1]]
            #     with torch.no_grad():
            #         features_all = get_features(P_all, ensemble, sim, cfg).flatten(start_dim=1) # [bs_all, feature_dim]
            #     features_pool_remember = features_all[rem]
                
            #     for _ in range(self.batch_acquire):
            #         if sel.shape[0] == 0:
            #             new = torch.randint(0, rem.shape[0], (1,)).item()
            #         else:
            #             distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0) # [bs_sel, bs_rem]
            #             min_dist = torch.min(distances, dim=0).values # [bs_rem]
            #             new = torch.argmax(min_dist) # index in rem
            #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
            #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
            #         rem = torch_delete(rem, new)
            #     P = P_all[new_sel]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "X_kmeanspp":
            #     if cfg.acquisition.mode.split("_")[0] == "p":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = P_pool
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     elif cfg.acquisition.mode.split("_")[0] == "tp":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = torch.cat([P_pool, self.P_list], dim=0)
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.arange(P_pool.shape[0], P_all.shape[0], device=cfg.device, dtype=torch.int64)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     with torch.no_grad():
            #         features_all = get_features_ycov(P_all, ensemble, sim, cfg) # [bs_all, N, dim]
            #         features_target_remember = features_all[rem]
            #         features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]

            #     for _ in range(self.batch_acquire):
            #         if sel.shape[0] == 0:
            #             new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
            #         else:
            #             distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
            #             scores = torch.min(distances, dim=0).values # [bs_rem]
            #             # sample from the unnormalized probability scores
            #             new = torch.multinomial(scores, 1) # index in rem
            #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
            #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
            #         rem = torch_delete(rem, new)
            #     P = P_all[new_sel]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            # elif cfg.acquisition.init_method == "X_lcmd":
            #     if cfg.acquisition.mode.split("_")[0] == "p":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = P_pool
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     elif cfg.acquisition.mode.split("_")[0] == "tp":
            #         P_pool = sim.get_params(cfg.acquisition.pool_size)
            #         P_all = torch.cat([P_pool, self.P_list], dim=0)
            #         rem = torch.arange(P_pool.shape[0], device=cfg.device, dtype=torch.int64)
            #         sel = torch.arange(P_pool.shape[0], P_all.shape[0], device=cfg.device, dtype=torch.int64)
            #         new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
            #     with torch.no_grad():
            #         features_all = get_features_ycov(P_all, ensemble, sim, cfg)
            #         features_target_remember = features_all[rem]
            #         features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]

            #     for _ in range(self.batch_acquire):
            #         if sel.shape[0] == 0:
            #             new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
            #         else:
            #             distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
            #             centers = torch.argmin(distances, dim=0) # [bs_rem]: indices in bs_sel
            #             largest_cluster = torch.argmax(torch.tensor([torch.sum(distances[i, centers==i]) for i in range(len(sel))])) # index in bs_sel
            #             mask = torch.where(centers == largest_cluster, 0.0, -1e10)
            #             new = torch.argmax(distances[largest_cluster] + mask) # index in rem
            #         sel = torch.cat([sel, rem[new:new+1]], dim=0)
            #         new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
            #         rem = torch_delete(rem, new)
            #     P = P_all[new_sel]
            #     if cfg.use_V:
            #         dP = torch.randn_like(P) # [self.batch_acquire, ...]
            if cfg.acquisition.init_method == "pool_random":
                weights = torch.ones(1, self.X_pool.shape[0])
                index = torch.multinomial(weights, num_samples=self.batch_acquire, replacement=False)
                index = index[0] # [self.batch_acquire]
                X = self.X_pool[index]
                self.X_pool = torch_delete(self.X_pool, index)
            elif cfg.acquisition.init_method == "pool_X_random_search":
                num_candidates = cfg.acquisition.pool_size
                weights = torch.ones(num_candidates, self.X_pool.shape[0])
                cand_index = torch.multinomial(weights, num_samples=self.batch_acquire, replacement=False) # [num_candidates, self.batch_acquire]
                assert cand_index.shape[0] == num_candidates
                cand_X = self.X_pool[cand_index] # [num_candidates, self.batch_acquire, ...]
                # cand_X = sim.query_in(cand_X) # [num_candidates, self.batch_acquire, ...]
                utilities = self.acq({'X': cand_X}, ensemble, sim, trial, cfg, self) # [num_candidates]
                index = torch.argmax(utilities)
                X = cand_X[index] # [self.batch_acquire, ...]
                self.X_pool = torch_delete(self.X_pool, cand_index[index])
            elif cfg.acquisition.init_method == "pool_X_greedy_search":
                X = torch.Tensor().to(cfg.device)
                features_target = get_features_ycov(torch.cat([self.X_pool, self.train_list['X']], dim=0), ensemble, sim, cfg)
                for _ in range(self.batch_acquire):
                    cand_index = torch.randint(0, self.X_pool.shape[0], (cfg.acquisition.pool_size,))
                    cand_X = self.X_pool[cand_index] # [num_candidates, ...]
                    cand_X = cand_X.unsqueeze(1) # [num_candidates, 1, ...]
                    if X.shape[0] > 0:
                        cand_X  = torch.cat([torch.stack([X]*cand_X.shape[0], dim=0), cand_X], dim=1) # [num_candidates, current_batch+1, ...]
                    utilities = self.acq({'X': cand_X}, ensemble, sim, trial, cfg, self, features_target=features_target) # [num_candidates]
                    index = torch.argmax(utilities)
                    X = cand_X[index] # [current_batch, ...]
                    self.X_pool = torch_delete(self.X_pool, cand_index[index])
            elif cfg.acquisition.init_method == "pool_X_kmeanspp":
                if cfg.acquisition.mode.split("_")[0] == "p":
                    X_all = self.X_pool
                    rem = torch.arange(X_all.shape[0], device=cfg.device, dtype=torch.int64)
                    sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                    new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                elif cfg.acquisition.mode.split("_")[0] == "tp":
                    X_all = torch.cat([self.X_pool, self.train_list['X']], dim=0)
                    rem = torch.arange(self.X_pool.shape[0], device=cfg.device, dtype=torch.int64)
                    sel = torch.arange(self.X_pool.shape[0], X_all.shape[0], device=cfg.device, dtype=torch.int64)
                    new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                with torch.no_grad():
                    features_all = get_features_ycov(X_all, ensemble, sim, cfg) # [bs_all, feature_dim]
                    features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
                
                for _ in range(self.batch_acquire):
                    if sel.shape[0] == 0:
                        new = torch.argmax(torch.norm(features_all[rem], dim=1)) # index in rem
                    else:
                        distances = torch.cdist(features_all[sel].unsqueeze(0), features_all[rem].unsqueeze(0)).squeeze(0).square() # [bs_sel, bs_rem]
                        scores = torch.min(distances, dim=0).values
                        new = torch.multinomial(scores, 1)
                    sel = torch.cat([sel, rem[new:new+1]], dim=0)
                    new_sel = torch.cat([new_sel, rem[new:new+1]], dim=0)
                    rem = torch_delete(rem, new)
                self.X_pool = X_all[rem]
                X = X_all[new_sel]
            elif cfg.acquisition.init_method == "pool_X_lcmd":
                if cfg.acquisition.mode.split("_")[0] == "p":
                    X_all = self.X_pool
                    rem = torch.arange(X_all.shape[0], device=cfg.device, dtype=torch.int64)
                    sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                    new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                elif cfg.acquisition.mode.split("_")[0] == "tp":
                    X_all = torch.cat([self.X_pool, self.train_list['X']], dim=0)
                    rem = torch.arange(self.X_pool.shape[0], device=cfg.device, dtype=torch.int64)
                    sel = torch.arange(self.X_pool.shape[0], X_all.shape[0], device=cfg.device, dtype=torch.int64)
                    new_sel = torch.tensor([], dtype=torch.int64, device=cfg.device)
                with torch.no_grad():
                    features_all = get_features_ycov(X_all, ensemble, sim, cfg) # [bs_all, feature_dim]
                    features_all = features_all.flatten(start_dim=1) # [bs_all, N*dim]
                
                for _ in range(self.batch_acquire):
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
                self.X_pool = X_all[rem]
                X = X_all[new_sel]
            else:
                raise NotImplementedError

        # if cfg.acquisition.gradient_steps>0:
        #     optim_class = OPTIMIZERS[cfg.acquisition.gradient_params.name]
        #     if cfg.acquisition.gradient_params.constrain=='clip':
        #         Z = clip(P).clone().detach().unsqueeze(0).requires_grad_()
        #     elif cfg.acquisition.gradient_params.constrain=='sigmoid':
        #         Z = inv_sigmoid(P).clone().detach().unsqueeze(0).requires_grad_()
        #     optimizer_query = optim_class([Z], lr=cfg.acquisition.gradient_params.lr)
        #     if cfg.acquisition.gradient_params.use_SAM:
        #         optimizer_query = SAM(optimizer_query, rho=cfg.acquisition.gradient_params.rho)
            
        #     if cfg.acquisition.gradient_params.fix_target:
        #         if features_target_remember is None:
        #             with torch.no_grad():
        #                 P_target = sim.get_params(cfg.acquisition.target_size, mode=cfg.acquisition.target_mode)
        #                 features_target = get_features_ycov(P_target, ensemble, sim, cfg)
        #         else:
        #             features_target = features_target_remember
                    
        #         P_val = sim.get_params(cfg.acquisition.target_size, mode=cfg.acquisition.target_mode)
        #         P_best = P.unsqueeze(0).detach().clone()
        #         i_best = 0
        #         with torch.no_grad():
        #             features_val = get_features_ycov(P_val, ensemble, sim, cfg)
        #             loss_val_best = -self.acq({'P': P.unsqueeze(0)}, ensemble, sim, trial, cfg, self, features_target=features_val)
                
        #         early_stop_counter = 0

            # for i in range(1, cfg.acquisition.gradient_steps+1):
            #     def closure():
            #         optimizer_query.zero_grad()
            #         if cfg.acquisition.gradient_params.constrain=='clip':
            #             P_in = clip(Z)
            #         elif cfg.acquisition.gradient_params.constrain=='sigmoid':
            #             P_in = sigmoid(Z)
            #         data = {'P': P_in}
            #         if cfg.acquisition.gradient_params.fix_target:
            #             loss = -self.acq(data, ensemble, sim, trial, cfg, self, features_target=features_target)
            #         else:
            #             loss = -self.acq(data, ensemble, sim, trial, cfg, self)
            #         loss.backward()
            #         return loss, P_in.detach()
            #     loss, P = optimizer_query.step(closure)
            #     wandb.log({f"acquisition/loss": loss})
            #     if cfg.acquisition.gradient_params.fix_target:
            #         with torch.no_grad():
            #             loss_val = -self.acq({'P': P}, ensemble, sim, trial, cfg, self, features_target=features_val)
            #         wandb.log({f"acquisition/loss_val": loss_val})
            #         if loss_val < loss_val_best:
            #             P_best = P.detach().clone()
            #             loss_val_best = loss_val
            #             i_best = i
            #         else:
            #             early_stop_counter += 1
            #             if early_stop_counter >= cfg.acquisition.gradient_params.early_stop and cfg.acquisition.gradient_params.early_stop>=0:
            #                 break
            # if cfg.acquisition.gradient_params.fix_target and cfg.acquisition.gradient_params.early_stop>=0:
            #     P = P_best
            #     wandb.log({f"acquisition/early_stop": i_best, f"acquisition/loss_val_best": loss_val_best})
            # P = P.detach().squeeze(0)

        # P = constrain(P + torch.randn_like(P) * cfg.acquisition.perturb)
        
        if cfg.perturb:
            X_p = X.clone().detach()
            with torch.no_grad():
                features_target = get_features_ycov(torch.cat([self.X_pool, self.train_list['X']], dim=0), ensemble, sim, cfg)
                if cfg.perturbation.mode=="random":
                    p = torch.randn_like(X_p)
                    dX = sample_dX(p, X_p)
                    X_p = X_p + dX
                elif cfg.perturbation.mode=="greedy":
                    for i in range(1, X_p.shape[0]+1):
                        p = torch.randn(cfg.perturbation.num_candidates, *X_p.shape[1:]).to(cfg.device)
                        dX = sample_dX(p, X_p[i-1].unsqueeze(0))
                        X_ = X_p[:i].unsqueeze(0) + torch.cat([torch.zeros_like(X_p[:i-1].unsqueeze(0)).repeat(cfg.perturbation.num_candidates, *((1,)*X_p.dim())), dX.unsqueeze(1)], dim=1)
                        utilities = self.acq_p({'X': X_}, ensemble, sim, trial, cfg, self, features_target=features_target)
                        index = torch.argmax(utilities)
                        dX = dX[index]
                        X_p[i-1] += dX
                else:
                    raise NotImplementedError
            score_0 = self.acq_p({'X': X_p.unsqueeze(0)}, ensemble, sim, trial, cfg, self, features_target=features_target)
            wandb.log({f"perturbation/score_0": score_0.item()})
            if cfg.perturbation.gradient_steps > 0:
                p = X_p - X
                X, p = X.unsqueeze(0), p.unsqueeze(0)
                p = p.detach().clone().requires_grad_()
                optim_class = OPTIMIZERS[cfg.perturbation.gradient_params.name]
                optimizer_query = optim_class([p], lr=cfg.perturbation.gradient_params.lr)
                # if cfg.acquisition.gradient_params.use_SAM:
                #     optimizer_query = SAM(optimizer_query, rho=cfg.acquisition.gradient_params.rho)
                def closure():
                        optimizer_query.zero_grad()
                        loss = -self.acq_p({'X': X+sample_dX(p, X)}, ensemble, sim, trial, cfg, self, features_target=features_target)
                        loss.backward()
                        return loss
                for i in range(1, cfg.perturbation.gradient_steps+1):
                    optimizer_query.step(closure)
                X, p = X.squeeze(0), p.squeeze(0)
                # print(sample_dX(p).shape)
                # print(sample_dX(p).norm(dim=1) /32)
                X_p = X + sample_dX(p, X)
                score_1 = self.acq_p({'X': X_p.unsqueeze(0)}, ensemble, sim, trial, cfg, self, features_target=features_target)
                wandb.log({f"perturbation/score_1": score_1.item(), f"perturbation/score_diff": score_1.item()-score_0.item()})
                # print(score_1-score_0)
            self.query(X=X_p)
            # print(X_p.norm(dim=1))
            # print(self.acq({'X': X_p.unsqueeze(0)}, ensemble, sim, trial, cfg, self, features_target=features_target))
        
        if not (cfg.perturbation.just_p and cfg.perturb):
            # self.query(X=X*10)
            # print(torch.linalg.norm(X, dim=tuple(range(1, X.dim()))))
            self.query(X=X)
    
    def acquire(self, ensemble:List, trial:int=0):
        self.suggest(ensemble, trial)
        sim = self.sim
        cfg = self.cfg
        
        # X = data['X']
        # dX = data['dX']

        # if not self.cfg.use_V:
        #     P = data['P']
        #     # X = sim.query_in(P)
        # else:
        #     P = data['P']
        #     dP = data['dP']

        # with torch.no_grad():
        #     utility = self.acq({'P': P.unsqueeze(0)}, ensemble, sim, trial, cfg, self).item()
        #     metric = {f"acquisition/utility": utility, f"acquisition/trial": trial}
        #     if self.cfg.use_V:
        #         metric["acquisition/utility_with_grad"] = self.acq_V({'P': P.unsqueeze(0), 'dP': dP.unsqueeze(0)}, ensemble, sim, trial, cfg, self).item()

        #     P_prev = self.P_list[torch.randperm(self.P_list.shape[0])[:min(1000, P.shape[0])]]
        #     features = self.sim.query_in(P).flatten(start_dim=1) # [bs, feature_dim]
        #     features_prev = self.sim.query_in(P_prev).flatten(start_dim=1) # [bs_prev, feature_dim]
        #     features_all = torch.cat([features_prev, features], dim=0) # [bs_all, feature_dim]
        #     distances = torch.cdist(features_all.unsqueeze(0), features.unsqueeze(0)).squeeze(0) # [bs_all, bs]
        #     distances = distances + 1e10 * torch.cat([torch.zeros(features_prev.shape[0], features.shape[0]), torch.eye(features.shape[0])], dim=0).to(features.device) # [bs_all, bs]
        #     min_dist = torch.min(distances, dim=0).values # [bs]
        #     diversity_mean = torch.mean(min_dist) # [1]
        #     diversity_min = torch.min(min_dist) # [1]
        #     metric["acquisition/diversity_mean"] = diversity_mean
        #     metric["acquisition/diversity_min"] = diversity_min
        #     metric["acquisition/diversity_n_overlap"] = torch.sum(min_dist < 1e-2*np.sqrt(features.shape[1])).item()
            
        #     distances = torch.cdist(features.unsqueeze(0), features.unsqueeze(0)).squeeze(0)
        #     distances = distances + 1e10 * torch.eye(features.shape[0]).to(features.device)
        #     min_dist = torch.min(distances, dim=0).values
        #     metric["acquisition/diversity_self_min"] = torch.min(min_dist).item()
        #     metric["acquisition/diversity_self_mean"] = torch.mean(min_dist).item()
        #     metric["acquisition/diversity_self_n_overlap"] = torch.sum(min_dist < 1e-2*np.sqrt(features.shape[1])).item()

        #     # MMD using gaussian kernel
        #     P_prev = self.P_list
        #     # P_prev = self.P_list[torch.randperm(self.P_list.shape[0])[:min(100, P.shape[0])]]
        #     P_all = torch.cat([P_prev, P], dim=0)
        #     P_pool = self.sim.get_params(cfg.acquisition.pool_size, mode='sobol')
        #     features_all = self.sim.query_in(P_all).flatten(start_dim=1) # [bs_all, feature_dim]
        #     features_pool = self.sim.query_in(P_pool).flatten(start_dim=1) # [bs_pool, feature_dim]
        #     dim = features_all.shape[1]
        #     dist_all = torch.cdist(features_all.unsqueeze(0), features_all.unsqueeze(0)).squeeze(0).square() # [bs_all, bs_all]
        #     dist_pool = torch.cdist(features_pool.unsqueeze(0), features_pool.unsqueeze(0)).squeeze(0).square() # [bs_pool, bs_pool]
        #     dist_cross = torch.cdist(features_all.unsqueeze(0), features_pool.unsqueeze(0)).squeeze(0).square() # [bs_all, bs_pool]
        #     sigma=np.sqrt(dim)/2
        #     k_xx = torch.exp(-dist_all / (2 * sigma**2))
        #     k_yy = torch.exp(-dist_pool / (2 * sigma**2))
        #     k_xy = torch.exp(-dist_cross / (2 * sigma**2))
        #     mmd = torch.mean(k_xx) + torch.mean(k_yy) - 2 * torch.mean(k_xy)
        #     metric["acquisition/mmd_gaussian"] = mmd.item()

        #     metric["acquisition/max_abs"] = P.abs().max(dim=1).values.mean().item()
        #     metric["acquisition/mean_abs"] = P.abs().mean(dim=1).mean().item()

        # wandb.log(metric)

        

        # if not cfg.active.batch_acquire_random==0:
        #     random_P = self.sim.get_params(cfg.active.batch_acquire_random)
        #     if self.cfg.use_V:
        #         random_dP = torch.randn_like(random_P)
        #         self.query(P=random_P, dP=random_dP)
        #     else:
        #         self.query(P=random_P)

        return self.train_list
