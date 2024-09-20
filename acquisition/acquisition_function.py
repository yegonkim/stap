
import torch
from utils import torch_expand, split_model, ensemble_mean_model

# This class is used to calculate the main acquisition functions (EER and MF-EER)
# It should be reinstantiated for each acquisition step, since it calculates the prior expected error using the current ensemble
class EER_Calculator:
    def __init__(self, ensemble: list, X: torch.Tensor, L: int, eval_batch_size: int, device, mode='EER'):
        self.ensemble = ensemble
        self.eval_batch_size = eval_batch_size
        self.device = device
        self.X = X
        self.L = L
        self.mode = mode
        self.prior_variance, self.u_m, self.u = self._prior_variance(ensemble, X, L, eval_batch_size, device)
        
    def __call__(self, S):
        if self.mode == 'EER':
            return self._eer(self.ensemble, self.X, self.L, S, self.eval_batch_size, self.device)
        elif self.mode == 'MF-EER':
            return self._mf_eer(self.ensemble, self.X, S, self.eval_batch_size, self.device)

    @torch.no_grad()
    def _prior_variance(self, ensemble: list, X: torch.Tensor, L: int, eval_batch_size: int=256, device='cuda'):
        # X has shape [bs, channels, (nx)]
        # S has shape [bs, nt-1] with boolean values
        X = X.to(device)
        bs = X.shape[0]
        M = len(ensemble)

        for model in ensemble:
            model.eval()

        u_m = []
        for m in range(M):
            model_m = ensemble[m]
            pred = [X]
            for t in range(L):
                pred.append(split_model(model_m, eval_batch_size)(pred[-1]))
            pred = torch.stack(pred, dim=1) # [bs, nt, 1, nx]
            u_m.append(pred)
        u_m = torch.stack(u_m, dim=0) # [n_models, bs, nt, 1, nx]

        u = u_m.mean(dim=0, keepdim=True) # [1, bs, nt, 1, nx]

        prior_error =(u - u_m).square() # [n_models, bs, nt, 1, nx]
        prior_error = prior_error.mean(dim=0) # [bs, nt, 1, nx]
        prior_error = prior_error.flatten(start_dim=1).sum(dim=1) # [bs]

        return prior_error, u_m, u


    @torch.no_grad()
    # expected error reduction
    def _eer(self, ensemble: list, X: torch.Tensor, L: int, S: torch.Tensor, eval_batch_size: int=256, device='cuda'):
        # X has shape [bs, channels, (nx)]
        # S has shape [bs, nt-1] with boolean values
        X = X.to(device)
        bs = X.shape[0]
        L = S.shape[1]
        assert L == self.L
        M = len(ensemble)
        u_m = self.u_m

        for model in ensemble:
            model.eval()

        u_mk = []
        for m in range(M):
            model_m = ensemble[m]
            u_m_ = []
            for k in range(M):
                model_k = ensemble[k]
                pred = [X]
                for t in range(L):
                    X_t = pred[-1].clone()
                    if S[:,t].any():
                        X_t[S[:,t]] = split_model(model_m, eval_batch_size)(X_t[S[:,t]])
                    if (~S[:,t]).any():
                        X_t[~S[:,t]] = split_model(model_k, eval_batch_size)(X_t[~S[:,t]])
                    pred.append(X_t)
                pred = torch.stack(pred, dim=1) # [bs, nt, 1, nx]
                u_m_.append(pred)
            u_m_ = torch.stack(u_m_, dim=0) # [n_models(k), bs, nt, 1, nx]
            u_mk.append(u_m_)
        u_mk = torch.stack(u_mk, dim=0) # [n_models(m), n_models(k), bs, nt, 1, nx]

        prior_error = 2 * self.prior_variance # [bs]   E[(X_1 - X_2)^2] = 2 * E[(X - E[X])^2]
        posterior_error = (u_m[:,None] - u_mk).square() # [n_models(m), n_models(k), bs, nt, 1, nx]
        posterior_error = posterior_error.mean(dim=(0,1)) # [bs, nt, 1, nx]
        posterior_error = posterior_error.flatten(start_dim=1).sum(dim=1) # [bs]

        scores = prior_error - posterior_error # [bs]
        assert scores.shape == (bs,)
        scores[scores < 0] = 0

        return scores


    @torch.no_grad()
    # Mean field expected error reduction
    def _mf_eer(self, ensemble: list, X: torch.Tensor, S: torch.Tensor, eval_batch_size: int=256, device='cuda'):
        # X has shape [bs, channels, (nx)]
        # S has shape [bs, nt-1] with boolean values
        X = X.to(device)
        bs = X.shape[0]
        L = S.shape[1]
        M = len(ensemble)
        u = self.u

        for model in ensemble:
            model.eval()

        u_k = []
        for k in range(M):
            model_k = ensemble[k]
            pred = [X]
            for t in range(L):
                X_t = pred[-1].clone()
                if S[:,t].any():
                    X_t[S[:,t]] = split_model(ensemble_mean_model(ensemble), eval_batch_size)(X_t[S[:,t]])
                if (~S[:,t]).any():
                    X_t[~S[:,t]] = split_model(model_k, eval_batch_size)(X_t[~S[:,t]])
                pred.append(X_t)
            pred = torch.stack(pred, dim=1) # [bs, nt, 1, nx]
            u_k.append(pred)
        u_k = torch.stack(u_k, dim=0) # [n_models(k), bs, nt, 1, nx]


        prior_error = self.prior_variance
        posterior_error = (u - u_k).square() # [n_models(m), n_models(k), bs, nt, 1, nx]
        posterior_error = posterior_error.mean(dim=0) # [bs, nt, 1, nx]
        posterior_error = posterior_error.flatten(start_dim=1).sum(dim=1) # [bs]

        scores = prior_error - posterior_error # [bs]
        assert scores.shape == (bs,)
        scores[scores < 0] = 0
        return scores.cpu()
    