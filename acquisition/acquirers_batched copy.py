import torch
from tqdm import tqdm
from .feature_functions import get_features_ycov, get_features_hidden
from utils import direct_model, trajectory_model, split_model

class Acquirer_batched:
    def __init__(self, ensemble, Y, train_nts, **cfg):
        self.device = cfg.get('device', 'cpu')
        self.ensemble = ensemble
        self.Y = Y.to(self.device)
        self.nt = Y.shape[1]
        self.train_nts = train_nts
        self.cfg = cfg
        self.eval_batch_size = cfg.get('eval_batch_size', 256)
        self.cost = 0

    def select_initial_batch(self, initial_selection_method, batch_size):
        bs = self.Y.shape[0]
        if initial_selection_method == "random":
            weights = (self.train_nts < self.nt).float()
            batch_indices = torch.multinomial(weights, num_samples=batch_size, replacement=False)
        elif initial_selection_method == "variance":
            indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() < self.nt], device=self.device)
            scores = self._compute_scores(starting_time=0, batch_indices=indices).sum(dim=-1)
            _, top_indices = torch.topk(scores, k=batch_size)
            batch_indices = indices[top_indices]
        elif initial_selection_method == "stochastic":
            indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() < self.nt], device=self.device)
            scores = self._compute_scores(starting_time=0, batch_indices=indices).sum(dim=-1)
            probabilities = scores / scores.sum()
            top_indices = torch.multinomial(probabilities, num_samples=batch_size, replacement=False)
            batch_indices = indices[top_indices]
        else:
            raise ValueError(f"Initial selection method {initial_selection_method} not implemented.")
        
        return batch_indices

    def post_selection(self, selection_method, batch_indices):
        if len(batch_indices) == 0:
            return self.train_nts
        
        optimization_method = self.cfg.get('optimization_method', 'greedy')

        nt = self.nt
        bs = len(batch_indices)

        if selection_method == "all":
            self.train_nts[batch_indices] = self.nt
            return self.train_nts
        

        if type(selection_method) == int:
            self.train_nts[batch_indices] = selection_method+1
            return self.train_nts

        if selection_method == "variance_prior":
            scores = self._compute_scores(starting_time=0, batch_indices=batch_indices) # [bs, nt]
            # Compute cumulative sum
            scores = scores.cumsum(dim=1) # [bs, nt]
        elif selection_method == "variance_direct":
            scores_list = []
            for starting_time in range(nt):
                scores = self._compute_scores(starting_time=starting_time, batch_indices=batch_indices)
                scores_list.append(scores)
            scores = torch.stack(scores_list, dim=2) # [bs, nt, nt]
            scores = scores.sum(dim=1) # [bs, nt]
            scores = scores[:, 0:1] - scores # [bs, nt]
        # select_nts = torch.zeros(scores.shape[0], device=self.device, dtype=torch.int64) # [bs]
        select_nts = torch.ones(scores.shape[0], device=self.device, dtype=torch.int64) * (nt-1) # [bs]
        if optimization_method == "greedy":
            for _ in range(10):
                for i in range(bs):
                    greedy_utility = (torch.sum(scores[torch.arange(bs, device=self.device), select_nts][torch.arange(bs)!=i]) + scores[i, torch.arange(nt, device=self.device)]) / (select_nts[torch.arange(bs)!=i].sum() + torch.arange(nt, device=self.device))
                    greedy_utility[0] = 0 # division by zero led to nan
                    select_nts[i] = torch.argmax(greedy_utility)
        elif optimization_method == "individual":
            for i in range(bs):
                utility = scores[i, :] / torch.arange(nt, device=self.device)
                utility[0] = 0 # division by zero led to nan
                select_nts[i] = torch.argmax(utility)
        else:
            raise ValueError(f"Optimization method {optimization_method} not implemented.")
        
        print(select_nts)
        self.train_nts[batch_indices] = select_nts + 1
        return self.train_nts

    def select_flexible_batch(self, selection_method, batch_size):
        bs = self.Y.shape[0]
        if selection_method == "random":
            pass
            

    def select(self):
        scenario = self.cfg.get('scenario', 'fixed')
        batch_size = self.cfg.get('batch_acquire', 1)

        if scenario == 'fixed':
            initial_selection_method = self.cfg.get('initial_selection_method', 'random')
            post_selection_method = self.cfg.get('post_selection_method', 'all')
            batch_indices = self.select_initial_batch(initial_selection_method, batch_size)
            self.post_selection(post_selection_method, batch_indices)
        elif scenario == 'flexible':
            selection_method = self.cfg.get('flexible_selection_method', 'random')
            for acquire_step in range(self.nt - 1):
                self.select_flexible_batch(selection_method, batch_size)
        else:
            raise ValueError(f'Invalid scenario: {scenario}')

        return self.train_nts



    def _compute_scores(self, starting_time=0, batch_indices=None):
        # starting time: 0, 1, 2, ..., nt-1
        nt = self.nt
        ensemble = self.ensemble

        if batch_indices is None:
            batch_indices = torch.arange(self.Y.shape[0], device=self.device)

        if starting_time >= nt-1:
            return torch.zeros(len(batch_indices), nt, device=self.device)

        # assert torch.all(torch.logical_or(train_nts == nt, train_nts == 1)) # all or nothing

        X_unlabelled = self.Y[batch_indices, 0:1]
        X_unlabelled = X_unlabelled.to(self.device)

        if starting_time > 0:
            pred_X_starting_time = []
            with torch.no_grad():
                for model in ensemble:
                    model.eval()
                    pred_X_starting_time.append(split_model(direct_model(model, starting_time), self.eval_batch_size)(X_unlabelled))
            pred_X_starting_time = torch.stack(pred_X_starting_time, dim=0) # [n_models, bs, 1, nx]
            pred_X_starting_time = torch.mean(pred_X_starting_time, dim=0) # [bs, 1, nx]
        else:
            pred_X_starting_time = X_unlabelled

        # Compute predictions for remaining time steps
        pred_trajectories = []
        with torch.no_grad():
            for model in ensemble:
                model.eval()
                pred_trajectories.append(split_model(trajectory_model(model, (nt-1)-starting_time), self.eval_batch_size)(pred_X_starting_time))
        pred_trajectories = torch.stack(pred_trajectories, dim=0) # [n_models, bs_unlabelled, nt-1-starting_time, nx]

        # Compute variance across ensemble
        variance = torch.var(pred_trajectories, dim=0)  # [bs_unlabelled, nt-1-starting_time, nx]
        
        # Compute scores (sum of variances across spatial dimension and time steps)
        scores = variance.sum(dim=-1)  # [bs_unlabelled, nt-1-starting_time]

        # Create a tensor to hold scores for all samples
        all_scores = torch.zeros(X_unlabelled.shape[0], nt, device=self.device)
        all_scores[:, starting_time+1:] = scores

        return all_scores