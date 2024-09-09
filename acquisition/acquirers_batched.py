import torch
from tqdm import tqdm
from .feature_functions import get_features_ycov, get_features_hidden
from utils import direct_model, trajectory_model, split_model, torch_delete, torch_expand
import numpy as np

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

    @torch.no_grad()
    def select_initial_batch(self, initial_selection_method, batch_size):
        bs = self.Y.shape[0]
        if initial_selection_method == "random":
            weights = (self.train_nts == 1).float()
            batch_indices = torch.multinomial(weights, num_samples=batch_size, replacement=False)
        elif initial_selection_method == "variance":
            indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() == 1], device=self.device)
            scores = self._compute_scores(starting_time=0, batch_indices=indices).sum(dim=-1)
            _, top_indices = torch.topk(scores, k=batch_size)
            batch_indices = indices[top_indices]
        elif initial_selection_method == "stochastic":
            indices = torch.tensor([b for b in range(bs) if self.train_nts[b].item() == 1], device=self.device)
            scores = self._compute_scores(starting_time=0, batch_indices=indices).sum(dim=-1)
            probabilities = scores / scores.sum()
            top_indices = torch.multinomial(probabilities, num_samples=batch_size, replacement=False)
            batch_indices = indices[top_indices]
        else:
            raise ValueError(f"Initial selection method {initial_selection_method} not implemented.")
        
        return batch_indices

    @torch.no_grad()
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

    @torch.no_grad()
    def select_flexible_batch(self, selection_method, batch_size, num_random_pool=1000):
        bs = self.Y.shape[0]
        nt = self.nt
        train_nts= self.train_nts
        std = 1e-2

        if "single" in selection_method.split("_"):
            if "zero" in selection_method.split("_"):
                starting_time = train_nts * 0
                mask = train_nts
            elif "last" in selection_method.split("_"):
                starting_time = train_nts - 1
                mask = train_nts
            elif "ignore" in selection_method.split("_"):
                starting_time = (train_nts > 0).int() * (nt-1)
                mask = (train_nts > 0).int() * nt
            else:
                raise ValueError(f"Selection method {selection_method} not implemented.")

            if "variance" in selection_method.split("_"):
                stochastic_method = "plain"
                if "prior" in selection_method.split("_"):
                    scores = self._compute_variance_prior(self.Y, starting_time) # [bs, nt]
                elif "direct" in selection_method.split("_"):
                    scores = self._compute_variance_direct(self.Y, starting_time) # [bs, nt]
                else:
                    raise ValueError(f"Selection method {selection_method} not implemented.")
            elif "mutual" in selection_method.split("_"):
                stochastic_method = "log"
                if "self" in selection_method.split("_"):
                    scores = self._compute_mutual_self(self.Y, starting_time) # [bs, nt]
                elif "transductive" in selection_method.split("_"):
                    scores = self._compute_mutual_transductive(self.Y, starting_time)
                print(scores)
            else:
                raise ValueError(f"Selection method {selection_method} not implemented.")

            selected = []
            total_cost = 0
            costs = torch.arange(nt, device=self.device).unsqueeze(0) - (train_nts.unsqueeze(1)-1) # [bs, nt]
            utility = scores / costs # [bs, nt]
            utility[torch.arange(bs), :mask+1] = -np.inf
            while total_cost < batch_size * (nt-1):
                if "max" in selection_method.split("_"):
                    max_indices = torch.argmax(utility.flatten()) # an index in [0, bs*nt)
                    index = max_indices // nt # an index in [0, bs)
                    time = max_indices % nt # an index in [0, nt)
                    assert time < nt
                    cost = costs[index, time]
                    assert cost > 0
                    assert utility[index, time] == torch.max(utility).item()
                elif "stochastic" in selection_method.split("_"):
                    if stochastic_method == "plain":
                        utility_temp = utility.clone()
                        utility_temp[utility_temp < 0] = 0
                        max_indices = torch.multinomial(utility_temp, num_samples=1)
                    elif stochastic_method == "log":
                        gumbel_noise = -torch.log(-torch.log(torch.rand_like(utility)))
                        utility_noisy = utility + gumbel_noise
                        max_indices = torch.argmax(utility_noisy.flatten())
                    index = max_indices // nt
                    time = max_indices % nt
                    assert time < nt
                    cost = costs[index, time]
                    assert cost > 0
                total_cost += cost
                selected.append((index, time))
                utility[index, :] = -np.inf
            selected.pop(-1) # remove the last selection so that the total cost is less than batch_size * (nt-1)
            
            for index, time in selected:
                assert train_nts[index] < time + 1
                train_nts[index] = time + 1
            self.train_nts = train_nts
        elif "batch" in selection_method.split("_"):
            raise NotImplementedError()
        elif "lcmd" in selection_method.split("_"):
            raise NotImplementedError()
        else:
            raise ValueError(f"Selection method {selection_method} not implemented.")

        return self.train_nts

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
            num_random_pool = self.cfg.get('num_random_pool', 1000)
            self.select_flexible_batch(selection_method, batch_size, num_random_pool)
        else:
            raise ValueError(f'Invalid scenario: {scenario}')

        return self.train_nts

    @torch.no_grad()
    def _compute_scores(self, starting_time=0, batch_indices=None):
        # starting time: 0, 1, 2, ..., nt-1
        nt = self.nt
        ensemble = self.ensemble

        if batch_indices is None:
            batch_indices = torch.arange(self.Y.shape[0], device=self.device)
        
        if starting_time >= nt-1:
            return torch.zeros(len(batch_indices), nt, device=self.device)

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
        
        scores_temp = self._compute_variance(pred_X_starting_time, timesteps=nt-1-starting_time) # [bs, nt-1-starting_time]
        scores = torch.zeros(len(batch_indices), nt, device=self.device)
        scores[:, starting_time+1:] = scores_temp
        return scores

    @torch.no_grad()
    def _compute_variance(self, X, timesteps):
        features = self._compute_features(X, timesteps) # [N, bs, max_timesteps, nx]
        variance = torch.sum(features**2, dim=(0, 3)) # [bs, max_timesteps]
        return variance

    @torch.no_grad()
    def _compute_features(self, X, timesteps):
        if type(timesteps) == int:
            timesteps = torch.ones(X.shape[0], device=self.device, dtype=torch.int64) * timesteps
        if type(timesteps) == list:
            timesteps = torch.tensor(timesteps, device=self.device, dtype=torch.int64)
        nt = self.nt
        ensemble = self.ensemble

        X = X.to(self.device) # [bs, 1, nx]
        timesteps = timesteps.to(self.device) # [bs]

        max_timesteps = timesteps.max()

        pred_trajectory = torch_expand(X.unsqueeze(0), 0, len(ensemble)) # [n_models, bs, 1, nx]
        for t in range(max_timesteps):
            indices = timesteps > t
            if not indices.any():
                break
            
            X_t = pred_trajectory[:, indices, t:t+1] # [n_models, bs_indices, 1, nx]
            pred = []
            with torch.no_grad():
                for i, model in enumerate(ensemble):
                    model.eval()
                    pred.append(model(X_t[i]))
            pred = torch.stack(pred, dim=0) # [n_models, bs_indices, 1, nx]
            pred_with_zeros = torch.zeros(len(ensemble), X.shape[0], 1, *X.shape[2:], device=self.device) # [n_models, bs, 1, nx]
            pred_with_zeros[:, indices] = pred
            pred_trajectory = torch.cat([pred_trajectory, pred_with_zeros], dim=2) # [n_models, bs, t+2, nx]

        # Compute variance across ensemble
        mean = torch.mean(pred_trajectory, dim=0, keepdim=True) # [1, bs, max_timesteps+1, nx]
        features = pred_trajectory - mean # [n_models, bs, max_timesteps+1, nx]
        features /= np.sqrt(len(ensemble)-1) # [n_models, bs, max_timesteps+1, nx]
        features = features.flatten(start_dim=3)  # [n_models, bs, max_timesteps+1, nx]
        return features[:, :, 1:, :] # [n_models, bs, max_timesteps, nx]

    @torch.no_grad()
    def _compute_variance_prior(self, Y, starting_time):
        nt = self.nt
        bs = Y.shape[0]
        timesteps = (nt - 1) - starting_time # [bs]
        features_temp = self._compute_features(self.Y[torch.arange(bs), starting_time.cpu()], timesteps.cpu()) # [n_models, bs, max_timesteps, nx]
        features = torch.zeros(len(self.ensemble), bs, nt, features_temp.shape[-1], device=self.device) # [n_models, bs, nt, nx]
        features[:, :, starting_time+1:, :] = features_temp # [n_models, bs, nt, nx]
        variance = torch.sum(features**2, dim=(0, 3)) # [bs, nt]
        scores = torch.cumsum(variance, dim=1) # [bs, nt]
        return scores
    
    @torch.no_grad()
    def _compute_variance_direct(self, Y, starting_time):
        nt = self.nt
        bs = Y.shape[0]
        timesteps = (nt - 1) - starting_time
        ensemble = self.ensemble

        X = self.Y[torch.arange(bs), starting_time.cpu()] # [bs, 1, nx]
        assert X.shape == (bs, 1, Y.shape[2])
        max_timesteps = timesteps.max()
        pred_trajectory = torch_expand(X.unsqueeze(0), 0, len(ensemble)) # [n_models, bs, 1, nx]
        assert pred_trajectory.shape == (len(ensemble), bs, 1, X.shape[2])
        for t in range(max_timesteps):
            indices = timesteps > t
            if not indices.any():
                break
            
            X_t = pred_trajectory[:, indices, t:t+1] # [n_models, bs_indices, 1, nx]
            assert X_t.shape == (len(ensemble), indices.sum(), 1, X.shape[2])
            pred = []
            with torch.no_grad():
                for i, model in enumerate(ensemble):
                    model.eval()
                    pred.append(model(X_t[i]))
            pred = torch.stack(pred, dim=0) # [n_models, bs_indices, 1, nx]
            assert pred.shape == (len(ensemble), indices.sum(), 1, X.shape[2])
            pred_with_zeros = torch.zeros(len(ensemble), X.shape[0], 1, *X.shape[2:], device=self.device) # [n_models, bs, 1, nx]
            assert pred_with_zeros.shape == (len(ensemble), bs, 1, X.shape[2])
            pred_with_zeros[:, indices] = pred
            pred_trajectory = torch.cat([pred_trajectory, pred_with_zeros], dim=2) # [n_models, bs, t+2, nx]
            assert pred_trajectory.shape == (len(ensemble), bs, t+2, X.shape[2])
        pred_trajectory = pred_trajectory.mean(dim=0) # [bs, max_timesteps+1, nx]
        assert pred_trajectory.shape == (bs, max_timesteps+1, X.shape[2])
        pred = torch.zeros(bs, nt, *X.shape[2:], device=self.device)
        pred[torch.arange(bs), starting_time:] = pred_trajectory[torch.arange(bs), :timesteps+1] # [bs, nt, nx]
        
        scores_list = []
        for i in range(nt):
            indices = starting_time <= i
            if not indices.any():
                scores_list.append(torch.zeros(bs, device=self.device))
                continue
            features = self._compute_features(pred[indices, i:i+1], (nt-1)-i) # [n_models, bs_indices, nt-1-i, nx]
            assert features.shape == (len(ensemble), indices.sum(), nt-1-i, X.shape[2])
            variance_temp = torch.sum(features**2, dim=(0, 2, 3)) # [bs_indices]
            variance = torch.zeros(bs, device=self.device)
            variance[indices] = variance_temp
            scores_list.append(variance)
        variance = torch.stack(scores_list, dim=1) # [bs, nt]
        assert (variance[:,-1]==0).all()
        assert variance.shape == (bs, nt)

        scores = variance[torch.arange(bs), starting_time].unsqueeze(1) - variance # [bs, nt]
        scores[torch.arange(bs), :starting_time] = 0
        assert scores[0, starting_time[0]] == 0
        return scores
    
    @torch.no_grad()
    def _compute_mutual_self(self, Y, starting_time):
        nt = self.nt
        bs = Y.shape[0]
        ensemble = self.ensemble
        
        # Compute features for all timesteps
        features = self._compute_features(Y[:, starting_time], nt - starting_time)  # [n_models, bs, max_timesteps, nx]
        
        # Compute self-mutual information
        mutual_info = torch.zeros(bs, nt, device=self.device)
        
        for t in range(starting_time, nt):
            # Compute entropy of the current timestep
            entropy_t = self._compute_entropy(features[:, :, t-starting_time])
            
            # Compute joint entropy of current timestep with itself
            joint_entropy = self._compute_joint_entropy(features[:, :, t-starting_time], features[:, :, t-starting_time])
            
            # Mutual information is I(X;X) = H(X) + H(X) - H(X,X) = H(X)
            mutual_info[:, t] = entropy_t
        
        return mutual_info

    @torch.no_grad()
    def _compute_mutual_transductive(self, Y, starting_time):
        nt = self.nt
        bs = Y.shape[0]
        ensemble = self.ensemble
        
        # Compute features for all timesteps
        features = self._compute_features(Y[:, starting_time], nt - starting_time)  # [n_models, bs, max_timesteps, nx]
        
        # Select a random subset of the test set (assuming self.Y_test exists)
        num_test_samples = min(100, self.Y_test.shape[0])
        test_indices = torch.randperm(self.Y_test.shape[0])[:num_test_samples]
        Y_test_subset = self.Y_test[test_indices]
        
        # Compute features for the test subset
        test_features = self._compute_features(Y_test_subset[:, 0], nt)  # [n_models, num_test_samples, nt, nx]
        
        # Compute transductive mutual information
        mutual_info = torch.zeros(bs, nt, device=self.device)
        
        for t in range(starting_time, nt):
            # Compute entropy of the current timestep
            entropy_t = self._compute_entropy(features[:, :, t-starting_time])
            
            # Compute entropy of the test set at the same timestep
            entropy_test = self._compute_entropy(test_features[:, :, t])
            
            # Compute joint entropy of current timestep with test set
            joint_entropy = self._compute_joint_entropy(features[:, :, t-starting_time], test_features[:, :, t])
            
            # Mutual information is I(X;Y) = H(X) + H(Y) - H(X,Y)
            mutual_info[:, t] = entropy_t + entropy_test - joint_entropy
        
        return mutual_info

    def _compute_entropy(self, features):
        # Compute entropy using the ensemble
        probs = self._compute_probabilities(features)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=0)
        return entropy

    def _compute_joint_entropy(self, features1, features2):
        # Compute joint entropy using the ensemble
        joint_probs = self._compute_joint_probabilities(features1, features2)
        joint_entropy = -torch.sum(joint_probs * torch.log(joint_probs + 1e-10), dim=(0, 1))
        return joint_entropy

    def _compute_probabilities(self, features):
        # Estimate probabilities using the ensemble
        return torch.softmax(features.mean(dim=0), dim=-1)

    def _compute_joint_probabilities(self, features1, features2):
        # Estimate joint probabilities using the ensemble
        joint_features = torch.cat([features1.unsqueeze(-1), features2.unsqueeze(-1)], dim=-1)
        return torch.softmax(joint_features.mean(dim=0).flatten(start_dim=-2), dim=-1).view(*joint_features.shape[1:-1])

    


def entropy(X, ensemble, **cfg):
    std = cfg.get('std', 1e-2)
    bs, bs_acquire = X.shape[:2]
    features = get_features_ycov(X.reshape(bs*bs_acquire, *X.shape[2:]), ensemble) # [bs*bs_acquire, N, dim]
    features = features.view(bs, bs_acquire, *features.shape[1:]) # [bs, bs_acquire, N, dim]
    covariance_like = torch.einsum('bcnd,bcmd->bnm', features, features) # [bs, N, N]
    log_det_cov = torch.logdet(covariance_like + std**2*torch.eye(covariance_like.shape[1], device=features.device).unsqueeze(0)) # [bs]
    score = log_det_cov / 2 + (features.shape[1] + features.shape[3]) / 2 * torch.log(torch.tensor(2 * 3.14159265358979323846, device=features.device))
    return score
