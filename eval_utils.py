import torch
import torch.nn.functional as F
from tqdm import tqdm

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True, device='cpu', eval_batch_size=256):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average
        self.device = device
        self.eval_batch_size = eval_batch_size

    # @torch.no_grad()
    def abs(self, x, y):
        num_examples = x.size()[0]
        device_original = x.device

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0) if x.size()[1] > 1 else 1.0
        
        all_norms = []
        for X, Y in zip(x.split(self.eval_batch_size), y.split(self.eval_batch_size)):
            X, Y = X.to(self.device), Y.to(self.device)
            norms = (h**(self.d/self.p))*torch.norm(X.flatten(start_dim=1) - Y.flatten(start_dim=1), self.p, 1)
            all_norms.append(norms)
        all_norms = torch.cat(all_norms)

        # x,y = x.to(self.device), y.to(self.device)
        # all_norms = (h**(self.d/self.p))*torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms).to(device_original)
            else:
                return torch.sum(all_norms).to(device_original)

        return all_norms.to(device_original)

    @torch.no_grad()
    def rel(self, x, y):
        num_examples = x.size()[0]
        device_original = x.device

        all_rel = []
        for X, Y in zip(x.split(self.eval_batch_size), y.split(self.eval_batch_size)):
            X, Y = X.to(self.device), Y.to(self.device)
            diff_norms = torch.norm(X.flatten(start_dim=1) - Y.flatten(start_dim=1), self.p, 1)
            y_norms = torch.norm(Y.flatten(start_dim=1), self.p, 1)
            all_rel.append(diff_norms/y_norms)
        all_rel = torch.cat(all_rel)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_rel).to(device_original)
            else:
                return torch.sum(all_rel).to(device_original)

        return all_rel.to(device_original)

        # x, y = x.to(self.device), y.to(self.device)
        # diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        # y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        # if self.reduction:
        #     if self.size_average:
        #         return torch.mean(diff_norms/y_norms)
        #     else:
        #         return torch.sum(diff_norms/y_norms)

        # return (diff_norms/y_norms).to(device_original)

    # @torch.no_grad()
    # def mse(self, x, y):
    #     device_original = x.device
    #     all_mse = []
    #     for X, Y in zip(x.split(self.eval_batch_size), y.split(self.eval_batch_size)):
    #         X, Y = X.to(self.device), Y.to(self.device)
    #         X, Y = X.flatten(start_dim=1), Y.flatten(start_dim=1)
    #         mse = torch.norm(X - Y, self.p, 1).pow(2)
    #         # y_norms = torch.norm(Y, self.p, 1).pow(2)
    #         all_mse.append(mse)
    #     all_mse = torch.cat(all_mse)

    #     if self.reduction:
    #         if self.size_average:
    #             return torch.mean(all_mse).to(device_original)
    #         else:
    #             return torch.sum(all_mse).to(device_original)
        
    #     return all_mse.to(device_original)

    #     # x, y = x.to(self.device), y.to(self.device)
    #     # mse = F.mse_loss(x, y, reduction='none')
    #     # mse = mse.mean(dim=tuple(range(1, mse.ndim)))
    #     # return mse.to(device_original)

    @torch.no_grad()
    def mae(self, x, y):
        device_original = x.device
        all_mae = []
        for X, Y in zip(x.split(self.eval_batch_size), y.split(self.eval_batch_size)):
            X, Y = X.to(self.device), Y.to(self.device)
            X, Y = X.flatten(start_dim=1), Y.flatten(start_dim=1)
            mae = torch.abs(X - Y).mean(dim=1)
            all_mae.append(mae)
        all_mae = torch.cat(all_mae)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_mae).to(device_original)
            else:
                return torch.sum(all_mae).to(device_original)
        
        return all_mae.to(device_original)

    def __call__(self, x, y):
        return self.rel(x, y)

def compute_metrics(y, y_pred, d=1, device='cpu', reduction=False) :
    L2_func = LpLoss(d=d, p=2, reduction=reduction, device=device)
    if y.shape != y_pred.shape :
        raise ValueError('y and y_pred must have the same shape')
    l2 = L2_func.abs(y_pred, y) # [bs]
    relative_l2 = L2_func.rel(y_pred, y) # [bs]
    # mse = L2_func.mse(y_pred, y) # [bs]
    mae = L2_func.mae(y_pred, y) # [bs]
    return l2, relative_l2, mae
