import torch
import torch.nn.functional as F

class LpLoss(object):
    def __init__(self, d=2, p=2, size_average=True, reduction=True):
        super(LpLoss, self).__init__()

        #Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x, y):
        num_examples = x.size()[0]

        #Assume uniform mesh
        h = 1.0 / (x.size()[1] - 1.0) if x.size()[1] > 1 else 1.0

        all_norms = (h**(self.d/self.p))*torch.norm(x.view(num_examples,-1) - y.view(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(all_norms)
            else:
                return torch.sum(all_norms)

        return all_norms

    def rel(self, x, y):
        num_examples = x.size()[0]

        diff_norms = torch.norm(x.reshape(num_examples,-1) - y.reshape(num_examples,-1), self.p, 1)
        y_norms = torch.norm(y.reshape(num_examples,-1), self.p, 1)

        if self.reduction:
            if self.size_average:
                return torch.mean(diff_norms/y_norms)
            else:
                return torch.sum(diff_norms/y_norms)

        return diff_norms/y_norms

    def __call__(self, x, y):
        return self.rel(x, y)

def compute_metrics(y, y_pred, d=1) :
    L2_func = LpLoss(d=d, p=2, reduction=False)
    if y.shape != y_pred.shape :
        raise NotImplementedError
    l2 = L2_func.abs(y, y_pred) # [bs]
    relative_l2 = L2_func.rel(y, y_pred) # [bs]
    mse = F.mse_loss(y_pred, y, reduction='none') # [bs]
    mse = mse.mean(dim=tuple(range(1, mse.ndim)))
    return l2, relative_l2, mse
