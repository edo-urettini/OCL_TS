import torch 
import torch.nn as nn

#Define Student-t loss function (should work as standard nn.MSE)
class StudentTLoss(nn.Module):
    def __init__(self, nu=100, scale=1, reduction='mean'):
        super(StudentTLoss, self).__init__()
        self.nu = nu
        self.reduction = reduction

    def forward(self, x, y, scale):
        diff = x - y

        #we ignore the constant term
        loss = 0.5 * (self.nu + 1)  * torch.log1p(diff**2 / (self.nu * scale**2))

        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'sum':
            return torch.sum(loss)
        else:
            return loss