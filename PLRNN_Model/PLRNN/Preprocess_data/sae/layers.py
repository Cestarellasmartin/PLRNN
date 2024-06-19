import torch as tc
import torch.nn as nn


class SkipFCLayer(nn.Module):
    def __init__(self, num_units, norm, activation):
        super(SkipFCLayer, self).__init__()
        # weight
        self.layer = nn.Linear(num_units, num_units)
        # activation
        self.activ = activation()
        
        # norm
        if norm == nn.LayerNorm:
            self.norm = norm(num_units)
        elif norm == nn.BatchNorm1d:
            self.norm = norm(num_units)
        else:
            self.norm = nn.Identity()

    def forward(self, X):
        X_ = self.activ(self.layer(X))
        return self.norm(X + X_)


"""
# NOT READY YET
class PermuteDims(nn.Module):
    def __init__(self):
        super(PermuteDims, self).__init__()

    def forward(self, X):
        sz = X.size()
        return tc.reshape(X, (sz[0], sz[2], sz[1], sz[3]))

class ResNetBlock(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_sz, padding, Phi = nn.ReLU):
        super(ResNetBlock, self).__init__()

        self.C1 = nn.Conv2d(in_ch, out_ch, kernel_sz, 1, padding)
        self.N1 = nn.BatchNorm2d(out_ch)

        self.C2 = nn.Conv2d(out_ch, out_ch, kernel_sz, 1, padding)
        self.N2 = nn.BatchNorm2d(out_ch)

        self.Phi = Phi()

        # "identity connection"
        if in_ch == out_ch:
            self.IdC = nn.Identity()
        else:
            self.IdC = nn.Conv2d(in_ch, out_ch, 1, 1, 0, bias=False)

    def forward(self, X):
        XX = self.C2(self.Phi(self.N1(self.C1(X))))
        return self.Phi(self.N2(XX + self.IdC(X)))
"""
