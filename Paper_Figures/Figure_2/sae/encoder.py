import torch.nn as nn
import math

from sae.layers import SkipFCLayer

class EncoderMLP(nn.Module):
    def __init__(self, in_dim, out_dim, norm=nn.LayerNorm, activation=nn.ReLU, n_blocks_per_depth=1):
        super(EncoderMLP, self).__init__()
        
        # compute max depth
        d = math.floor(math.log2(in_dim / out_dim))

        # initialize layers
        layers = [nn.Linear(in_dim, in_dim*2), SkipFCLayer(in_dim*2, norm, activation), nn.Linear(in_dim*2, in_dim)]
        for i in range(d):
            for _ in range(n_blocks_per_depth):
                layers.append(SkipFCLayer(in_dim // 2**i, norm, activation))
            if i != d-1:
                layers.append(nn.Linear(in_dim // 2**i, in_dim // 2**(i+1)))
            else:
                layers.append(nn.Linear(in_dim // 2**i, out_dim))

        # gather
        self.MLP = nn.Sequential(*layers)
        self.lin_path = nn.Linear(in_dim, out_dim)


    def forward(self, X):
        return self.MLP(X) + self.lin_path(X)
    

class Encoder_Custom(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU, num_layers=2):
        super(Encoder_Custom, self).__init__()
        
        # compute max depth
        d = math.floor((in_dim*2 - out_dim)/num_layers)

        # initialize layers
        layers = [nn.Linear(in_dim, in_dim*2), activation]
        for i in range(num_layers):
            layers.append(nn.Linear(in_dim*2-d*i, in_dim*2-d*(i+1)))
            layers.append(activation)
        
        layers.append(nn.Linear(in_dim*2-d*(i+1), out_dim))

        # gather
        self.MLP = nn.Sequential(*layers)

    def forward(self, X):
        return self.MLP(X)

