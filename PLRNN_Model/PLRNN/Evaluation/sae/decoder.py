import torch.nn as nn
import math
from sae.layers import SkipFCLayer


class DecoderMLP(nn.Module):
    def __init__(self, in_dim, out_dim, norm=nn.LayerNorm, activation=nn.ReLU, n_blocks_per_depth=1):
        super(DecoderMLP, self).__init__()

        # compute max depth
        d = math.floor(math.log2(out_dim / in_dim))

        # initialize layers
        layers = [nn.Linear(in_dim, out_dim // 2**(d-1))]
        for i in range(1, d+1):
            for _ in range(n_blocks_per_depth):
                layers.append(SkipFCLayer(out_dim // 2**(d-i), norm, activation))
            if i != d:
                layers.append(nn.Linear(out_dim // 2**(d-i), out_dim // 2**(d-i-1)))

        layers.append(nn.Linear(out_dim, out_dim*2))
        layers.append(SkipFCLayer(out_dim*2, norm, activation))
        layers.append(nn.Linear(out_dim*2, out_dim))
        # gather
        self.MLP = nn.Sequential(*layers)
        self.lin_path = nn.Linear(in_dim, out_dim)
    
    def forward(self, X):
        return self.MLP(X) + self.lin_path(X)
    

class Decoder_Custom(nn.Module):
    def __init__(self, in_dim, out_dim, activation=nn.ReLU(), num_layers=2):
        super(Decoder_Custom, self).__init__()

        # compute max depth
        d = math.floor((out_dim*2 - in_dim)/num_layers)

        # initialize layers
        layers = [nn.Linear(out_dim*2, out_dim), activation]
        for i in range(num_layers):
            layers.append(nn.Linear(out_dim*2-d*(i+1), out_dim*2-d*i))
            layers.append(activation)
        
        layers.append(nn.Linear(in_dim, out_dim*2-d*(i+1)))
        layers = layers[::-1]

        # gather
        self.MLP = nn.Sequential(*layers)

    def forward(self, X):
        return self.MLP(X)


