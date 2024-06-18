from typing import List
import torch as tc
import torch.nn as nn
#from bptt.models import Model


def pearson_r(X: tc.Tensor, Y: tc.Tensor):
    corr_mat = tc.corrcoef(tc.stack([X, Y]))
    return corr_mat[0, 1]

def mean_trial_correlation(X: tc.Tensor, Y: tc.Tensor):
    N = X.size(1)
    eps = tc.randn_like(X) * 1e-5
    X_eps_noise = X + eps
    rs = tc.zeros(N)
    for n in range(N):
        rs[n] = pearson_r(X_eps_noise[:, n], Y[:, n])
    return rs.mean()

def flatten_tensors(W):
    flattened_tensors = []
    for tensor in W:
        if tensor.numel() == 1:  # check if tensor has only one element
            flattened_tensor = tensor.reshape(1)  # reshape to 1D tensor
        else:
            flattened_tensor = tensor.flatten()  # flatten tensor
        flattened_tensors.append(flattened_tensor)
    return tc.stack(flattened_tensors).T

def compute_correlation_matrix(x):
    """
    Compute correlation matrix for PyTorch tensor x.
    
    Args:
    - x: torch.Tensor with shape (N, D)
    
    Returns:
    - corr_matrix: torch.Tensor with shape (D, D), where corr_matrix[i, j] is the correlation
                   coefficient between x[:, i] and x[:, j]
    """
    # Subtract mean and divide by standard deviation
    x = (x - x.mean(dim=0)) / x.std(dim=0)
    
    # Compute correlation matrix
    corr_matrix = tc.matmul(x.T, x) / x.shape[0]
    corr_matrix = corr_matrix - tc.eye(corr_matrix.size(0))
    return corr_matrix

def unfold_W(W):
    W = W.detach().cpu()
    return flatten_tensors(W)


@tc.no_grad()
def autoencoder_reconstruction_correlation(X: tc.Tensor, E: nn.Module,
                                           D: nn.Module) -> float:
    rs = tc.zeros(len(X))
    for i, x in enumerate(X):
        rs[i] = mean_trial_correlation(x, D(E(x)))
    return rs

@tc.no_grad()
def generative_reconstruction_correlation(X: tc.Tensor, S: tc.Tensor,
                                          model: nn.Module) -> List[float]:
    ntr = len(X)
    rs = tc.zeros(ntr)
    for i in range(ntr):
        x, s = X[i], S[i]
        xg, _ = model.generate_free_trajectory(x, s, x.size(0), i)
        rs[i] = mean_trial_correlation(x, xg)
    return rs

@tc.no_grad()
def comp_W_corr(W):
    ws = unfold_W(W)
    return compute_correlation_matrix(ws)
