import torch
import torch.nn.functional as F


def get_laplacian_matrix(adj):
    '''
    Calculating laplacian matrix.
    
    Args:
        adj: adjacent matrix or discrete similarity matrix.
    
    Returns:
        normalized laplacian matrix.
    '''
    EPS = 1e-6
    # check and remove self-loop
    I = torch.eye(adj.shape[0], device=adj.device)
    if torch.diag(adj).sum().item()+EPS >= adj.shape[0]:
        tmp = adj - I
    else:
        tmp = adj

    D = tmp.sum(dim=1).clip(1)
    D_rsqrt = torch.rsqrt(D)
    D_rsqrt = torch.diag(D_rsqrt)
    lap_mat = I - D_rsqrt@tmp@D_rsqrt
    return lap_mat

def similarity(z1: torch.Tensor, z2: torch.Tensor):
    z1 = F.normalize(z1)
    z2 = F.normalize(z2)
    return torch.mm(z1, z2.t())