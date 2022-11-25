import numpy as np
import torch

def eigendecompostition_without_zero_eigs(matrix: torch.Tensor, keep = 'all'): 
    """
    Select eigenvectors for nonzero eigenvalues.
    
    Inputs: 
    =======
    matrix: torch.Tensor
        symmetric matrix
    keep: Union[str, int] = 'all'
        if int, keeps only this number of largest (in magnitude) eigenvalues
        otherwise, all nonzero eigenvalues are kept
        
    Returns:
    =======
    eigenvals_nonzero: Tensor
        nonzero eigenvalues
    eigenvecs_nonzero: Tensor
        corresponding eigenvalues
    """
    eigvals, eigvecs = torch.linalg.eigh(matrix)
    if keep == 'all':
        indices_ = eigvals.nonzero()
    else:
        indices_ = torch.topk(torch.abs(eigvals), k = keep).indices

    dim = matrix.shape[0]
    num_entries = len(indices_)
    eigenvals_ = eigvals[indices_[:]].reshape((num_entries,))
    eigenvecs_ = eigvecs[:, indices_[:]].reshape((dim,num_entries))
    return eigenvals_, eigenvecs_


