import numpy as np
import torch

def eigendecompostition_without_zero_eigs(matrix: torch.Tensor, keep = 'all', eigval_cutoff = 0): 
    """
    Select eigenvectors for nonzero eigenvalues.
    
    Inputs: 
    =======
    matrix: torch.Tensor
        symmetric matrix
    keep: Union[str, int] = 'all'
        if int, keeps only this number of largest (in magnitude) eigenvalues
        otherwise, all nonzero eigenvalues are kept
    eigval_cutoff: float 
        i >0 keeps only (in magnitude) eigenvalues larger than cutoff
        otherwise, all nonzero eigenvalues are kept
    
    Returns:
    =======
    eigenvals_nonzero: Tensor
        nonzero eigenvalues
    eigenvecs_nonzero: Tensor
        corresponding eigenvalues
    """
    eigvals_np, eigvecs_np = np.linalg.eigh(matrix.numpy())
    eigvals = torch.Tensor(eigvals_np)
    eigvecs = torch.Tensor(eigvecs_np)
    
    if keep == 'all' and eigval_cutoff == 0:
        indices_ = eigvals.nonzero()
    elif isinstance(keep, int) and eigval_cutoff == 0:
        indices_ = torch.topk(torch.abs(eigvals), k = keep).indices
    elif isinstance(keep, str) and eigval_cutoff > 0:
        indices_ = torch.argwhere(torch.abs(eigvals) > eigval_cutoff)
    elif isinstance(keep, int) and eigval_cutoff > 0:
        raise ValueError( "Specify either top k number of eigenvalues" \
                         +"to keep or eigenvalue cutoff.")

    dim = matrix.shape[0]
    num_entries = len(indices_)
    if num_entries == 0 : # no entries meet criterion
        eigenvals_ = torch.ones(1)
        eigenvecs_ = torch.zeros(dim).unsqueeze(0)
    eigenvals_ = eigvals[indices_[:]].reshape((num_entries,))
    eigenvecs_ = eigvecs[:, indices_[:]].reshape((dim,num_entries))
    return eigenvals_, eigenvecs_


if __name__ == "__main__": 
    mat = torch.normal(0,1, size =(10,10))
    mat_ = mat + mat.T
    eigendecompostition_without_zero_eigs(mat, keep = 'all', eigval_cutoff = 0)