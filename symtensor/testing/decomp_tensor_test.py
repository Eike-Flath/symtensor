# %%
import torch
import itertools
import numpy as np
from scipy.special import binom

# %% tags=["active-ipynb", "remove-input"]
# # Module only imports
# from symtensor.decomp_symmtensor import DecompSymmetricTensor
# import symtensor.utils as utils 
# import symtensor.symalg as symalg

# %% tags=["active-py", "remove-cell"]
Script only imports
from .decomp_symmtensor import DecompSymmetricTensor
from . import utils
from . import symalg


# %%
import pytest
from collections import Counter
from symtensor.utils import symmetrize

import itertools
            
def two_comp_test_tensor(d,r):
    A = DecompSymmetricTensor(rank=r, dim=d)
    A.weights = torch.randn(size =(2,))
    A.factors =  torch.randn(size =(2,d))+1
    A.multiplicities = (r,)
    return A
    
def two_factor_test_tensor(d,r, q = 1):
    assert q<r
    A = DecompSymmetricTensor(rank=r, dim=d)
    A.weights = torch.randn(size =(2,2))
    A.factors =  torch.randn(size =(2,d))+1
    A.multiplicities = (r-q,q)
    return A
    
def three_factor_test_tensor(d,r, q = 1):
    assert r>=3
    assert 2*q<r
    A = DecompSymmetricTensor(rank=r, dim=d)
    A.weights = torch.randn(size =(2,2,2))
    A.factors =  torch.randn(size =(2,d))+1
    A.multiplicities = (r-2*q,q,q)
    return A


# %% [markdown]
# # Tests

# %% [markdown]
# ## Instantiation, getting and setting of weights, factors and multiplicities

# %%
if __name__ == "__main__":
    # instantiation of vector
    A = DecompSymmetricTensor(rank = 1, dim =10) 
    print(type(A).mro())
    assert A.rank == 1
    assert A.dim == 10
    
    weights = [0,1]
    factors =  torch.randn(size =(2,10))
    multiplicities = (1,)
    A.weights = weights
    assert (weights == A.weights)
    A.factors = factors 
    assert (factors == A.factors).all()
    A.multiplicities =  multiplicities
    assert (A.multiplicities  == multiplicities)
    
    #outer product a x a x b
    B = DecompSymmetricTensor(rank = 2, dim =10) 
    assert B.rank == 2
    assert B.dim ==10
    
    weights = torch.randn(size =(2,2))
    factors =  torch.randn(size =(2,10))
    multiplicities = (2,1)
    B.weights = weights
    assert (weights == B.weights).all()
    B.factors = factors 
    assert (factors == B.factors).all()
    B.multiplicities =  multiplicities
    assert (B.multiplicities  == multiplicities)


# %%


# %% [markdown]
# ### Indexing
#
# Test indexing for tensors of shape 
#
# $$
# T = \sum_{m} \lambda^m t^m \otimes t^m \otimes \dots t^m
# $$
# with $t^m$ vectors. 

    # %%
    A = DecompSymmetricTensor(rank = 3, dim =5)
    assert all(A[index] == 0 for index in A.indep_iter_repindex())
    
    d = 2
    r = 1
    B_1 = two_comp_test_tensor(d,r)
    assert np.isclose(B_1[0] , B_1.weights[0]*B_1.factors[0,0] 
                              + B_1.weights[1]*B_1.factors[1,0])
    assert np.isclose(B_1[1] , B_1.weights[0]*B_1.factors[0,1] 
                              + B_1.weights[1]*B_1.factors[1,1])
    
    d = 2_2
    r = 2
    B_2 = two_comp_test_tensor(d,r)
    assert np.isclose(B_2[0,0] , B_2.weights[0]*B_2.factors[0,0]**2 
                                  + B_2.weights[1]*B_2.factors[1,0]**2)
    assert np.isclose(B_2[0,1] , B_2.weights[0]*B_2.factors[0,0]*B_2.factors[0,1] 
                                  + B_2.weights[1]*B_2.factors[1,0]*B_2.factors[1,1])
    assert np.isclose(B_2[1,0] , B_2[0,1])
    assert np.isclose(B_2[1,1] , B_2.weights[0]*B_2.factors[0,1]**2 
                                  + B_2.weights[1]*B_2.factors[1,1]**2)
    
    d = 3
    r = 3
    B_3 = two_comp_test_tensor(d,r)
    assert np.isclose(B_3[0,0,0], B_3.weights[0]*B_3.factors[0,0]**3 + B_3.weights[1]*B_3.factors[1,0]**3)
    assert np.isclose(B_3[1,1,1], B_3.weights[0]*B_3.factors[0,1]**3 + B_3.weights[1]*B_3.factors[1,1]**3)
    assert np.isclose(B_3[0,1,1], B_3.weights[0]*B_3.factors[0,0]*B_3.factors[0,1]**2 
                                  + B_3.weights[1]*B_3.factors[1,0]*B_3.factors[1,1]**2)
    assert np.isclose(B_3[0,2,2], B_3.weights[0]*B_3.factors[0,0]*B_3.factors[0,2]**2 
                                  + B_3.weights[1]*B_3.factors[1,0]*B_3.factors[1,2]**2)
    assert np.isclose(B_3[1,2,2], B_3.weights[0]*B_3.factors[0,1]*B_3.factors[0,2]**2 
                                  + B_3.weights[1]*B_3.factors[1,1]*B_3.factors[1,2]**2)
    assert np.isclose(B_3[1,1,0], B_3[0,1,1])
    assert np.isclose(B_3[2,2,0], B_3[0,2,2])
    assert np.isclose(B_3[1,1,2], B_3[2,1,1])
    assert np.isclose(B_3[2,0,2], B_3[0,2,2])
    assert np.isclose(B_3[0,1,2], B_3.weights[0]*B_3.factors[0,0]*B_3.factors[0,1]*B_3.factors[0,2] 
                                  + B_3.weights[1]*B_3.factors[1,0]*B_3.factors[1,1]*B_3.factors[1,2])

# %% [markdown]
# Test indexing for tensors of shape 
# $$
# T = \sum_{m} \lambda^{m,n} \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots v^n}_{l \text{ times}}
# $$
# with $u^m, v^m $ vectors. 

    # %%
    d = 2
    r = 2
    q = 1
    A = DecompSymmetricTensor(rank=r, dim=d)
    A.weights = torch.Tensor([[1,0],[0,0]])
    A.factors =  torch.randn(size =(2,d))
    A.multiplicities = (r-q,q)
    assert np.isclose(A.factors[0,0]**2, A[0,0])
    
    A_1 = DecompSymmetricTensor(rank=r, dim=d)
    A_1.weights = torch.Tensor([[1,0],[0,1]])
    A_1.factors =  torch.randn(size =(2,d))
    A_1.multiplicities = (r-q,q)
    assert np.isclose(A_1.factors[0,0]**2+A_1.factors[1,0]**2, A_1[0,0])
    assert np.isclose(A_1.factors[0,1]**2+A_1.factors[1,1]**2, A_1[1,1])
    assert np.isclose(A_1[1,0], A_1[0,1])
    assert np.isclose(A_1.factors[0,1]*A_1.factors[0,0]
                      +A_1.factors[1,1]*A_1.factors[1,0], A_1[1,0])
    
    r = 3
    q = 1
    A_2 = DecompSymmetricTensor(rank=r, dim=d)
    A_2.weights = torch.Tensor([[1,0],[0,1]])
    A_2.factors =  torch.randn(size =(2,d))
    A_2.multiplicities = (r-q,q)
    assert np.isclose(A_2.factors[0,0]**3+A_2.factors[1,0]**3, A_2[0,0,0])
    assert np.isclose(A_2.factors[0,1]**3+A_2.factors[1,1]**3, A_2[1,1,1])
    assert np.isclose(A_2[1,0,0], A_2[0,0,1])
    assert np.isclose(A_2.factors[0,1]**2*A_2.factors[0,0]
                      +A_2.factors[1,1]**2*A_2.factors[1,0], A_2[1,1,0])

    # %%
    d = 13
    r = 2
    B_1 = two_factor_test_tensor(d,r)
    assert np.isclose(B_1[0,0] , B_1.weights[0,0]*B_1.factors[0,0]**2
                              + B_1.weights[1,1]*B_1.factors[1,0]**2
                              +(B_1.weights[1,0]*B_1.factors[1,0]*B_1.factors[0,0]
                               +B_1.weights[0,1]*B_1.factors[0,0]*B_1.factors[1,0]))
    assert np.isclose(B_1[1,0] , (2*B_1.weights[0,0]*B_1.factors[0,0]*B_1.factors[0,1]
                              + 2*B_1.weights[1,1]*B_1.factors[1,0]*B_1.factors[1,1]
                              +(B_1.weights[1,0]*(B_1.factors[1,0]*B_1.factors[0,1]+B_1.factors[1,1]*B_1.factors[0,0])
                               +B_1.weights[0,1]*(B_1.factors[1,0]*B_1.factors[0,1]+B_1.factors[1,1]*B_1.factors[0,0])))/2)
    
    d = 13
    r = 3
    B_1 = two_factor_test_tensor(d,r)
    assert np.isclose(B_1[0,0,0] , B_1.weights[0,0]*B_1.factors[0,0]**3
                              + B_1.weights[1,1]*B_1.factors[1,0]**3
                              +(B_1.weights[1,0]*B_1.factors[1,0]**2*B_1.factors[0,0]
                               +B_1.weights[0,1]*B_1.factors[0,0]**2*B_1.factors[1,0]))
                                                                                       
    assert np.isclose(B_1[1,0,0] ,B_1[0,1,0] )
    assert np.isclose(B_1[10,0,0] ,B_1[0,10,0] )                                                                                   
    assert np.isclose(B_1[10,0,0] , (3*B_1.weights[0,0]*B_1.factors[0,10]*B_1.factors[0,0]**2
                              + 3*B_1.weights[1,1]*B_1.factors[1,10]*B_1.factors[1,0]**2
                              +B_1.weights[1,0]*(B_1.factors[1,0]**2*B_1.factors[0,10] 
                                                 + 2*B_1.factors[1,10]*B_1.factors[1,0]*B_1.factors[0,0])
                               +B_1.weights[0,1]*(B_1.factors[0,0]**2*B_1.factors[1,10] 
                                                  + 2*B_1.factors[0,10]*B_1.factors[0,0]*B_1.factors[1,0]))/3 )
    assert np.isclose(B_1[10,11,0] , (3*B_1.weights[0,0]*B_1.factors[0,10]*B_1.factors[0,0]*B_1.factors[0,11]
                              + 3*B_1.weights[1,1]*B_1.factors[1,10]*B_1.factors[1,0]*B_1.factors[1,11]
                              +B_1.weights[1,0]*(B_1.factors[1,0]*B_1.factors[1,11]*B_1.factors[0,10] 
                                                 + B_1.factors[1,10]*B_1.factors[1,0]*B_1.factors[0,11]
                                                + B_1.factors[1,10]*B_1.factors[1,11]*B_1.factors[0,0])
                               +B_1.weights[0,1]*(B_1.factors[0,0]*B_1.factors[0,11]*B_1.factors[1,10] 
                                                 + B_1.factors[0,10]*B_1.factors[0,0]*B_1.factors[1,11]
                                                + B_1.factors[0,10]*B_1.factors[0,11]*B_1.factors[1,0]))/3 )

# %% [markdown]
# Test indexing for tensors of shape 
# $$
# T = \sum_{m} \lambda^{m,n,o} \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots v^n}_{l \text{ times}} \otimes \underbrace{t^o \otimes \dots t^o}_{l \text{ times}}
# $$
# with $u^m, v^m, t^o$ vectors. 

    # %%
    d = 3
    r = 3
    q = 1
    A_2 = DecompSymmetricTensor(rank=r, dim=d)
    A_2.weights = torch.zeros((2,2,2))
    A_2.weights[0,0,0] = 1
    A_2.weights[0,1,1] = 2
    A_2.factors =  torch.randn(size =(2,d))
    A_2.multiplicities = (r-2*q,q,q)
    assert np.isclose((A_2.factors[0,0]**3+2*A_2.factors[1,0]**2*A_2.factors[0,0]), A_2[0,0,0])
    assert np.isclose(A_2.factors[0,1]**3+2*A_2.factors[1,1]**2*A_2.factors[0,1], A_2[1,1,1])
    assert np.isclose(A_2[1,0,0], A_2[0,0,1])
    assert np.isclose(A_2.factors[0,1]**2*A_2.factors[0,0] 
                      +2/3.0*(A_2.factors[1,1]**2*A_2.factors[0,0]
                              +2*A_2.factors[1,0]*A_2.factors[1,1]*A_2.factors[0,1]) , A_2[1,1,0])
    
    d = 3
    r = 4
    A_2 = DecompSymmetricTensor(rank=r, dim=d)
    A_2.weights = torch.zeros((2,2,2))
    A_2.weights[0,0,0] = 1
    A_2.weights[0,1,1] = 2
    A_2.factors =  torch.randn(size =(2,d))
    A_2.multiplicities = (2,1,1)
    assert np.isclose((A_2.factors[0,0]**4+2*A_2.factors[1,0]**2*A_2.factors[0,0]**2), A_2[0,0,0,0])
    assert np.isclose(A_2.factors[0,1]**4+2*A_2.factors[1,1]**2*A_2.factors[0,1]**2, A_2[1,1,1,1])
    assert np.isclose(A_2[1,0,0,0], A_2[0,0,1,0])
    # ABAB BABA BAAB ABBA AABB BBAA
    assert np.isclose(A_2.factors[0,1]**2*A_2.factors[0,0]**2
                      +2/6.0*(A_2.factors[0,0]**2*A_2.factors[1,1]**2+A_2.factors[0,1]**2*A_2.factors[1,0]**2
                              +4*A_2.factors[1,0]*A_2.factors[1,1]*A_2.factors[0,1]*A_2.factors[0,0]) , A_2[1,1,0,0])

# %% [markdown]
# ## Shape, size, dtype

    # %%
    d = 2
    r = 3
    A = two_comp_test_tensor(d,r)
    assert A.shape == (2,2,2)
    assert A.size == 6
    assert A.dtype == torch.float64
    
    
    d = 20
    r = 3
    A = two_comp_test_tensor(d,r)
    assert A.shape == (20,20,20)
    assert A.size == 42
    assert A.dtype == torch.float64
    
    
    d = 20
    r = 10
    A = two_comp_test_tensor(d,r)
    assert A.shape == (20,)*10
    assert A.size == 20*2+2
    assert A.dtype == torch.float64

# %% [markdown]
# ### Casting to dense

# %%
if __name__ == "__main__": 
    # vector 
    A = DecompSymmetricTensor(rank = 1, dim =10)     
    weights = [0,1]
    factors =  torch.randn(size =(2,10))
    A.weights = weights
    A.factors = factors 
    A.multiplicities =  (1,)

    assert (A.todense()==A.factors[1,:]).all()
    
    #third order fully decomposed: AAA
    B = DecompSymmetricTensor(rank = 3, dim =3)   
    weights = [0.5,1, 0.01]
    factors =  torch.randn(size =(3,3))
    B.weights = weights
    B.factors = factors 
    B.multiplicities =  (3,)
    
    B_dense = 0.5*torch.tensordot(factors[0,:],torch.outer(factors[0,:],factors[0,:]),dims=0) \
                + torch.tensordot(factors[1,:],torch.outer(factors[1,:],factors[1,:]),dims=0) \
             +0.01*torch.tensordot(factors[2,:],torch.outer(factors[2,:],factors[2,:]),dims=0) 
    assert torch.allclose(B.todense(),B_dense)
    
    #third order partially decomposed: Two "factors" AAB
    C = DecompSymmetricTensor(rank = 3, dim =3)   
    weights = torch.Tensor([[0.5,0.5],[0,0.1]])
    factors =  torch.randn(size =(2,3))
    C.weights = weights
    C.factors = factors 
    C.multiplicities =  (2,1)
    
    C_dense = 0.5*torch.tensordot(factors[0,:],torch.outer(factors[0,:],factors[0,:]),dims=0) \
                + 0.5/binom(3,1)*(torch.tensordot(factors[0,:],torch.outer(factors[0,:],factors[1,:]),dims=0) \
                     +torch.tensordot(factors[0,:],torch.outer(factors[1,:],factors[0,:]),dims=0) \
                     +torch.tensordot(factors[1,:],torch.outer(factors[0,:],factors[0,:]),dims=0)) \
             +0.1*torch.tensordot(factors[1,:],torch.outer(factors[1,:],factors[1,:]),dims=0) 
    assert torch.allclose(C.todense(),C_dense)
    
    D = DecompSymmetricTensor(rank = 3, dim =3)   
    weights = torch.Tensor([[0.5,0.5],[0,0.1]])
    factors =  torch.randn(size =(2,3))
    D.weights = weights
    D.factors = factors 
    D.multiplicities =  (1,2)
    
    D_dense = 0.5*torch.tensordot(factors[0,:],torch.outer(factors[0,:],factors[0,:]),dims=0) \
                + 0.5/binom(3,1)*(torch.tensordot(factors[0,:],torch.outer(factors[1,:],factors[1,:]),dims=0) \
                     +torch.tensordot(factors[1,:],torch.outer(factors[1,:],factors[0,:]),dims=0) \
                     +torch.tensordot(factors[1,:],torch.outer(factors[0,:],factors[1,:]),dims=0)) \
             +0.1*torch.tensordot(factors[1,:],torch.outer(factors[1,:],factors[1,:]),dims=0) 
                           
    assert torch.allclose(D.todense(),D_dense)
    
    #third order partially decomposed : Three factors AABC   
    C = DecompSymmetricTensor(rank = 4, dim =3)   
    weights = torch.zeros(3,3,3)
    weights[0,1,2] = 1
    weights[0,0,0] = 2.0
    weights[1,1,2] = 0.1
    factors =  torch.randn(size =(3,3))
    C.weights = weights
    C.factors = factors 
    C.multiplicities =  (2,1,1)
    
    C_dense = ( torch.einsum('i,j,k,l -> ijkl', factors[0,:], factors[0,:],factors[1,:],factors[2,:])
               +2*torch.einsum('i,j,k,l -> ijkl', factors[0,:], factors[0,:],factors[0,:],factors[0,:])
               +0.1*torch.einsum('i,j,k,l -> ijkl', factors[1,:], factors[1,:],factors[1,:],factors[2,:]))
    sym_C_dense = utils.symmetrize(C_dense.numpy())
    assert np.allclose(C.todense().numpy(),sym_C_dense )
    
    #fourth order partially decomposed : Four factors ABCD   
    C = DecompSymmetricTensor(rank = 4, dim =3)   
    weights = torch.zeros(3,3,3,3)
    weights[0,0,1,2] = 1
    weights[0,0,0,0] = 2.0
    weights[1,1,1,2] = 0.1
    factors =  torch.randn(size =(3,3))
    C.weights = weights
    C.factors = factors 
    C.multiplicities =  (1,1,1,1)
    
    C_dense = ( torch.einsum('i,j,k,l -> ijkl', factors[0,:], factors[0,:],factors[1,:],factors[2,:])
               +2*torch.einsum('i,j,k,l -> ijkl', factors[0,:], factors[0,:],factors[0,:],factors[0,:])
               +0.1*torch.einsum('i,j,k,l -> ijkl', factors[1,:], factors[1,:],factors[1,:],factors[2,:]))
    sym_C_dense = utils.symmetrize(C_dense.numpy())
    assert np.allclose(C.todense().numpy(),sym_C_dense )


# %% [markdown]
# ### Copying

    # %%
    d = 10
    r = 4
    A = two_comp_test_tensor(d,r)
    B = A.copy()
    assert torch.allclose(B.todense(),A.todense())
    
    d = 10
    r = 4
    A = two_factor_test_tensor(d,r, q=2)
    B = A.copy()
    assert torch.allclose(B.todense(),A.todense())

# %% [markdown]
# ### Tensor comparison

    # %%
    for d in range(1,5): 
        for r in range(1,4): 
            A = two_comp_test_tensor(d,r)
            B = two_comp_test_tensor(d,r)
            C = A.copy()
            assert torch.allclose(B.todense(),A.todense())== np.allclose(A,B)
            assert torch.allclose(C.todense(),A.todense())== np.allclose(C,A)

# %% [markdown]
# ### Splitting up factors (changing multiplicities)
# #### Splitting off single factors

    # %%
    d = 3
    r =3
    A = two_comp_test_tensor(d,r)
    B = A.copy()
    B.split_factors(0)
    assert torch.allclose(A.todense(), B.todense(), atol = 1e-5)
    assert B.multiplicities == (2,1)
    
    d = 3
    r = 3
    A = two_factor_test_tensor(d,r,q=1)
    B = A.copy()
    B.split_factors(0)
    assert torch.allclose(A.todense(), B.todense(), atol = 1e-5)
    assert B.multiplicities == (1,1,1)
    
    d = 3
    r = 5
    A = three_factor_test_tensor(d,r,q=2)
    B = A.copy()
    B.split_factors(1)
    assert torch.allclose(A.todense(), B.todense(), atol = 1e-4)
    assert B.multiplicities == (1,1,1,2)

# %% [markdown]
# #### Matching up multiplicities between decomposed Tensors

    # %%
    d = 3
    r =3
    A = two_comp_test_tensor(d,r)
    B = A.copy()
    B.match_multiplicities((2,1))
    assert torch.allclose(A.todense(), B.todense())
    assert B.multiplicities == (2,1)
    assert A.find_common_multiplicities(B) == (2,1)
    
    d = 3
    r = 4
    A = two_factor_test_tensor(d,r,q=2)
    B = A.copy()
    B.match_multiplicities((2,1,1))
    assert torch.allclose(A.todense(), B.todense(),atol = 1e-5)
    assert B.multiplicities == (2,1,1)
    assert A.find_common_multiplicities(B) == (2,1,1)
    
    d = 3
    r = 5
    A = three_factor_test_tensor(d,r,q=2)
    B = A.copy()
    B.match_multiplicities((1,1,1,2))
    assert torch.allclose(A.todense(), B.todense(), atol = 1e-4)
    assert B.multiplicities == (1,1,1,2)
    assert A.find_common_multiplicities(B) == (2,1,1,1)

# %% [markdown]
# ## Addition

# %% [markdown]
# First, pure decomposed tensors.

    # %%
    d = 5
    r = 3
    A_1 = two_comp_test_tensor(d,r)
    B_1 = two_comp_test_tensor(d,r)
    
    C_1 = np.add(A_1,B_1)
    assert all(np.isclose(C_1[index], A_1[index]+B_1[index]) for index in  C_1.indep_iter_repindex())
    
    d = 10
    r = 5
    A_2 = two_comp_test_tensor(d,r)
    B_2 = two_comp_test_tensor(d,r)
    C_2 = np.add(A_2,B_2)

    assert torch.allclose(C_2.todense(), A_2.todense()+B_2.todense(), atol = 1e-5)


# %% [markdown]
# second, higher order decomposed tensors

    # %%
    d = 5
    r = 3
    
    A_1 = two_factor_test_tensor(d,r, q = 1)
    B_1 = two_factor_test_tensor(d,r, q = 1)
    C_1 = A_1+B_1
    assert all(np.isclose(C_1[index], A_1[index]+B_1[index]) for index in  C_1.indep_iter_repindex())
    
    d = 5
    r = 4
    
    A_2 = two_factor_test_tensor(d,r, q = 1)
    B_2 = two_factor_test_tensor(d,r, q = 1)
    C_2 = A_2+B_2
    assert all(np.isclose(C_2[index], A_2[index]+B_2[index]) for index in  C_2.indep_iter_repindex())
    
    
    A_3 = two_factor_test_tensor(d,r, q = 2)
    B_3 = two_factor_test_tensor(d,r, q = 2)
    C_3 = A_3+B_3
    assert all(np.isclose(C_3[index], A_3[index]+B_3[index]) for index in  C_3.indep_iter_repindex())

# %% [markdown]
# third, decomposed tensors with nonmatching multiplicites

    # %%
    d = 5
    r = 3
    
    A_1 = two_comp_test_tensor(d,r)
    B_1 = two_factor_test_tensor(d,r, q = 1)
    C_1 = A_1+B_1
    assert torch.allclose(C_1.todense(),A_1.todense()+B_1.todense(), atol = 1e-5)
    d = 5
    r = 4
    
    A_2 = two_comp_test_tensor(d,r)
    B_2 = three_factor_test_tensor(d,r, q = 1)
    C_2 = A_2+B_2
    assert torch.allclose(C_2.todense(),A_2.todense()+B_2.todense(), atol = 1e-5)


# %% [markdown]
# ### outer product

    # %%
    #rank 1 & rank 1, fully decomposed
    A = DecompSymmetricTensor(rank = 1, dim =10)     
    A_weights = torch.Tensor([1,0])
    A_factors =  torch.randn(size =(2,10))
    A.weights = A_weights
    A.factors = A_factors 
    A.multiplicities =  (1,)
    
    B = DecompSymmetricTensor(rank = 1, dim =10)     
    B_weights = torch.Tensor([0,1])
    B_factors =  torch.randn(size =(2,10))
    B.weights = B_weights
    B.factors = B_factors 
    B.multiplicities =  (1,)
    
    C = np.outer(A,B)
    C_dense = torch.outer(A_factors[0,:],B_factors[1,:])
    #compare to symmetrized tensor
    assert torch.allclose( C.todense(), (C_dense+ C_dense.T)/2.0)
    
    # rank 2 & rank 2, fully decomposed
    A = DecompSymmetricTensor(rank = 2, dim =10)     
    A_weights = torch.Tensor([1,0])
    A_factors =  torch.randn(size =(2,10))
    A.weights = A_weights
    A.factors = A_factors 
    A.multiplicities =  (2,)
    
    B = DecompSymmetricTensor(rank = 2, dim =10)     
    B_weights = torch.Tensor([0,1])
    B_factors =  torch.randn(size =(2,10))
    B.weights = B_weights
    B.factors = B_factors 
    B.multiplicities =  (2,)
    
    C = np.outer(A,B)
    assert torch.isclose(C[0,0,0,0],A[0,0]*B[0,0])
    assert  torch.isclose(C[1,0,0,0],(A[1,0]*B[0,0]+A[0,0]*B[1,0])/2.0)
    assert  torch.isclose(C[1,1,0,0] , (A[1,1]*B[0,0]+A[0,0]*B[1,1]+4*A[1,0]*B[1,0])/6.0)
    assert  torch.isclose(C[1,2,3,3] , (A[1,2]*B[3,3]+2*A[1,3]*B[2,3] \
                                       +2*A[2,3]*B[1,3]+A[3,3]*B[1,2])/6.0)
    assert  torch.isclose(C[1,2,3,4] , (A[1,2]*B[3,4]+A[1,3]*B[2,4]+A[1,4]*B[2,3] \
                                       +A[2,3]*B[1,4]+A[2,4]*B[1,3]+A[3,4]*B[1,2]
                                       )/6.0)
    
    # rank 2 bipartite & rank 2 fully decomposed
    A = two_factor_test_tensor(3,2, q = 1)
    B = two_comp_test_tensor(3,2)
    C = np.outer(A,B)
    
    C_dense = np.tensordot(A.todense(), B.todense(), axes=0)
    sym_C_dense = utils.symmetrize(C_dense)
    assert np.allclose(C.todense().numpy(), sym_C_dense, atol =1e-5)

    # rank 3 tripartite & rank 2 fully decomposed
    A = three_factor_test_tensor(2,3, q = 1)
    B = two_comp_test_tensor(2,2)
    C = np.outer(A,B)
    C_dense = np.tensordot(A.todense(), B.todense(), axes=0)
    sym_C_dense = utils.symmetrize(C_dense)
    assert np.allclose(C.todense().numpy(), sym_C_dense, atol =1e-5)

    # rank 2 bipartite & rank 2 bipartite
    A1 = two_factor_test_tensor(2,2, q = 1)
    B1 = two_factor_test_tensor(2,2, q = 1)
    C1 = np.outer(A,B)
    C1_dense = np.tensordot(A.todense(), B.todense(), axes=0)
    sym_C1_dense = utils.symmetrize(C1_dense)
    assert np.allclose(C1.todense().numpy(), sym_C1_dense, atol =1e-5)


# %% [markdown]
# ## Tensordot
#

    # %%
    #fully decomposed tensors
    d = 2
    for r in range(2,4): 
        tensor_1 = two_comp_test_tensor(d,r)
        for r_1 in range(2,4):
            tensor_2 = two_comp_test_tensor(d,r_1)
            test_tensor_13 = np.tensordot(tensor_1, tensor_2, axes=0)
            assert np.allclose(test_tensor_13, np.multiply.outer(tensor_1,tensor_2))

            #Contract over first and last indices:
            test_tensor_14 =  np.tensordot(tensor_1, tensor_2, axes=1)
            dense_tensor_14 = utils.symmetrize(np.tensordot(
                tensor_1.todense(), tensor_2.todense(), axes=1 ))
            assert np.allclose(test_tensor_14.todense(), dense_tensor_14)

            test_tensor_141 =  np.tensordot(tensor_1, tensor_2, axes = 2)
            if tensor_1.rank+tensor_2.rank > 5:
                dense_141 = torch.tensordot(tensor_1.todense(), tensor_2.todense(), dims=2).numpy()
                sym_dense_141 = utils.symmetrize(dense_141)
                assert np.allclose(test_tensor_141.todense().numpy(), sym_dense_141)
                if tensor_1.rank==3 and tensor_2.rank == 3:
                    test_tensor_142 =  np.tensordot(tensor_1, tensor_2, axes = 3)
                    assert torch.allclose(test_tensor_142, 
                                      torch.tensordot(tensor_1.todense(), tensor_2.todense(), dims=3))
                test_tensor_141 =  np.tensordot(tensor_1, tensor_2, axes = 2)
            elif tensor_1.rank+tensor_2.rank == 4:
                assert torch.allclose(test_tensor_141, 
                                      torch.tensordot(tensor_1.todense(), tensor_2.todense(), dims=2))

    # %%
    #partially decomposed tensors
    for r in range(2,4): 
        tensor_1 = two_factor_test_tensor(d,r)
        for r_1 in range(2,4):
            tensor_2 = two_comp_test_tensor(d,r_1)
            #Contract over first and last indices:
            test_tensor_14 =  np.tensordot(tensor_1, tensor_2, axes=1)
            dense_tensor_14 = utils.symmetrize(np.tensordot(
                tensor_1.todense(), tensor_2.todense(), axes=1 ))
            assert np.allclose(test_tensor_14.todense(), dense_tensor_14, atol =1e-5)


# %% [markdown]
# ### Contraction with matrix along all indices

# %%
if __name__ == "__main__":
    d = 3
    r = 2 
    A = two_comp_test_tensor(d,r)
    W = torch.randn(size=(3,3))
    B = symalg.contract_all_indices_with_matrix(A,W).todense().numpy()
    assert np.isclose(B, symmetrize(torch.einsum('ab, ai,bj -> ij', A.todense(), W,W).numpy())).all()
    r = 3
    A = two_comp_test_tensor(d,r)
    W = torch.randn(size=(3,3))
    assert np.isclose(symalg.contract_all_indices_with_matrix(A,W).todense(), symmetrize(torch.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W).numpy())).all()
    
    r = 2
    A = two_factor_test_tensor(d,r, q = 1)
    W = torch.randn(size=(3,3))
    B = symalg.contract_all_indices_with_matrix(A,W).todense().numpy()
    assert np.isclose(B, symmetrize(torch.einsum('ab, ai,bj -> ij', A.todense(), W,W).numpy())).all()
    r = 3
    A = two_factor_test_tensor(d,r, q = 1)
    W = torch.randn(size=(3,3))
    assert np.isclose(symalg.contract_all_indices_with_matrix(A,W).todense(), symmetrize(torch.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W).numpy())).all()


# %% [markdown]
# ### Multinomial 

    # %%
    d = 3
    r =3
    A = two_comp_test_tensor(d,r)
    for i in range(d): 
        x = torch.zeros(d)
        x[i] =1
        assert torch.isclose(A[(i,)*r],symalg.contract_all_indices_with_matrix(A,x))
    
    d = 3
    r =3
    A = two_factor_test_tensor(d,r)
    for i in range(d): 
        x = torch.zeros(d)
        x[i] =1
        assert torch.isclose(A[(i,)*r], A.multinomial(x))
        
    d = 3
    r = 5
    A = three_factor_test_tensor(d,r)
    for i in range(d): 
        x = torch.zeros(d)
        x[i] = 1
        assert torch.isclose(A[(i,)*r], A.multinomial(x))


# %%
## Test split factors

    # %%
    d = 3
    r =3
    A = two_comp_test_tensor(d,r)
    B = A.copy()
    B.split_factors(0)
    assert torch.allclose(A.todense(), B.todense())
    assert B.multiplicities == (2,1)
    
    d = 3
    r = 3
    A = two_factor_test_tensor(d,r,q=1)
    B = A.copy()
    B.split_factors(0)
    assert torch.allclose(A.todense(), B.todense())
    assert B.multiplicities == (1,1,1)
    
    d = 3
    r = 5
    A = three_factor_test_tensor(d,r,q=2)
    B = A.copy()
    B.split_factors(1)
    assert torch.allclose(A.todense(), B.todense(), rtol=1e-4)
    assert B.multiplicities == (1,1,1,2)

# %%
