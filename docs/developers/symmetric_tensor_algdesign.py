# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Algorithmic design of tensor contractions
#
# ::::{margin}
# :::{note}
# The `%%cython --annotate` line adds the `.cython` class to this pages’s `<body>`, which is why everything is in a monospace font.  
# If someone knows how to fix this, that would be nice…
# :::
# ::::
#
# Tensor contractions easily become the computational bottleneck in these theoretical studies. The faster we can make them, the bigger the systems we can consider.
#
# This notebook tests and compares different implementation for these operations. It serves to support various design choices.
#
# :::{Warning}
# This file is badly out of date. Many of the profiled functions have now been integrated and further updated.
# :::

# %% [markdown]
# :::{admonition} TODO
# :class: important
# Split into sub-files: at present, if any code cell is changed, every test is re-run – some of them are quite long.
# :::

# %% tags=["remove-cell"]
from __future__ import annotations

# %%
from symtensor import utils

# %% tags=[]
from typing import Union, List
import pytest

import itertools
import math  # For operations on plain Python objects, math can be 10x faster than NumPy
from numpy.random import RandomState  # The legacy RandomState is preferred for testing (see https://numpy.org/neps/nep-0019-rng-policy.html#supporting-unit-tests)
from symtensor.utils import _get_permclass, get_permclass_size
from symtensor import symalg

import time
import timeit
from mackelab_toolbox.utils import TimeThis

import numpy as np

from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import ipywidgets as widgets

from collections import Counter

# %% tags=["remove-cell"]
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %% tags=["remove-cell"]
import pandas as pd
import holoviews as hv
hv.extension('bokeh')
#hv.extension('matplotlib')

# %% [markdown]
# ## Profiled operations
#
# Definitions:
# - $A$ is a symmetric tensor of rank $n$ and dimension $d$.
# - $W$ is a *non*-symmetric tensor of rank $n$ and dimension $d$.
# - $b$ is a vector of dimension $b$
#
# Contractions between two *symmetric* tensors are generally not too onerous, and since we symmetrize after applying one layer, one of two tensors in an operation is always symmetric. Thus the bottleneck operations are those contracting $A$ with $W$, and they come in 2 types: 
#
# - **Rank-preserving contractions:**
#   
#   $$B_{j_1,\dotsc,j_r} = \sum_{i_1,\dotsc,i_r} A_{i_1,\dotsc,i_r} W_{i_1,j_1} \dotsb W_{i_r,j_r} \hphantom{b_{i_k+1} \dotsb b_{i_r}} $$
#
# - **Rank-lowering contractions:**
#
#   $$B_{j_1,\dotsc,j_k} = \sum_{i_1,\dotsc,i_r} A_{i_1,\dotsc,i_r} W_{i_1,j_1} \dotsb W_{i_k,j_k} b_{i_k+1} \dotsb b_{i_r}$$
#   
#   Note that this second operation could be written as two operations: 
#
#   - **Lower the rank:**
#
#      $$C_{j_1,\dotsc,j_k} = \sum_{i_{k+1},\dotsc,i_r} A_{i_1,\dotsc,i_r} b_{i_k+1} \dotsb b_{i_r}$$
#
#   - **Rank-preserving contraction:**
#
#     $$B_{j_1,\dotsc,j_k} = \sum_{i_1,\dotsc,i_k} C_{i_1,\dotsc,i_k} W_{i_1,j_1} \dotsb W_{i_k,j_k} $$
#  
# Here again, the second operation is the expesive one, since only the second operation involves non-symmetric tensors (vectors are trivially symmetric). 

# %% [markdown] tags=["remove-cell"]
# ## Notes on profiling
#
# For precise measurement of tight operations, in particular to compare different implementations, the functions from the `timeit` module. (The equivalent `%timeit` and `%%timeit` may also be useful in interactive sessions.) To get accurate timings of very short code segments, calls are made a large *number* of times (100-1000000), and then that experiment is *repeated* a few times (1-10).
#
# For tracking down *where* bottlenecks are located (and therefore what to profile with `timeit`), the `TimeThis` context manager from *mackelab_toolbox.utils* may be useful.
#
# The magics `%timeit` and `%%timeit` auto-determine the number of calls and repeats, then print the mean and std. dev. of the execution time. This is extremely convenient for interactive sessions where we don't know the execution time before hand, but has a few disadvantages:
# - Some computation time is wasted on runs with insufficient number of calls.
# - The result is not accessible as a variable, so we can't create a summary table automatically.
# - The `timeit` documentation recommends taking the minimum rather than the mean + std dev of the execution times.
#
# So once we've determined good numbers for calls and repeats (possibly from `%timeit`), it's probably best for the profiling script to using functions from the `timeit` module.

# %% [markdown]
# ## Implementations

# %% [markdown]
# ### Current implementation with numpy and itertools
#
# Let $\alpha_{\text{list}}, \beta_{\text{list}}$ be lists containing all the unique indices of the symmetric tensors $A,B$. 
# For example, for $A$ a tensor of rank $r=4$ with dimension 10: 
#
# $$ \alpha_{\text{list}} = \begin{pmatrix}
# 0,0,0,0 \\
# 0,0,0,1 \\
# \vdots \\
# 9,9,9,9 \end{pmatrix} \,. $$
#
# Let $\alpha, \beta$ be elements of $\alpha_{\text{list}}, \beta_{\text{list}}$.
# Then to compute the entry $B_{\beta}$, we must do: 
#
# $$
# B_{\beta} = \sum_{\alpha} A_{\alpha} \sum_{(i_1, ... i_r) \in \, σ(\alpha)} W_{i_1,\beta_1} ... W_{i_r, \beta_r} 
# $$
#
# where $σ(\alpha)$ are all unique permutations of $\alpha$. (For example, if $\alpha$ is $(1,1,2,3)$, then $(i_1,i_2,i_3,i_4)$ and $(i_2,i_1,i_3,i_4)$ are not unique permutations.)
#
#       

# %% [markdown]
# So at the moment we do: 
#
# ```python
# for beta in beta_list: 
#   for alpha in alpha_list:   
#     B[beta] += A[alpha]*index_perm_prod_sum(W, beta, alpha)
# ```

# %% [markdown]
# ### `index_perm_prod_sum`
#
# This function computes
#
# $$\sum_{(i_1, ... i_r) \in \, σ(\alpha)} W_{i_1,\beta_1} ... W_{i_r, \beta_r} $$
#
# for $\beta = (j_1, ... j_r) =$ `idx_fixed` and $\alpha =$ `idx_permute`.

# %%
def index_perm_prod_sum_no_cython(W, idx_fixed,idx_permute):
    """
    For index_fixed = (j_1, ... j_r)
    \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
    where σ(idx_permute) are all unique permutations.
    
    (for example, if idx_permute is (1,1,2), then, i_1,i_2,i_3 and i_2,i_1,i_3 are not unique permutations.)
    """
    idx_repeats = _get_permclass(idx_permute) # number of repeats of indices
    permutations_of_identical_idx = math.prod([math.factorial(r) for r in idx_repeats])
    matr = np.array([[W[i,j] for i,j in zip(permuted_idx,idx_fixed)]
                                      for permuted_idx in itertools.permutations(idx_permute)])
    result = np.sum( np.prod(matr, axis =1)) /permutations_of_identical_idx
    return result 


# %%
# %load_ext cython

# %% magic_args="--annotate" language="cython"
# import numpy as np
# cimport numpy as np
# np.import_array()
# import itertools
# import math
# from symtensor.utils import _get_permclass
# def index_perm_prod_sum_old(np.ndarray W, tuple idx_fixed, tuple idx_permute):
#     """
#     For index_fixed = (j_1, ... j_r)
#     \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
#     where σ(idx_permute) are all unique permutations.
#     
#     (for example, if idx_permute is (1,1,2), then, i_1,i_2,i_3 and i_2,i_1,i_3 are not unique permutations.)
#     """
#     cdef tuple idx_repeats = _get_permclass(idx_permute) # number of repeats of indices
#     cdef int permutations_of_identical_idx = math.prod([math.factorial(r) for r in idx_repeats])
#     cdef np.ndarray matr = np.array([[W[i,j] for i,j in zip(permuted_idx,idx_fixed)]
#                                       for permuted_idx in itertools.permutations(idx_permute)])
#     cdef float result = np.sum( np.prod(matr, axis =1)) /permutations_of_identical_idx
#     return result 
#
# def index_perm_prod_sum(np.ndarray W, tuple idx_fixed, tuple idx_permute):
#     """
#     For index_fixed = (j_1, ... j_r)
#     \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
#     where σ(idx_permute) are all unique permutations.
#     
#     (for example, if idx_permute is (1,1,2), then, i_1,i_2,i_3 and i_2,i_1,i_3 are not unique permutations.)
#     """
#     cdef tuple idx_repeats = _get_permclass(idx_permute) # number of repeats of indices
#     cdef int permutations_of_identical_idx = math.prod(math.factorial(r) for r in idx_repeats)
#     cdef float result = sum([math.prod(W[i,j] for i,j in zip(permuted_idx,idx_fixed))
#                                       for permuted_idx in itertools.permutations(idx_permute)]) /permutations_of_identical_idx
#     #cdef float result = np.sum(matr) /permutations_of_identical_idx
#     return result 
#
#

# %% [markdown]
# ### Some insights on `index_perm_prod_sum` : 
#
# I looked at a few of the functions used in`index_perm_prod_sum` and how to make them faster. 
# 1) math.prod is faster: 
# 2) pythons builtin sum is fastest - on lists or off lists
#
# With just these changes I can decrease the time for`index_perm_prod_sum`.

# %%
#compare products
import math
idx_repeats = (1,3)
# %timeit np.prod([math.factorial(r) for r in idx_repeats])
# %timeit np.prod(np.fromiter((math.factorial(r) for r in idx_repeats), dtype = int))
# %timeit math.prod(math.factorial(r) for r in idx_repeats)


# %%
#compare sums
to_sum = np.random.randn(100)
# %timeit sum(np.random.rand() for i in range(10000))
# %timeit np.sum(np.fromiter((np.random.rand() for i in range(10000)), dtype= float))
# %timeit sum([np.random.rand() for i in range(10000)])

# %%
#compare resulting functions
W = np.random.randn(100,100)
idx_fixed= (1,2,40,5)
idx_permute = (10,0,1,9)
#test whether new function still produces the same result
for i in range(10): 
    W = np.random.randn(100,100)
    assert  index_perm_prod_sum_old(W, idx_fixed, idx_permute) ==  index_perm_prod_sum(W, idx_fixed, idx_permute)
#compare times
# %timeit index_perm_prod_sum_old(W, idx_fixed, idx_permute)
# %timeit index_perm_prod_sum(W, idx_fixed, idx_permute)

# %% [markdown]
# ## How fast is the iteration over the tensor? 
#
# Trying to isolate even further which part of the computation is slow: Iterating over the tensors is not expensive. Retrieving the values is also not expensive, since we iterate over them in a sequential manner, as we do in our final contraction method. At, least, the time needed for the retrieval of values is far below  the time needed for the computation of the contraction. 

# %% [markdown]
# ::::{margin}
# :::{admonition} TODO
# :class: important
# Report with DataFrame as with [](#micro-tests).
# :::
# ::::

# %% tags=[]
from symtensor import PermClsSymmetricTensor as SymmetricTensor

def iterate_over_tensor_retrieve_value(x):
    a=0
    for idx_permute in x.indep_iter_repindex(): 
        for idx_permute1 in x.indep_iter_repindex(): 
            y = x[idx_permute1]
            a +=1
    return a

def iterate_over_tensor(x):
    a=0
    for idx_permute in x.indep_iter_repindex(): 
        for idx_permute1 in x.indep_iter_repindex(): 
            a +=1
    return a


for dim in [1,2,4,6,8]:#,10]: 
    x = SymmetricTensor(rank = 4, dim= dim)
    x['iiii'] = 1
    #print(iterate_over_tensor(x))
    # %timeit iterate_over_tensor(x)
    # %timeit iterate_over_tensor_retrieve_value(x)


# %% [markdown]
# ### `contract_all_indices_with_matrix`
#
# Compute the rank-preserving contraction 
# $$B_{j_1,\dotsc,j_r} = \sum_{i_1,\dotsc,i_r} A_{i_1,\dotsc,i_r} W_{i_1,j_1} \dotsb W_{i_r,j_r} \hphantom{b_{i_k+1} \dotsb b_{i_r}} $$
# as described above. 

# %%
def contract_all_indices_with_matrix_old( x, W):
    """
    compute the contraction over all indices with a non-symmetric matrix, e.g.

    C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}

    if current tensor has rank 3.
    """

    C = SymmetricTensor(rank = x.rank, dim = x.dim)

    for perm_cls in x.perm_classes:
        C[perm_cls] = [ np.sum([index_perm_prod_sum_old(W, idx_fixed, idx_permute)*x[idx_permute]
                      for idx_permute in x.indep_iter_repindex()]) for idx_fixed in x.indep_iter_repindex(class_label= perm_cls) ]
    return C

def contract_all_indices_with_matrix( x, W):
    """
    compute the contraction over all indices with a non-symmetric matrix, e.g.

    C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}

    if current tensor has rank 3.
    """

    C = SymmetricTensor(rank = x.rank, dim = x.dim)

    for perm_cls in x.perm_classes:
        C[perm_cls] = [ sum(index_perm_prod_sum(W, idx_fixed, idx_permute)*x[idx_permute]
                      for idx_permute in x.indep_iter_repindex())
                       for idx_fixed in x.permcls_indep_iter_repindex(perm_cls) ]
    return C


# %% [markdown]
# ## Tensor contraction as in Schatz paper 
#
# We implement the tensor contraction as indicated in the schatz paper for rank 3 and rank 4 tensors. (Compare to their fig. 3.1).  
#
#
#     

# %%
def contract_all_indices_with_matrix_schatz(x, W): 
    
    if x.rank == 3: 
        return contract_all_indices_with_matrix_schatz_3(x, W)
    elif x.rank == 4: 
        return contract_all_indices_with_matrix_schatz_4(x, W)

def contract_all_indices_with_matrix_schatz_3(x, W): 
    
    new_tensor = SymmetricTensor(rank =3, dim = x.dim)
    
    for i in range(0, x.dim): 
        y = SymmetricTensor(rank =1, dim = x.dim)
        y['i'] = W[i,:]
        t_1 = symalg.tensordot(x, y, axes = 1) 
        for j in range(0, i+1): 
            y = SymmetricTensor(rank =1, dim = x.dim)
            y['i'] = W[j,:]
            t_2 = symalg.tensordot(t_1, y, axes = 1) 
            for k in range(0, j+1):
                y = W[k,:]
                new_tensor[i,j,k] = np.dot(t_2['i'], y)
    return new_tensor

def contract_all_indices_with_matrix_schatz_4(x, W): 
    
    new_tensor = SymmetricTensor(rank =4, dim = x.dim)
    
    for i in range(0, x.dim): 
        y = SymmetricTensor(rank =1, dim = x.dim)
        y['i'] = W[:,i]
        t_1 = symalg.tensordot(x, y, axes = 1) 
        for j in range(0, i+1): 
            y = SymmetricTensor(rank =1, dim = x.dim)
            y['i'] = W[:,j]
            t_2 = symalg.tensordot(t_1, y, axes = 1) 
            for k in range(0, j+1): 
                y = SymmetricTensor(rank =1, dim = x.dim)
                y['i'] = W[:,k]
                t_3 = symalg.tensordot(t_2, y, axes = 1) 
                for l in range(0, k+1):
                    y = W[:,l]
                    new_tensor[i,j,k,l] = np.dot(t_3['i'], y)
    return new_tensor
                


# %% [markdown]
# Now we compare how far we got with the improvements. We use a rank $4$ tensor because we would like to go at least that high in rank. 

# %% [markdown]
# :::{margin}
# **Removed** (15.11.2022): `x.contract_all_indices_with_matrix`
# Probably should be replacet with `contract_tensor_list`.
# :::
#
# ::::{margin}
# :::{admonition} TODO
# :class: important
# Report with DataFrame as with [](#micro-tests).
# :::
# ::::

# %%

for dim in [2,4,6,8,9,10]:#,20,30]: 
    x = SymmetricTensor(rank =4, dim= dim)
    W = np.random.rand(dim,dim)
    x['iiii'] = np.random.rand( get_permclass_size((4,), dim) )
    x['ijkl'] = np.random.rand( get_permclass_size((1,1,1,1), dim) )
    x['ijjj'] = np.random.rand( get_permclass_size((3,1), dim) )
    x['iijj'] = np.random.rand( get_permclass_size((2,2), dim) )
    x['iijk'] = np.random.rand( get_permclass_size((2,1,1), dim) )
    print('dim = ',dim)
    
    #check that implementations still work
    assert np.isclose(contract_all_indices_with_matrix(x,W).todense(), np.einsum('ijkl, ia, jb, kc, ld -> abcd',x.todense(), W,W,W,W)).all()
    assert np.isclose(contract_all_indices_with_matrix_schatz(x,W).todense(), np.einsum('ijkl, ia, jb, kc, ld -> abcd',x.todense(), W,W,W,W)).all()
    #print('without cython or any other improvements:')
    # #%timeit x.contract_all_indices_with_matrix(W) #without cython or any other improvements
    if dim > 6:
        print("Skipping our method for dim > 6 (too slow).")
    else:
        print('fastest version of our method (using cython etc):')
        # %timeit contract_all_indices_with_matrix(x,W) #fastest contraction atm 
    print('schatz method, dirty implementation')
    # %timeit contract_all_indices_with_matrix_schatz(x,W)


# %% [markdown]
# ### `contract_tensor_list`
#
# TODO: explanation

# %%
def contract_tensor_list(self, tensor_list, n_times =1, rule = 'second_half'):
    """
    Do the following contraction:

    out_{i_1,i_2,..., i_(r-n_times), j_1, j_2, ...j_m, k_1, k_2, ... k_m, ...}
    = Symmetrize( \sum_{i_{r-n_times+1}, ..., i_r} outer( self_{i_1,i_2,.. i_r}, tensor_list[i_{r-n_times+1}]_{j_1,j_2,...j_m},

    Important: The tensors in tensor_list must be symmetric.
    This is essentially a way to do a contraction between a symmetric and quasi_symmetric tensor \chi. Let

    \chi_{i,j_1,j_2,...,j_m} = tensor_list[i]_{j_1,j_2,...j_m}

    Then even if \chi is not symmetric under exchange of the first indices with the rest, but the subtensors \chi_i,...
    for fixed i are, we can do a contraction along the first index.
    """
    if not n_times <= self.rank:
        raise ValueError(f"n_times is {n_times}, but cannot do more contractions than {self.rank} with tensor of rank {self.rank}")
    for list_entry in tensor_list:
        if not isinstance(list_entry, SymmetricTensor):
            raise  TypeError("tensor_list entries must be SymmetricTensors")
    if self.rank ==1 and n_times ==1:
        return sum((tensor_list[i]*self[i] for i in range(self.dim)),
                    start=SymmetricTensor(tensor_list[0].rank, tensor_list[0].dim))
    else:
        get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-n_times)
        if rule == 'second_half':
            first_half = int(np.ceil(self.dim/2.0))
            indices_for_contraction = range(first_half, self.dim)
            indices = itertools.product( indices_for_contraction, repeat = n_times)
        else:
            indices = itertools.product(range(self.dim), repeat = n_times)
        chi_rank = tensor_list[0].rank
        C = SymmetricTensor(dim = self.dim, rank = self.rank +(chi_rank-1)*n_times) #one dimension used for contraction
        if n_times < self.rank:
            for idx in indices:
                slice_idx = get_slice_index(idx, self.rank)
                C += self[slice_idx].outer_product([tensor_list[i] for i in idx])
        else:
            for idx in indices:
                slice_idx = get_slice_index(idx, self.rank)
                C += tensor_list[idx[0]].outer_product([ tensor_list[i] for i in idx[1:]])*self[slice_idx]
        return C


# %% [markdown]
# ## Micro tests

# %% tags=["remove-cell"]
ARRAY_RNGKEY = 0
INDEX_RNGKEY = 1

INDEX_DTYPE = np.uint32

# %% tags=[]
# Exhaustive values
n_calls = 1000
n_repeats = 4
n_seeds = 4      # For each (N, dist) combination, repeat this many times with different array values

# %% tags=["remove-cell", "skip-execution"]
# Quick run
n_calls = 1000
n_repeats = 3
n_seeds = 1

# %% tags=["skip-execution", "remove-cell"]
# Debugging values
n_calls = 100
n_repeats = 1
n_seeds = 1

# %%
#Nlst = [100, 10000, 1000000, 100000000]
Nlst = [100, 400, 1000, 4000, 10000]
Mfrac_lst = [0.1, 1, 3, 10]   #  Values for M/N
dists = [('normal', {'scale': 3}),
         ('normal', {'scale': 10}),
         ('uniform', {'high': 3}),
         ('uniform', {'high': 10})]


# %% tags=["hide-input"]
def extract_numbers(s) -> List[int]:
    """
    Find integers contained in a string, and return them in a list.
    For example, given
    
        "aaaa 33.1 ee2,8 ii2 2099"
    
    returns
    
        [33, 1, 2, 8, 2, 2099]
        
    Decimals are not currently supported (i.e. '.' and ',' have no special treatment)
    """
    substr = []
    num = ""
    for c in s:
        if c.isdigit():
            num += c
        else:
            if num:
                substr.append(int(num))
                num = ""
    if num:
        substr.append(int(num))


# %% tags=["hide-input"]
def get_idx(idx_desc: str, A: Union[np.ndarray, int], Mfrac: float=1, seed=None):
    """
    Convert a compact, human-readable index description to an
    object which can be used to slice the array `A`.
    
    Parameters
    ----------
    idx_desc: May be of one of the following forms:

       - "None", or ":"
         Returns a slice, so indexing will be equivalent to ``A[:]``.
       - "full"
         Return the indices [0,...,N]. Indexing will be functionally equivalent ``A[:]``,
         but possibly slower due to the use of advanced indexing.
         This option may provide a more honest bound for operations that must use advanced indexing.
         Equivalent to 'sequential no replace', Mfrac=1.
       - "random replace"
       - "random no replace"
       - "sequential replace"
       - "sequential no replace"
       - "symmetric, {r}" or "symmetric, rank {r}"
         where {r} is an integer.
         This returns indices as they would be laid out in a SymmetricTensor of rank r.
         
     A: Array to be indexed, or size of the array.
     Mfrac: The ratio M/N
     seed: An RNG seed to pass to .
        For convenience, a plain integer is also accepted.
        In both cases, the key is prefixed with `INDEX_RNGKEY`.
    """
    if isinstance(A, np.ndarray):
        if A.ndim != 1:
            raise ValueError(f"`A` should be an integer or a 1D array. Received array with shape {A.shape}.")
        N = len(A)
    elif isinstance(A, int):
        N = A
    else:
        raise ValueError(f"`A` should be an integer or a 1D array. Received: {A} (type: {type(A)})")
    M = max(round(Mfrac*N), 1)
    rs = RandomState(seed)
    
    if idx_desc in {"full", "sequential full"}:
        idx_desc = "sequential no replace"
        if Mfrac != 1:
            raise ValueError(f"The '{idx_desc}' description is short hand for "
                             "'sequential no replace' and `Mfrac=1`.")
    
    if idx_desc.lower() in {"none", ":"}:
        return slice(None)
    elif idx_desc.lower() == "random replace":
        return rs.randint(N, size=M, dtype=INDEX_DTYPE)
    elif idx_desc.lower() == "random no replace":
        return rs.choice(np.arange(N, dtype=INDEX_DTYPE), size=M, replace=False)
    elif idx_desc.lower() == "sequential replace":
        return np.sort(rs.randint(N, size=M, dtype=INDEX_DTYPE))
    elif idx_desc.lower() == "sequential no replace":
        if M == N:
            return np.arange(N, dtype=INDEX_DTYPE)
        else:
            return np.sort(rs.choice(np.arange(N, dtype=INDEX_DTYPE), size=M, replace=False))
    elif idx_desc.lower().startswith("symmetric"):
        numbers = extract_numbers(idx_desc)
        if len(numbers) == 0:
            raise ValueError("'symmetric' option requires to also specify rank")
        elif len(numbers) > 1:
            raise ValueError("'symmetric' option should contain exactly one number, specifying rank.\n"
                             f"Received '{idx_desc}'.")
        rank = numbers[0]
        # TODO: Use `rank`, `N` and the index generation functions from symtensor.py
        raise NotImplementedError
    else:
        raise ValueError(f"Unrecognized index descriptor '{idx_desc}'.")


# %% tags=["remove-cell", "skip-execution"]
# Unit testing for get_idx()
A = np.array([8, 2, 0, 1])
B = np.arange(50)

assert get_idx(":", A) == get_idx("None", A) == slice(None)
assert list(get_idx("full", A)) == [0, 1, 2, 3]

for idx_desc in ["random replace", "random no replace", "sequential replace", "sequential no replace"]:
    I1 = get_idx(idx_desc, B, 0.7, seed=1)
    I2 = get_idx(idx_desc, B, 0.7, seed=1)
    I3 = get_idx(idx_desc, B, 0.7, seed=2)
    assert list(I1) == list(I2) != list(I3)

I = get_idx("random replace", A, 0.1); assert len(I) == 1; assert np.all(I < 4)
I = get_idx("random replace", A, 0.5); assert len(I) == 2; assert np.all(I < 4)
I = get_idx("random replace", A, 1);   assert len(I) == 4; assert np.all(I < 4)
I = get_idx("random replace", A, 2);   assert len(I) == 8; assert np.all(I < 4)
I = get_idx("random no replace", A, 0.1); assert len(I) == 1; assert np.all(I < 4)
I = get_idx("random no replace", A, 0.5); assert len(I) == 2; assert np.all(I < 4)
I = get_idx("random no replace", A, 1)  ; sorted(I) == [0, 1, 2, 3]
with pytest.raises(ValueError):
    assert len(get_idx("random no replace", A, 2)) == 8

I = get_idx("sequential replace", A, 0.1); assert len(I) == 1; assert np.all(I < 4)
I = get_idx("sequential replace", A, 0.5); assert len(I) == 2; assert np.all(I < 4); assert list(I) == sorted(I)
I = get_idx("sequential replace", A, 1);   assert len(I) == 4; assert np.all(I < 4); assert list(I) == sorted(I)
I = get_idx("sequential replace", A, 2);   assert len(I) == 8; assert np.all(I < 4); assert list(I) == sorted(I)
I = get_idx("sequential no replace", A, 0.1); assert len(I) == 1; assert np.all(I < 4); assert list(I) == sorted(I)
I = get_idx("sequential no replace", A, 0.5); assert len(I) == 2; assert np.all(I < 4); assert list(I) == sorted(I)
I = get_idx("sequential no replace", A, 1)  ; list(I) == [0, 1, 2, 3]
with pytest.raises(ValueError):
    assert len(get_idx("sequential no replace", A, 2)) == 8


# %% tags=["remove-cell"]
def dist_str(d):
    """
    Given a dist index, return a string similar to the Python
    code which would create the corresponding distribution.
    """
    s = ",".join(f"{nm}={val}" for nm, val in dists[d][1].items())
    return f"{dists[d][0]}({s})"


# %% tags=["hide-input"]
def get_test_array(N_idx, dist_idx, seed):
    """
    :param:i: Use this to generate different arrays with the same N and dist.
    :param:N_idx: Index (into `Nlst`) of the desired N.
    :param:dist: Index (into `dists`) of the desired distribution.
    """
    N = Nlst[N_idx]
    dist_info = dists[dist_idx]
    rs = RandomState(seed)
    dist_method = getattr(rs, dist_info[0])
    return dist_method(**dist_info[1], size=N)


# %% tags=["hide-input"]
def timeit_repeat(stmt, number=None, repeat=None, **variables):
    """
    Wraps `timeit.repeat`:
    - Provides defaults for `globals`, `number` and `repeat`.
    - Converts results to time per call.
    - Converts results to milliseconds.
    
    `stmt` may include the following identifiers:
    - `np`
    
    Additional variables used in `stmt` can be defined with keyword arguments.
    """
    if number is None:
        number = globals()["n_calls"]
    if repeat is None:
        repeat = globals()["n_repeats"]
    namespace = {"np": np, **variables}
    res = timeit.repeat(stmt, globals=namespace,
                        number=number, repeat=repeat)
    res = [t/number*1000 for t in res]
    return res


# %% [markdown] tags=["remove-cell"]
#     # Use this block to test timing parameters
#
#     N = 0
#     d = 0
#
#     A1 = get_test_array(N, d, 0)
#     A2 = get_test_array(N, d, 1)
#
#     %timeit np.dot(A1, A2)

# %% tags=["remove-cell"]
timings = {}

# %% tags=["hide-input", "remove-output", "skip-execution"]
stmt = "np.dot(A1[i1], A2[i2])"
indexing = ["None", "full", "random replace"]

status_label = widgets.Label("Indexing: {idx_desc}; M/N: {Mfrac}; Array size: {size}; calls per repeat: {number}"
                             .format(idx_desc=None, Mfrac=None, size=None, number=None))
display(status_label)

for idx_desc in tqdm(indexing, desc="indexing"):
    _Mfrac_lst = [1] if idx_desc in {"None", "full"} else Mfrac_lst
    for Mfrac in tqdm(_Mfrac_lst, desc="M/N", leave=False):
        for Ni in trange(len(Nlst), desc="N idx", leave=False):          # Determines array size
            for d in trange(len(dists), desc="dist idx", leave=False):   # Determines distribution from which entries are drawn
                for i in trange(n_seeds, desc="data seed", leave=False): # Determines array entries (same size & dist)
                    N = Nlst[Ni]
                    
                    # Make the number of calls inversely proportional to array size
                    size = int(round(N*Mfrac))
                    number = n_calls * 100000 // size
                    number = min(number, 1000000)                  # Make no more than 1 million calls per repeat
                    number = max(number, 5)                        # Hard minimum of 5 calls per repeat
                    ndigits = -math.ceil(math.log10(number)) + 2   # Keep up to 2 significant digits
                    number = round(number, ndigits)

                    status_label.value = f"Indexing: {idx_desc}; M/N: {Mfrac}; Array size: {size}; calls per repeat: {number}"

                    res = timeit_repeat(
                        stmt, number=number,
                        A1 = get_test_array(Ni, d, seed=2*i),
                        A2 = get_test_array(Ni, d, seed=2*i+1),
                        i1 = get_idx(idx_desc, N, Mfrac=Mfrac, seed=2*i),
                        i2 = get_idx(idx_desc, N, Mfrac=Mfrac, seed=2*i+1),
                    )

                    timings[(stmt, idx_desc, Mfrac, N, dist_str(d), i)] = min(res)

# %% tags=["hide-input"]
index = pd.MultiIndex.from_tuples(timings.keys(), names=["operation", "indexing", "M/N", "N", "dist", "seed"])
df = pd.DataFrame({"time (ms)": timings}, index=index, columns=["time (ms)"])
df = df.unstack("seed").unstack("dist")
df

# %% tags=["remove-input"]
from mackelab_toolbox.utils import GitSHA    # Keep this cell below all others
GitSHA(show_hostname=True)
