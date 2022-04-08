# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-jupytext.kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: Python (statGLOW)
#     language: python
#     name: statglow
# ---

# %% [markdown]
# # Algorithmic design of tensor contractions
#
# Tensor contractions easily become the computational bottleneck in these theoretical studies. The faster we can make them, the bigger the systems we can consider.
#
# This notebook tests and compares different implementation for these operations. It serves to support the design choices made in [symmetric_tensor.py](./symmetric_tensor.py).

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-cell"]
if __name__ == "__main__":
    exenv = "script"
else:
    exenv = "module"

    # %% tags=["remove-cell"]
    exenv = "notebook"

# %% tags=["remove-cell"]
from typing import Union, List
import pytest

import itertools
import math  # For operations on plain Python objects, math can be 10x faster than NumPy
from numpy.random import RandomState  # The legacy RandomState is preferred for testing (see https://numpy.org/neps/nep-0019-rng-policy.html#supporting-unit-tests)
from statGLOW.smttask_ml.rng import get_seedsequence

import time
import timeit
from mackelab_toolbox.utils import TimeThis

import numpy as np

from tqdm.auto import tqdm, trange
from tqdm.contrib.logging import logging_redirect_tqdm
import ipywidgets as widgets

import statGLOW
from collections import Counter

# %% tags=["remove-cell"]
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# %% tags=["remove-cell"]
import pandas as pd
import holoviews as hv
if exenv == "notebook":
    hv.extension('bokeh')
elif exenv == "script":
    hv.extension('matplotlib')


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
#   $$B_{j_1,\dotsc,j_n} = \sum_{i_1,\dotsc,i_n} A_{i_1,\dotsc,i_n} W_{i_1,j_1} \dotsb W_{i_n,j_n} \hphantom{b_{i_k+1} \dotsb b_{i_n}} $$
# - **Rank-lowering contractions:**
#   $$B_{j_1,\dotsc,j_k} = \sum_{i_1,\dotsc,i_n} A_{i_1,\dotsc,i_n} W_{i_1,j_1} \dotsb W_{i_k,j_k} b_{i_k+1} \dotsb b_{i_n}$$

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
# `index_perm_prod_sum`
#
# TODO: explanation

# %%
def _index_perm_prod_sum(W, idx_fixed, idx_permute):
    """
    For index_fixed = (j_1, ... j_r)
    \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
    where σ(idx_permute) are all unique permutations.
    """
    idx_repeats = _get_perm_class(idx_permute) # number of repeats of indices
    permutations_of_identical_idx = np.prod([math.factorial(r) for r in idx_repeats])
    matr = np.array([[W[i,j] for i,j in zip(σidx,idx_fixed)]
                      for σidx in itertools.permutations(idx_permute)])
    return np.sum( np.prod(matr, axis =1)) /permutations_of_identical_idx


# %% [markdown]
# `contract_all_indices`
#
# TODO: explanation

# %%
def contract_all_indices(self,W):
    """
    compute the contraction over all indices with a non-symmetric matrix, e.g.

    C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}

    if current tensor has rank 3.
    """

    C = SymmetricTensor(rank = self.rank, dim = self.dim)

    for σcls in self.perm_classes:
        C[σcls] = [ np.sum([_index_perm_prod_sum(W, idx_fixed, idx_permute)*self[idx_permute]
                      for idx_permute in self.index_class_iter()]) for idx_fixed in self.index_class_iter(class_label= σcls) ]

    return C


# %% [markdown]
# `contract_tensor_list`
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

# %%
# Exhaustive values
n_calls = 10000
n_repeats = 16
n_seeds = 4      # For each (N, dist) combination, repeat this many times with different array values

# %% tags=["skip-execution", "remove-cell"]
# Quick run
n_calls = 4000
n_repeats = 3
n_seeds = 1

# %% tags=["skip-execution", "remove-cell"]
# Debugging values
n_calls = 100
n_repeats = 1
n_seeds = 1

# %%
Nlst = [100, 10000, 1000000, 100000000]
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
def get_idx(idx_desc: str, A: Union[np.ndarray, int], Mfrac: float=1, rng_key: Union[Tuple[int], int]=()):
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
     rng_key: An RNG key (int tuple) as expected by `smttask_ml.rng.get_seedsequence`.
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
    if isinstance(rng_key, int):
        rng_key = (rng_key,)
    rs = RandomState(get_seedsequence((INDEX_RNGKEY, *rng_key)).generate_state(1))
    
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
        # TODO: Used `rank`, `N` and the index generation functions from symmetric_tensor.py
        raise NotImplementedError
    else:
        raise ValueError(f"Unrecognized index descriptor '{idx_desc}'.")


# %% tags=["remove-cell", "skip-execution"]
# Unit testing for get_idx()
if exenv == "notebook":
    A = np.array([8, 2, 0, 1])
    B = np.arange(50)

    assert get_idx(":", A) == get_idx("None", A) == slice(None)
    assert list(get_idx("full", A)) == [0, 1, 2, 3]

    for idx_desc in ["random replace", "random no replace", "sequential replace", "sequential no replace"]:
        I1 = get_idx(idx_desc, B, 0.7, rng_key=1)
        I2 = get_idx(idx_desc, B, 0.7, rng_key=1)
        I3 = get_idx(idx_desc, B, 0.7, rng_key=2)
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
def get_test_array(N_idx, dist_idx, rng_key):
    """
    :param:i: Use this to generate different arrays with the same N and dist.
    :param:N_idx: Index (into `Nlst`) of the desired N.
    :param:dist: Index (into `dists`) of the desired distribution.
    """
    N = Nlst[N_idx]
    dist_info = dists[dist_idx]
    rs = RandomState(get_seedsequence((ARRAY_RNGKEY, N_idx, dist_idx, rng_key)).generate_state(1))
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

# %% tags=["hide-input"]
stmt = "np.dot(A1[i1], A2[i2])"
indexing = ["None", "full", "random replace"]

if exenv == "notebook":
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

                    if exenv == "notebook":
                        status_label.value = f"Indexing: {idx_desc}; M/N: {Mfrac}; Array size: {size}; calls per repeat: {number}"

                    res = timeit_repeat(
                        stmt, number=number,
                        A1 = get_test_array(Ni, d, rng_key=2*i),
                        A2 = get_test_array(Ni, d, rng_key=2*i+1),
                        i1 = get_idx(idx_desc, N, Mfrac=Mfrac, rng_key=2*i),
                        i2 = get_idx(idx_desc, N, Mfrac=Mfrac, rng_key=2*i+1),
                    )

                    timings[(stmt, idx_desc, Mfrac, N, dist_str(d), i)] = min(res)

# %% tags=["hide-input"]
index = pd.MultiIndex.from_tuples(timings.keys(), names=["operation", "indexing", "M/N", "N", "dist", "seed"])
df = pd.DataFrame({"time (ms)": timings}, index=index, columns=["time (ms)"])
df = df.unstack("seed").unstack("dist")
df

# %% tags=["remove-input"]
from mackelab_toolbox.utils import GitSHA    # Keep this cell below all others
GitSHA()                                     # (Eventually this will be replaced with `statGLOW.footer`)
