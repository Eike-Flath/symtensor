# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Utility functions for Symmetric Tensors

# %%
from __future__ import annotations

# %%
import itertools
import math
import functools
from itertools import chain
import numpy as np

from typing import Optional, Union, Any, List, Tuple, Sequence, Iterable, Generator
from scityping.numpy import Array

# %% tags=["remove-cell", "active-py"]
if __name__ != "__main__":
    exenv = "module"
else:
    exenv = "script"

# %% tags=["remove-cell", "active-ipynb"]
# exenv = "jbook"

# %% tags=["skip-execution", "remove-cell", "active-ipynb"]
# exenv = "notebook"

# %%
if exenv in {"notebook", "jbook"}:
    from mackelab_toolbox.utils import timeit

    def get_timeit_kwargs(nops: int) -> dict:
        ":param:nops: Estimate of the number of operations."
        kwds = {"number": 100000, "repeat": 7}  # Defaults
        if nops > 1e12:
            kwds["number"] = 1; kwds["repeat"] = 1
        elif nops > 1e11:
            kwds["number"] = 1; kwds["repeat"] = 3
        elif nops > 1e10:
            kwds["number"] = 1
        elif nops > 1e4:
            kwds["number"] = 10**(10-math.ceil(math.log10(nops)))  # The 10 inside the power comes from `nops > 1e10` above
        return kwds

# %% [markdown]
# ## Types and metaprogramming
# (Could be in a more generic library)

# %%
import sys
from collections import OrderedDict, deque
from collections.abc import Collection as Collection_
from typing import Tuple


# %% [markdown]
# ### `common_superclass`

# %%
def common_superclass(*instances_or_types):
    """
    Return the greatest common superclass to given instances.
    Based on: https://stackoverflow.com/a/25787091
    """
    mros = (o.mro() if isinstance(o, type) else type(o).mro()
            for o in instances_or_types)
    mro = next(mros)
    common = set(mro).intersection(*mros)
    return next((x for x in mro if x in common), None)


# %% [markdown]
# ## Bypassing coercion to ndarray
#
# (Note: the `make_array_like` context manager is not specific to `SymmetricTensor`, and could be moved to a more generic 'utils' module)
#
# A lot of the provided NumPy functions use `asarray` or `asanyarray` to ensure their inputs are array-like (and not, say, a list). Unfortunately this also coerces inputs into NumPy arrays, which we absolutely want to avoid with `SymmetricTensor`. The problem as that these functions are used for two different purposes:
# - When it is required that arguments truly be NumPy arrays;
# - When it is required that arguments be array-like, and implement parts of the NumPy array API (so-called “duck arrays”).
# The current solution is to use the keyword `like` (see [NEP 35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html)) when a duck array will suffice; this has the added benefit of being explicit about which duck array type is expected (since different types implement different subsets of the array API).
#
# This however doesn't address the use of `asarray` within NumPy functions, where it is mostly called without the `like` keyword. (Which makes sense – to use the keyword, the function would have to know the duck type desired by the user.) As a workaround, we provide below the `make_array_like` context manager: within the context, and only for the specified modules, the functions `asarray` and `asanyarray` are modified to inject a specified type as the default value for the `like` keyword. For example, within `np.einsum_path`, `asanyarray` is called on the array inputs. To ensure that this call returns a `SymmetricTensor` rather than an `ndarray`, we first determine the module in which this call is made (in this case, *np.core.einsumfunc*). Then
#
# ```python
# A = PermClsSymmetricTensor(3,4)
# with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#     np.einsum_path('iij', A)  # Does not cast to plain Array
# ```
#
# Using this context manager is essentially making the statement:
#
# > I have looked at all the code that will be run within this context, and I am confident that in all instances, a duck array satisfies the requirements of code calling array coercion functions like `asarray`. In no instances can a duck array be returned where a true NumPy array is needed.
#
# For example, any code which ultimately calls NumPy's C API will require true NumPy arrays. This is partly why the example above uses `einsum_path` (which is a pure Python function) and not `einsum`, which internally calls the C function `c_einsum`.
#
# Note: Rather than requiring the user to wrap calls with `make_array_like`, a better approach is to include those calls in the type-specific dispatch code, so that `np.einsum_path` always works as expected. See the implementation of `einsum_path` in *permcls_symtensor.py* for an example.

# %%
from collections.abc import Iterable as Iterable_
from contextlib import contextmanager
from numpy.core import numeric as _numeric
_make_array_like_patched_modules = set()  # Used in case of nested contexts
@contextmanager
def make_array_like(like, modules=()):
    """
    Monkey patch NumPy so that the type of `like` is recognized as an array.
    Within this context, and within `module`, the default signature of `asarray(x)`
    and `asanyarray(x)` is changed to include `like=like` (instead of `like=None`).

    .. Note:: Like must be an *instance* (not a type), and must implement the
       __array_function__ protocol. See NEP35 and NEP18.

    .. Caution:: This as hack of the ugliest kind. Please use sparingly, and
       only when no better solution is available.
    """
    if isinstance(modules, Iterable_):
        modules = set(modules)
    else:
        modules = {modules}
    #if any(mod is np for mod in modules):
    #    raise ValueError("`make_array_like` doesn't support overriding "
    #                     "methods in the base 'numpy' module.")
    # Open context: Monkey-patch Numpy function
    def asarray(a, dtype=None, order=None, *, like=like):
        if isinstance(a, type(like)):  # Without this, will break on normal arrays
            return _numeric.asanyarray(a, dtype, order, like=like)
        else:
            return _numeric.asanyarray(a, dtype, order)
    def asanyarray(a, dtype=None, order=None, *, like=like):
        if isinstance(a, type(like)): # Without this, will break on normal arrays
            return _numeric.asanyarray(a, dtype, order, like=like)
        else:
            return _numeric.asanyarray(a, dtype, order)
    new_funcs = {'asarray': asarray,
                 'asanyarray': asanyarray}
    old_funcs = {'asarray': _numeric.asarray,
                 'asanyarray': _numeric.asanyarray}
    # NB: Because most NumPy modules alias these functions when they use them,
    #     it's not sufficient to redefine np.asarray in _numeric: we need to
    #     replace the aliases in the modules.
    #     (assumption: aliases use the same function name)
    for mod in modules:
        if mod in _make_array_like_patched_modules:
            # Already patched by an outer context
            modules.remove(mod)
            continue
        for nm, f in new_funcs.items():
            if nm in mod.__dict__:
                #import pdb; pdb.set_trace()
                setattr(mod, nm, f)
    # Return control to code inside context
    try:
        yield None
    # Close context: Undo the monkey patching
    except Exception:
        raise
    finally:
        for mod in modules:
            for nm in old_funcs:
                # Iterating over .items() for some reason doesn't return the right values
                if nm in mod.__dict__:
                    setattr(mod, nm, old_funcs[nm])

# %% [markdown]
# ### Test

# %% [markdown]
# Test that the `make_array_like` context manager correctly binds custom functions to `asarray`, and cleans up correctly on exit.

# %%
if exenv in {"notebook", "jbook"}:
    import pytest
    from symtensor import DenseSymmetricTensor

    A = DenseSymmetricTensor(rank=2, dim=3)
    # Context manager works as expected…
    with make_array_like(DenseSymmetricTensor(0,0), np.core.einsumfunc):
        assert "<locals>" in str(np.core.einsumfunc.asanyarray)   # asanyarray has been substituted…
        np.einsum('iij', np.arange(8).reshape(2,2,2))  # …and einsum still works
        np.asarray(np.arange(3))                       # Plain asarray is untouched and still works
    # …and returns the module to its clean state on exit…
    assert "<locals>" not in str(np.core.einsumfunc.asanyarray)
    with pytest.warns(UserWarning):
        assert type(np.asarray(A)) is np.ndarray
    # …even when an error is raised within the context.
    try:
        with make_array_like(DenseSymmetricTensor(0,0), np.core.einsumfunc):
            assert "<locals>" in str(np.core.einsumfunc.asanyarray)
            raise ValueError
    except ValueError:
        pass
    assert "<locals>" not in str(np.core.einsumfunc.asanyarray)


# %% [markdown]
# ## Combinatorics utilities

# %% [markdown]
# ### `multinom`
# Applies
# $$\binom{n}{k_1,k_2,\dotsc,k_m} = \binom{n}{k_1} \binom{n-k_1}{k_2} \dotsb \binom{n-\sum_{i<m} k_i}{k_m} \,.$$
# Each binomial term is computed with `math.comb`.
# The $k_i$ terms are first sorted, since $\binom{n}{k}$ is less computationally expensive when $k$ is close to 0 or $n$.

# %%
def multinom(n: int, klst: List[int]) -> int:
    """
    Compute and return the multinomial
    ⎛      n      ⎞
    ⎝k1, k2, …, km⎠
    """
    klst = sorted(filter(None, klst))  # `filter` removes 0 elements
    s = sum(klst)
    if s < n:
        warn("Extending `klst` so that Σk = n.")
        klst.append(n-s)
    elif s > n:
        raise ValueError("Sum of `klst` values exceeds n.")
    nlst = [n] + list(n - np.cumsum(klst[:-1]))
    if nlst[-1] > n:
        return 0
    return math.prod(math.comb(n, k) for n, k in zip(nlst, klst))


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("multinom(2, (1, 1))", number=100000))
    display(timeit("multinom(5, (5, 0, 0, 0, 0))", number=100000))
    display(timeit("multinom(5, (3, 2, 0, 0, 0))", number=100000))
    display(timeit("multinom(5, (1, 1, 1, 1, 1))", number=100000))


# %% [markdown]
#     5.37 µs ± 49.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     6.06 µs ± 56.1 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     5.39 µs ± 22.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     6.03 µs ± 41.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# %% [markdown]
# ### `twoway_partitions`
#
# Given a list of length $S$, generate pairs of lists, with lengths $s_1$ and $s - s_1$, such that each pair contains all elements of the list. Order does not matter, but it treats the elements of the list as if they were all different. Thus a total of
#   \begin{equation*}
#   \binom{S}{s_1}
#   \end{equation*}
# pairs are created.
#
# Example:
#  [1,1,5] can be paired into pairs of lists with lengths $s_1=1, s_2=2$ in the following ways:
#
# | sublist 1 | sublist 2 |    |
# |------------|----------|----|
# |`{1}`       | `{1,5}`  |  |
# |`{1}`       | `{5,1}`  | (The two `1` elements are considered distinct.) |
# |`{5}`       | `{1,1}`  |  |
#
# Both the order in which sublist pairs are returned, and the order within sublists, is undefined. For implementation convenience and efficiency, sublists are returned as NumPy arrays, but this should be considered an implementation detail.

# %%
def twoway_partitions(lst: Sequence[Any], size1: int, size2: int, num_partitions: bool = True
    ) -> Tuple[Iterable[Sequence], Iterable[Sequence], int]:
    """
    Return all unique two-way partitions of `lst`, where partitions have
    sizes `size1` and `size2`. All elements in `lst` are treated as distinct.

    Both the order in which sublist pairs are returned, and the order within
    sublists, is undefined.

    .. Note:: The sum of `size1` and `size2` must equal the length of `lst`.

    Returns
    -------
    - Generator: sublists 1
    - Generator: sublists 2
    - int: The number of partitions if num_partitions is True
    """
    assert size1 + size2 == len(lst), 'Sizes must sum to length of `lst`.'

    arr = np.array(lst)  # To allow fancy indexing
    indices = set(range(len(lst)))
    indices_1 = [set(idcs) for idcs in itertools.combinations(indices, size1)]
    indices_2 = (indices - idcs1 for idcs1 in indices_1)  # Since we only iterate this once, we can use a generator.
    lsts_1 = (arr[list(idcs1)] for idcs1 in indices_1)
    lsts_2 = (arr[list(idcs2)] for idcs2 in indices_2)
    
    if num_partitions:
        return lsts_1, lsts_2, len(indices_1)
    else: 
        return lsts_1, lsts_2
    
def twoway_partitions_pairwise_iterator(lst: Sequence[Any], size1: int, size2: int, num_partitions: bool = True
    ) -> Tuple[Iterable[Sequence], Iterable[Sequence], int]:
    """
    Return all unique two-way partitions of `lst`, where partitions have
    sizes `size1` and `size2`. All elements in `lst` are treated as distinct.

    Both the order in which sublist pairs are returned, and the order within
    sublists, is undefined.

    .. Note:: The sum of `size1` and `size2` must equal the length of `lst`.

    Returns
    -------
    - Generator: pairs of (sublists 1,sublist2)
    - int: The number of partitions if num_partitions is True
    """
    assert size1 + size2 == len(lst), 'Sizes must sum to length of `lst`.'

    arr = np.array(lst)  # To allow fancy indexing
    indices = set(range(len(lst)))
    indices_1 = [set(idcs) for idcs in itertools.combinations(indices, size1)]
    indices_2 = (indices - idcs1 for idcs1 in indices_1)  # Since we only iterate this once, we can use a generator.
    lsts_1 = (arr[list(idcs1)] for idcs1 in indices_1)
    lsts_2 = (arr[list(idcs2)] for idcs2 in indices_2)
    
    if num_partitions:
        return zip(lsts_1, lsts_2), len(indices_1)
    else: 
        return zip(lsts_1, lsts_2)
    


# %%
def nway_partitions_iterator(lst: Sequence[Any], sizes: Tuple[int], num_partitions: bool = True
    ) -> Tuple[Iterable[Sequence], Iterable[Sequence], int]:
    """
    Return all unique n-way partitions of `lst`, where partitions have
    sizes as in `sizes`. All elements in `lst` are treated as distinct.

    Both the order in which sublist pairs are returned, and the order within
    sublists, is undefined.

    .. Note:: The sum of `sizes` must equal the length of `lst`.
    .. Note:: Works only up to n = 4.

    Returns
    -------
    - Generator: n-pairs of (sublists 1,..., sublist n)
    - int: The number of partitions if num_partitions is True
    """
    assert sum(sizes) == len(lst), 'Sizes must sum to length of `lst`.'
    n = len(sizes)
    if n>4: 
        raise NotImplementedError('decompositions with more than four different components not available')
    else:
        if n==2: 
            return twoway_partitions_pairwise_iterator(lst, sizes[0], sizes[1], num_partitions = num_partitions)
        elif n> 2: 
            lsts_1 = []
            lsts_2 = []
            lsts_3 = []
            lsts_4 = []
            
            arr = np.array(lst)  # To allow fancy indexing
            indices = set(range(len(lst)))
            #iteratively construct subsets
            for indcs_1 in itertools.combinations(indices, sizes[0]):
                indcs_1_set = set(indcs_1)
                indices_23_set = indices - indcs_1_set 
                for indcs_2 in itertools.combinations(list(indices_23_set), sizes[1]):
                    if n ==3:
                        indcs_2_set = set(indcs_2)
                        indcs_3_set = indices_23_set - indcs_2_set
                        lsts_1 += [arr[list(indcs_1_set)]]
                        lsts_2 += [arr[list(indcs_2_set)]]
                        lsts_3 += [arr[list(indcs_3_set)]]
                    else:
                        indcs_2_set = set(indcs_2)
                        indcs_34_set = indices_23_set - indcs_2_set
                        assert n==4
                        for indcs_3 in itertools.combinations(list(indcs_34_set), sizes[2]):
                            indcs_3_set = set(indcs_3)
                            indcs_4_set = indcs_34_set - indcs_3_set
                            lsts_1 += [arr[list(indcs_1_set)]]
                            lsts_2 += [arr[list(indcs_2_set)]]
                            lsts_3 += [arr[list(indcs_3_set)]]
                            lsts_4 += [arr[list(indcs_4_set)]]
        if num_partitions:
            if n==3:
                return zip(lsts_1, lsts_2, lsts_3), len(lsts_1)
            else: 
                return zip(lsts_1, lsts_2, lsts_3,lsts_4), len(lsts_1)
        else:
            if n==3:
                return zip(lsts_1, lsts_2, lsts_3)
            else: 
                return zip(lsts_1, lsts_2, lsts_3,lsts_4)
    
    

# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("twoway_partitions((5, 5, 5, 10, 10, 3, 2, 1), 4, 4)", number=100000))


# %% [markdown]
#     11.1 µs ± 82.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# %% [markdown]
# ## Array utilities

# %% [markdown]
# ### `symmetrize`
#
# **[TODO]** A version with `PermClsSymmetricTensor` which avoids redundant computation. Using `functools.singledispatch`, that version can be implemented in *permcls_symtensor.py*, with the function below used as fallback. Alternatively, an *array function* would also work, and might be more consistent with how we do dispatch elsewhere.

# %%
def symmetrize(dense_tensor: Array, out: Optional[Array]=None) -> Array:
    # OPTIMIZATION:
    # - If possible (i.e. if dense_tensor ≠ out), use `out` to avoid intermediate copies in sum
    # - In this regard, perhaps a version using `np.add.accumulate` might be faster ?
    # - Does not seem to use threading: only one CPU is active, even with large matrices
    if len(set(dense_tensor.shape)) > 1:
        raise ValueError(f"Cannot symmetrize tensor of shape {denser_tensor.shape}: "
                         "Dimensions do not all have the same length.")
    D = np.ndim(dense_tensor)
    if D <= 1:
        return dense_tensor # Nothing to symmetrize: EARLY EXIT
    
    n = math.prod(range(1,D+1))  # Factorial – number of permutations
    if out is None:
        out = np.empty(dense_tensor.shape, dtype=dense_tensor.dtype,
                       like=dense_tensor)
    out[:] = sum(dense_tensor.transpose(σaxes) for σaxes in itertools.permutations(range(D))) / n
    return out


# %%
if exenv in {"notebook", "jbook"}:
    for r in 2, 3, 4, 6, 8:
        for S in 1000, 1000000:
            #if r >= 8 and S >= 1000000:
            #    continue
            l = np.ceil(S**(1/r)).astype(int)  # Keep roughly same number of elements for each rank
            A = np.random.random((l,)*r)
            nops = 10*l**r * math.factorial(r)  # Estimate of the number of operations
            res = timeit("symmetrize(A)", **get_timeit_kwargs(nops))
            desc = "Array shape: " + " x ".join((f"{l}" for _ in range(r))) + f" = {l**r}"
            print(f"{desc:<52} – {res}")


# %% [markdown]
#     Array shape: 32 x 32 = 1024                       – 6.94 µs ± 21.4 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     Array shape: 1000 x 1000 = 1000000                – 2.51 ms ± 13.7 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     Array shape: 6 x 6 x 6 x 6 = 1296                 – 71.1 µs ± 248 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
#     Array shape: 32 x 32 x 32 x 32 = 1048576          – 53.4 ms ± 125 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
#     Array shape: 3 x 3 x 3 x 3 x 3 x 3 x 3 x 3 = 6561 – 592 ms ± 1.49 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

# %% [markdown]
# ### `is_symmetric`
#
# Note that at least for tensors of rank ≤ 2, `is_symmetric` is actually slower than `symmetrize`.

# %%
def is_symmetric(dense_tensor: Array, rtol=1e-5, atol=1e-8) -> bool:
    """
    Return True if `dense_tensor` is symmetric.
    Arrays are compared with `numpy.allclose`; tolerance parameters `rtol`
    and `atol` are passed on to that function.
    """
    if len(set(dense_tensor.shape)) > 1:
        return False
    D = dense_tensor.ndim
    n = math.factorial(D)  # Number of permutations
    A = dense_tensor
    return all(np.allclose(A, A.transpose(σaxes), rtol, atol, equal_nan=True)
               for σaxes in itertools.permutations(range(D)))

# %%
if exenv != "module":
    A = np.random.rand(3,3,3)
    S = symmetrize(A)
    assert not is_symmetric(A)
    assert is_symmetric(S)
    S = S.round(10)  # Avoid the need to do all comparisons with 'is_close'
    assert S[0, 1, 0] == S[1, 0, 0] == S[0, 0, 1]
    assert S[0, 2, 0] == S[2, 0, 0] == S[0 ,0, 2]
    assert S[0, 0, 0] != S[1, 1, 1] != S[2, 2, 2]

# %%
if exenv in {"notebook", "jbook"}:
    for r in 2, 3, 4, 6, 8:
        for S in 1000, 1000000:
            l = np.ceil(S**(1/r)).astype(int)  # Keep roughly same number of elements for each rank
            A = np.random.random((l,)*r)
            nops = 1e3*math.factorial(r)  # Estimate of the number of operations (Assumption: parallelized comparison)
            res = timeit("is_symmetric(A)", **get_timeit_kwargs(nops))
            # res = %timeit -q -o is_symmetric(A)
            desc = "Array shape: " + " x ".join((f"{l}" for _ in range(r))) + f" = {l**r}"
            print(f"{desc:<52} – {res}")


# %% [markdown]
#     Array shape: 32 x 32 = 1024                          – 12.6 µs ± 86.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     Array shape: 1000 x 1000 = 1000000                   – 3.17 ms ± 23.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     Array shape: 6 x 6 x 6 x 6 = 1296                    – 15.3 µs ± 112 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     Array shape: 32 x 32 x 32 x 32 = 1048576             – 3.13 ms ± 14 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     Array shape: 3 x 3 x 3 x 3 x 3 x 3 x 3 x 3 = 6561    – 36.4 µs ± 181 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
#     Array shape: 6 x 6 x 6 x 6 x 6 x 6 x 6 x 6 = 1679616 – 5.51 ms ± 21.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %% [markdown]
# ### `symmetrize_index(index)`
#
# Returns an advanced index representing all the symmetrically equivalent indices
# For example,
#
# $$I := \texttt{(0,1,2)} \mapsto \hat{I} := \texttt{([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [2, 1, 2, 0, 1, 0])}$$
#
# such that $A[\hat{I}] = 1$ produces
#
#       [[[0., 0., 0.],
#         [0., 0., 1.],
#         [0., 1., 0.]],
#
#        [[0., 0., 1.],
#         [0., 0., 0.],
#         [1., 0., 0.]],
#
#        [[0., 1., 0.],
#         [1., 0., 0.],
#         [0., 0., 0.]]]
#
# **Should you inline this function instead ?:** The savings from inlining are ~28 ns / function call, for a function defined in the local namespace. See [/docs/developors/comparitive_timings.py](../docs/developers/comparitive_timings.py#Inlining-a-function). So at least for anything with rank > 4, this should be negligible.
# It's still a good idea to save a few cycles by bringing the function into the local namespace though:
#
# ```python
# symmetrize_index = utils.symmetrize_index
# for index in indices:
#     symmetrize_index(index)
# ```

# %%
def symmetrize_index(index: Tuple[int]) -> Tuple[List[int], ...]:
    # NB: dict.fromkeys is used to keep only unique permutations
    return tuple(list(permuted_index) for permuted_index in  # Convert from tuple to list to trigger advanced indexing
                 zip(*dict.fromkeys(itertools.permutations(index)).keys()))


# %%
if exenv in {"notebook", "jbook"}:
    assert symmetrize_index((0,1,2)) == ([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [2, 1, 2, 0, 1, 0])

    kwds = lambda rank: get_timeit_kwargs(1e3*math.factorial(rank))
    display(timeit("symmetrize_index((0,0))", **kwds(8)))
    display(timeit("symmetrize_index((0,0,1,1))", **kwds(8)))
    display(timeit("symmetrize_index((0,0,1,1,2,2))", **kwds(8)))
    display(timeit("symmetrize_index((0,0,1,1,2,2,3,3))", **kwds(8)))
    display(timeit("symmetrize_index((0,0,0,0,2,2,2,2))", **kwds(8)))

# %% [markdown]
#     844 ns ± 11.9 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     2.33 µs ± 59.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     43 µs ± 158 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
#     2.68 ms ± 21.9 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     2.22 ms ± 12.1 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)

# %% [markdown]
# ### `symmetrize_slice_index` **[TODO]**
#
# Similar to `symmetrize_index`, for when the index includes slices.
# Returns a generator: for each `index` included in the slice, return `symmetrize_index(index)`.
#
# TBD: What to do with duplicate indices / multiple slice dimensions.

# %% [markdown]
# ## Permutation class utilities

# %% [markdown]
# :::{Note}
# There is a bijective map between string representations of permutation classes – `'iijk'` – and count representations – `(2,1,1)`. Public methods of `SymmetricTensor` use strings, while private methods use counts.
# :::

# %% [markdown]
# For public utility functions, both representations are accepted, while for private utility functions the count representation is used.

# %% [markdown]
# ### Public utilities

# %% [markdown]
# #### Converting from permutation class labels to counts

# %%
indices = 'ijklmnabcdefghopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'
    # cf numpy.core.einsumfunc.einsum_symbols; here we move 'ijklmn' to the front, so they are used as defaults


# %% [markdown]
# ##### `dense_index_to_permclass_label`
# Convert dense index to σ-class label

# %%
def dense_index_to_permclass_label(key: Tuple[int]) -> str:
    """
    Given an index tuple in dense format, return the string of the
    permutation class to which it belongs.

    Usage:
    >>> get_permclass((5,0,1,0))
    'iijk'
    """
    σcls = ""
    for count, s in zip(_get_permclass(key), indices):
        σcls += s*count
    return σcls


# %% [markdown]
# ##### `permclass_counts_to_label`
# Convert σ-class label to index counts

# %%
def permclass_counts_to_label(index_counts: Tuple[int]) -> str:
    """
    Convert an index from tuple of counts to string representation.

    Usage:
    >>> A.permclass_counts_to_label((2,1,1))
    'iijk'
    """
    return ''.join((s*c for s,c in zip(indices, index_counts)))


# %% [markdown]
# ##### `permclass_label_to_counts`
# Convert σ-class index counts to label

# %%
def permclass_label_to_counts(class_label: str) -> Tuple[int]:
    """
    Convert an index from string to tuple (counts) representation.
    """
    return tuple(sorted(
        (class_label.count(s) for s in set(class_label)),
        reverse=True))


# %% [markdown]
# #### Combinatorics

# %% [markdown]
# ##### `get_permclass_multiplicity`

# %%
def get_permclass_multiplicity(σcls: Union[str, Tuple[int]]) -> List[int]:
    """
    Return the number of times each component in the permutation class given
    by `class_label` is repeated in the full tensor.

    .. Note:: This is very similar to `permclass_label_to_counts`, with the
       difference that the output always has the same length as the rank.

    """
    if isinstance(σcls, str):
        index_counts = permclass_label_to_counts(σcls)
        rank = sum(index_counts)
    else:
        # Assume index count tuple
        rank = sum(σcls)
        index_counts = σcls + (0,)*(rank-len(σcls))
    return multinom(rank, index_counts)


# %% [markdown]
# ##### `get_permclass_size`

# %%
def get_permclass_size(σcls: Union[str, Tuple[int]], dim: int) -> int:
    """
    Return the number of independent components in the permutation class
    associated to `σcls`.
    """
    if isinstance(σcls, str):
        σcls = permclass_label_to_counts(σcls)
    return _get_permclass_size(σcls, dim)


# %% [markdown]
# #### Comparison

# %% [markdown]
# ##### `is_subσcls`

# %%
def is_subσcls(σcls: Union[str, Tuple[int]],
               subσcls:  Union[str, Tuple[int]]) -> bool:
    """
    Check if subσcls is a sub permutation class of σcls.
    Both σcls and subσcls are strings where the entries denote the number of repetitions of a single index.
    For example 'iiij' is the permutation class of indices of a fourth order tensor where all indices are equal but one.
    Then 'iij' is a subσcls of σcls,
    but 'ijk' is not a subσcls of 'iiij', because for 'ijk', there have to be at least three different indices.
    """
    if isinstance(σcls, str):
        σcls = permclass_label_to_counts(σcls)
    if isinstance(subσcls, str):
        subσcls = permclass_label_to_counts(subσcls)
    return _is_subσcls(σcls, subσcls)


# %% [markdown]
# ### Private utilities
#
# **OPTIMIZATION** These private functions are good candidates for Cythonization.

# %% [markdown]
# #### `_all_index_counts`
# Helper function for `SymmetricTensor.perm_classes`.
# Returns an iterator which generates all unique sets of index counts in the string representation of a permutation class up to permutation. For example, the counts for class `'iij'` are represented as `[2, 1]`. The resulting output has the shape $[[c_1, \dots], [c_1', \dots], \dots]$. Each list can have variable length, depending on the number of unique indices in that set.
#
# This function uses recursion to deal with the variable length of the count list. It applies three rules:
# - The *number of indices* must not exceed the rank.
#   This is enforced with the `remaining_idcs` argument.
# - The *sum of counts* must not exceed the rank.
#   This is enforced with the `remaining_counts` argument.
# - A count $c_k$ cannot be greater than $c_{k-1}$
#   This avoids duplicates like `'iij'` and `ijj`
#   and is enforced with the `max_count` argument.
#
# > The particular case when `remaining_counts` = `max_count` = 0 is treated specially, to ensure that it  returns `[[]]` (rather than `[[0]]`).

# %%
# NB: Index counts are also internally used as class labels
def _all_index_counts(remaining_idcs, remaining_counts, max_count):
    """
    Return an iterator which generates all unique sets of index counts in the
    string representation of a permutation class up to permutation.
    """
    assert remaining_counts <= remaining_idcs * max_count
    if remaining_counts <= max_count:
        if remaining_counts == 0:
            # This can happen with a rank 0 tensor; in that case, remaining_idcs = remaining_counts = max_count = 0
            # In this case, return an empty iterator – there are no indices left
            yield []
        else:
            yield [remaining_counts]
    if remaining_idcs > 1:
        for c in range(min(remaining_counts-1, max_count), 0, -1):
            for subcounts in _all_index_counts(
                  remaining_idcs-1, remaining_counts-c, max_count=c):
                yield [c] + subcounts


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("_all_index_counts(8, 8, 8)"))
    display(timeit("list(_all_index_counts(8, 8, 8))",  number=10000))


# %% [markdown]
#     120 ns ± 2.02 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
#     23.6 µs ± 229 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)

# %% [markdown]
# #### `_get_permclass`
# Given either a tuple of numerical indices (`(3, 0, 3)`) or string indices (`('j','i', 'j')`), return the tuple representing the permutation class it belongs to (`(2, 1)`).
# Note that a string index must be split into a tuple of single chars.
#
# **OPTIMIZATION**: I'm not sure this function can be accelerated much in pure Python, but Cython implementation might get a 100x speed up.

# %%
def _get_permclass(key: Tuple[int,str]) -> Tuple[int]:
    """
    Given either a tuple of numerical indices (`(3, 0, 3)`) or string indices
    (`('j','i', 'j')`), return the tuple representing the permutation class it
    belongs to (`(2, 1)`).

    .. Caution:: Note that a string index must be split into a tuple of single
       chars.
    """
    #_, i_counts = np.unique(key, return_counts=True)
    i_counts = (len(list(grouper)) for _, grouper in itertools.groupby(sorted(key)))
    return tuple(sorted(i_counts, reverse=True))


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("_get_permclass((3, 0, 3))"))
    display(timeit("_get_permclass(tuple('iij'))"))
    display(timeit("_get_permclass((3, 0, 3, 4, 5, 5))"))
    display(timeit("_get_permclass(tuple('iijjkl'))"))
    display(timeit("_get_permclass((3, 0, 3, 4, 5, 5, 7, 1))"))
    display(timeit("_get_permclass(tuple('iijjklmn'))"))


# %% [markdown]
#     846 ns ± 1.99 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     948 ns ± 10.7 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.25 µs ± 14.3 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.37 µs ± 10.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.61 µs ± 6.66 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.72 µs ± 24.4 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# %% [markdown]
# #### `_get_permclass_size(σcls, dim)`
#
# Return the number of independent components in the given permutation class.
# Computed as
# $$s_{\hat{I}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!} \,,$$
# where $r$ is the rank, $l$ is the number of different indices in the class and $m_n$ is the number of different indices which appear $n$ times in $I$. Examples:
#
# | $\hat{I}$ | $\hat{I}$ (str) | $l$ | $m$ |
# |-----------|-----------------|-----|-----|
# |`(3,2)`    | `iiijj`         | 2   | 0, 1, 1, 0, 0 |
# |`(1,1,1)`  | `ijk`           | 3   | 3, 0, 0 |

# %%
def _get_permclass_size(σcls: Tuple[int], dim: int) -> int:
    counts = σcls
    rank = sum(counts)
    l = len(counts)
    permutable_groups = [counts.count(c) for c in range(1, rank+1)]
    res = (math.prod(range(dim, dim-l, -1))
           / math.prod(math.factorial(m) for m in permutable_groups))
    assert res.is_integer()
    return int(res)


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("_get_permclass_size((2,2), 100)"))              # Rank 4
    display(timeit("_get_permclass_size((2,2,2,2), 100)"))          # Rank 8
    display(timeit("_get_permclass_size((1,1,1,1,1,1,1,1), 100)", number=100000))  # Rank 8


# %% [markdown]
#     1.25 µs ± 17.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.83 µs ± 8.6 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     2.13 µs ± 9.87 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)

# %% [markdown]
# #### `_is_subcls`(σcls, subσcls)
#
# Return True if `σcls` is a sub permutation class of σcls.

# %%
def _is_subσcls(σcls: Tuple[int], subσcls: Tuple[int]) -> bool:
    """
    Check if subσcls is a sub permutation class of σcls.
    Both σcls and subσcls are tuples where the entries denote the number of repetitions of a single index.
    For example σcls = (3,1) corresponds to ther σcls string 'iiij'. Then (2,1) <-> 'iij' is a subσcls of σcls,
    but (1,1,1) <-> 'ijk' is not a subσcls of 'iiij' <-> (3,1).
    """
    return len(σcls) >= len(subσcls) and all(a >= b for a, b in zip(σcls, subσcls))


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("_is_subσcls((4,2,2), (4,2))"))
    display(timeit("_is_subσcls((4,2,2), (4,1))"))
    display(timeit("_is_subσcls((2,2,2,2), (4,2))"))
    display(timeit("_is_subσcls((2,2,2,2), (2,2,2,2))"))
    display(timeit("_is_subσcls((2,2,2,2), (2,))"))


# %% [markdown]
#     409 ns ± 2.96 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     407 ns ± 4.41 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     404 ns ± 8.47 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     488 ns ± 2.39 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     365 ns ± 1.48 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# %% [markdown]
# #### `_perm_classes(rank)`
# iterates over all permutation classes of a tensor in their private representation.
# Example:
# `_perm_classes(5)`
# yields
# ```
# (5,)
# (4, 1)
# (3, 2)
# (3, 1, 1)
# (2, 2, 1)
# (2, 1, 1, 1)
# (1, 1, 1, 1, 1)
#
# ```

# %%
def _perm_classes(rank):
    return (tuple(counts)
            for counts in _all_index_counts(rank, rank, rank))


# %%
if exenv in {"notebook", "jbook"}:
    display(timeit("_perm_classes(2)"))
    display(timeit("_perm_classes(4)"))
    display(timeit("_perm_classes(8)"))

# %% [markdown]
#     296 ns ± 2 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     304 ns ± 0.755 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     294 ns ± 1.01 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)

# %% [markdown]
# ### Tests

# %%
if exenv != "module":
    def test_rank_dim() -> Generator:
        for d, r in itertools.product([2, 3, 4, 6, 8], [2, 3, 4, 5, 6]):
            yield r, d

# %% [markdown]
# #### Combinatorics

# %% [markdown]
# ##### σ-class sizes
#
# The tests below confirm that the permutation classes form a partition of tensor indices: the sum of their class sizes equals the total number of independent components in a symmetric tensor, which is given by
# $$\binom{d + r - 1}{r}\,.$$
# This is a well-known expression; it can be found for example [here](http://www.physics.mcgill.ca/~yangob/symmetric%20tensor.pdf).
# This test gives us good confidence that the methods `perm_classes` and `get_permclass_size` are correctly implemented.

    # %%
    for r, d in test_rank_dim():
        assert (sum(get_permclass_size(σcls, d) for σcls in _perm_classes(r))
                == math.prod(range(d, d+r)) / math.factorial(r))

# %% [markdown]
# ##### σ-class multiplicities
#
# We now check if class multiplicities are correctly evaluated, to validate `get_permclass_multiplicity`. We do this by checking the identity
# $$\sum_\hat{σ} s_\hat{σ} = d^r\,.$$
# (Here $\hat{σ}$ is a permutation class and $s_{\hat{σ}}$ the size of that class.)

    # %%
    for r, d in test_rank_dim():
        assert (sum(get_permclass_size(σcls, d) * get_permclass_multiplicity(σcls)
                    for σcls in _perm_classes(r))
                == d**r)

# %% [markdown]
# ##### `is_subcls`

    # %%
    # Rank 3
    assert is_subσcls('iii', 'ii')
    assert is_subσcls('iij', 'ij')
    assert is_subσcls('ijk', 'ij')
    assert not is_subσcls('iii', 'ij')
    assert not is_subσcls('ijk', 'ii')
    # Rank 4
    assert is_subσcls('iiii', 'iii')
    assert is_subσcls('iiii', 'ii')
    assert is_subσcls('iijj', 'ij')
    assert is_subσcls('iijj', 'iij')
    assert is_subσcls('iijk', 'iij')
    assert is_subσcls('iijk', 'ijk')
    assert is_subσcls('iijk', 'ij')
    assert is_subσcls('iijk', 'i')
    assert not is_subσcls('iiii', 'ij')
    assert not is_subσcls('iiii', 'ijk')
    assert not is_subσcls('iijj', 'ijk')
    assert not is_subσcls('iijk', 'ijkl')
    assert not is_subσcls('iijk', 'iijj')
    assert not is_subσcls('iijj', 'iii')

# %% [markdown]
# ## Profiling

# %%
import sys
from mackelab_toolbox.utils import total_size

from typing import Type, Iterable, Tuple


# %%
def compare_memory(
    symtensor_constructor: Type[SymmetricTensor],
    ranks_dims: Iterable[Tuple[int, int]],
    dtype="float64"
      ) -> Overlay:
    """
    Returns a plot comparing the memory requirements of a SymmetricTensor
    subclass to storing the same data as a dense NumPy array.
    Plot shows dependency on rank and dimension.

    :param:symtensor_constructor: A callable which returns instances of the
       SymmetricTensor we want to profile. Must take three arguments:
       *rank*, *dim* and *dtype*. This is intentionally compatible with a
       `SymmetricTensor` initializer, so one can simply pass a class.
       Passing a wrapper function is useful if a SymmetricTensor can have
       different footprints – eg. a wrapper could ensure that underlying data
       of a `PermClsSymmetricTensor` are full sized arrays, not just scalars.
    :param:ranks_dims: Iterable which returns (rank, dim) tuples.
       Use `itertools.product` to create an iterator from two lists of ranks and dims.
    :param:dtype: Use this dtype for the test tensors; any value accepted by
       ``numpy.dtype`` is valid. Default is ``float64``, but since reported
       values are relative, they depend only slightly on dtype. Smaller dtypes will
       evaluate quicker since less memory is allocating during testing, but will
       report proportionally higher overhead costs.
    """
    import holoviews as hv  # Only import holoviews when needed

    dtype = np.dtype(dtype)
    array_overhead = sys.getsizeof(np.array([], dtype=dtype))
    itemsize = dtype.itemsize
    data = []
    for rank, dim in ranks_dims:
        # Check that this rank/dim combination doesn't produce a crazy large tensor
        A = symtensor_constructor(rank=rank, dim=dim, data=None)
        if A.size > 1e4:
            # At around ~1e7, we start exceeding hard memory limits, and much
            # below that we may already unnecessary use of swap space.
            # => Estimate the overhead and compute what the data would be (for
            #    (large tensors, rel error on the overhead should be negligible)
            B = symtensor_constructor(rank=rank, dim=1)  # Assumption: the overhead may depend on rank, but not on dim
            sym_size = total_size(B) + (A.size-B.size)*itemsize
        else:
            # Size is reasonable; we may proceed with actually instantiating the tensor
            A = symtensor_constructor(rank=rank, dim=dim, dtype=dtype)
            sym_size = total_size(A)
        dense_size = dim**rank*itemsize + array_overhead
        rel_size = sym_size/dense_size  # Works because SymmetricTensor adds itself to mackelab_toolbox.utils._total_size_handlers
        data.append((rank, dim, rel_size))
    data = hv.Table(tuple(zip(*data)), kdims=["rank", "dim"], vdims=["rel size"])

    curves = data.groupby("rank", group_type=hv.Curve).overlay()

    fig = hv.Overlay(curves) * hv.HLine(1)
    fig.opts(hv.opts.HLine(color='grey', line_dash='dashed', line_width=2, alpha=0.5),
             hv.opts.Curve(width=500, logy=True, logx=True),
             hv.opts.Overlay(legend_position='right'))

    return fig


# %% tags=["remove-input"]
from mackelab_toolbox.utils import GitSHA    # Keep this cell below all others
GitSHA(show_hostname=True)
