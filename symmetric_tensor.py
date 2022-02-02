# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: py:percent,ipynb
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-jupytext.kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#   kernelspec:
#     display_name: statGLOW
#     language: python
#     name: statglow
# ---

# %% [markdown]
# # Symmetric tensor class

# %%
from __future__ import annotations
from warnings import warn
from ast import literal_eval
import itertools
import math  # For operations on plain Python objects, math can be 10x faster than NumPy
import numpy as np
from pydantic import BaseModel

from typing import Union, ClassVar, Iterator, Generator, Dict, List, Tuple, Set
import statGLOW
from statGLOW.smttask_ml.scityping import Serializable, Array, DType
import pytest
from statGLOW.utils import does_not_warn
from collections import Counter

# %%
if __name__ == "__main__":
    import holoviews as hv
    hv.extension('bokeh')

# %%
__all__ = ["SymmetricTensor"]

# %% [markdown]
# ## Rationale
#
# If we naively store tensors as high-dimensional arrays, they quickly grow quite large. However, reality does not need to be so bad, for two main reasons:
# 1. Since tensors must be symmetric, a fully specified tensor contains a lot of redundant information.
# 2. In practice, certain terms might be more important than others – e.g. the diagonal elements $A_{ii\dotsb}$.
#
# We want to avoid storing each element of the tensor, while still being efficient with those elements we do store. In particular, “similar” terms (like diagonal terms) should be stored as arrays, so that looping remains efficient and certain vectorized operations remain possible.

# %% [markdown]
# ## Approach
#
# Construct a `SymmetricTensor` class for which the outward interface is similar to an array: `A[1,2,3]` retrieves the tensor component $A_{123}$.
#
# Internally, components are grouped into permutation classes based on the form of their index. For example, all components with indices of the form `'iiii'` are grouped into one array; similarly for `'iij'`, etc. Instead of an array, a permutation class can also be associated to a single value. For example, if components of the form `'ijkl'` should all be zero, we simply associate zero to that class, instead of allocating a potentially large array of size $\frac{d(d-1)(d-2)(d-3)}{4!}$ elements.

# %% [markdown]
# **Situations where a `SymmetricTensor` may be well suited**
#
# - Constructing tensors while ensuring they are symmetric.
# - Storing symmetric tensors in a memory-efficient manner.
# - Computing combinatorics for particular terms (number of equivalent permutations; number of independent components within an “permutation class”)
# - Iterating over particular permutation classes (e.g. all terms of the form `'iijk'`).
# - Iterating over the entire tensor, when the order does not matter.
# - Broadcasted operations on permutation classes.
# - Random access of a specific entry.
#   (The current implementation is not quite $\mathcal{O}(1)$, but compensates by the much lower memory size compared to a dense array.)
#
# **Situations for which a `SymmetricTensor` may be less well suited:**
#
# - Slicing a tensor as a normal array (per axis).
#   (It might be possible to implement this, although with additional overhead compared to normal arrays.)
# - Broadcasted operations on the whole tensor.
#   (While it might possible to get this to work, it likely would not be much faster than just looping.)
# - Unsupported array operations.
#   Most operations (like `.dot`) are not implemented, but could be added as needed.
#
# In short, with a `SymmetricTensor` one gains a much reduced memory footprint for large tensors and very efficient operations on permutation classes, which are paid for by less efficient operations on the tensor as a whole.

# %% [markdown]
# ### Further development
#
# In this initial version, `SymmetricTensor` is more of a framework than a full-featured class. It has the basic indexing support that felt necessary for basic usage as a storage container, so that we can start using it and add features as they become needed.
#
# I anticipate that a lot of the work with `SymmetricTensor` will revolve around building different iterators for the tensor components (per row, per column, etc.). Most features we will need can likely be written a few simple lines once we have an appropriate iterator.

# %% [markdown]
# ### Notation & conceptual design
#
# **Multi-index**
#
# A particular element of a tensor $A$ of rank $r$ is denoted $A_{i_1,\dotsc,i_r}$. For notational brevity, we use $I$ to denote the multi-index, i.e.
# $$I := i_1,\dotsc,i_r\,.$$
# When the rank of the multi-index is important, we may use $I_r$.
#
# **Index class**
#
# The basic property of a symmetric tensor $A$ of rank $r$ is that for any permutation $σ \in S_r$,
# $$A_{I} = A_{σ(I)} \,.$$
# Thus the existence of a permutation such that $I' = σ(I)$ defines an equivalence relation on the set of multi-indices. An *index class* is the set of all indices satisfying this relation, and is denoted by with a representative element of that classes:
# $$\hat{I} := \{I' \in \mathbb{R}^r | \exists σ \in S_r, I' = σ(I)\} \,.$$
#
# In this implementation, w typically select the representative index by grouping equal indices, sorting groups first in decreasing multiplicity, and then in increasing index value.
#
# **Permutation class**
#
# A *permutation class* is a set of index classes which have the same index repetition pattern; for example, $\widehat{1110}$ and $\widehat{2220}$ both have four indices, with one repeated three times. We can represent these classes in two different ways:
#
# - A string: `'iiij'`;
# - A tuple of repetitions: `(3, 1)`
#
# The first notation is used for public-facing functions. The second is used internally and is described in more detail [below](#Indexing-utilities).
#
# There are a few advantages to grouping permutation classes:
#
# - The number of permutation classes is independent of $r$ (if we allow for empty classes).
# - Index classes in the same permutation class all have the same size.
#
# We also define
#
# - The **multiplicity** of a permutation class as the size of each of its index classes.
# - The **size** of a permutation class as the number of index classes it contains.
#
# **Notation summary**
#
# | Symbol | Desc| Examples |
# |------|-----|----------|
# |$d$|dimension| 2 |
# |$r$|rank| 4 |
# | $I$ | Multi-index | $1010$<br>$1011$ |
# | $\hat{I}$ | Index class | $\widehat{0011}$<br>$\widehat{1110}$ |
# | $\hat{σ}$ | Permutation class | `iijj`<br>`iiij` |
# |$γ_{\hat{σ}}$|multiplicity|6<br>4|
# |$s_{\hat{σ}}$|size|1<br>2|
#
# NB: In code, we usually just write $σ$ instead of $\hat{σ}$.

# %% [markdown]
# **Storage**
#
# Tensors are stored as dictionaries, containing one flat array per permutation class. Additionally, if all entries associated to a permutation are equal, we allow storing that value as a scalar. For example, a rank 3, dimension 3 tensor $A$ would be stored as three arrays:
#
# | $\hat{σ}$ | values |
# |-----------|--------|
# |`iii` | $$(A_{000}, A_{111}, A_{222})$$ |
# |`iij` | $$(A_{001}, A_{002}, A_{110}, A_{112}, A_{220}, A_{221})$$ |
# |`ijk` | $$(A_{012})$$ |
#
# Each array value is associated to an index class. Helper functions are provided to iterate over index classes in a consistent, predictable order, so that the position in an array suffices to associate a value to an index class. (See [`index_iter`](#index_iter).)
#
# This design has the following advantages:
#
# - The number of different arrays one needs to maintain is determined only by the rank of the matrix, and remains modest. For example, a rank-4 tensor requires only 4 arrays. The overhead due to the additional Python structures should thus be manageable, and large dimensional tensors maximally benefit from NumPy's memory efficiencies since they are stored as a few long, flat arrays.
# - Tensors with additional structure (e.g. diagonal tensors) can be efficiently represented by assigning a scalar (typically 0) to potentially large permutation classes.
# - Sums over a tensor can conceivably be performed by a vectorized operation on the array of a permutation class, then multiplying the result by the multiplicity of that class.
#

# %% [markdown]
# ### Usage

# %%
if __name__ == "__main__":
    from symmetric_tensor import SymmetricTensor

# %% [markdown]
# When created, `SymmetricTensors` default to being all zero. Note that only one scalar value is saved per “permutation class”, making this especially space efficient.

    # %%
    A = SymmetricTensor(rank=4, dim=6)
    #display(A)

# %% [markdown]
# Each permutation class can be assigned either a scalar, or an array of length matching the number of independent components of that class. For example, to make a tensor with 1 on the diagonal and different non-zero values for the double paired terms `'iijj'`, we can do the following.
# Note the use of `get_class_size` to avoid having to compute how many independent `'iijj'` terms there are.

    # %%
    A['iiii'] = 1
    A['iijj'] = np.arange(A.get_class_size('iijj'))
    #display(A)

# %% [markdown]
# The `indep_iter` and `index_iter` methods can be used to obtain a list of values where *each independent component appears exactly once*. Note that component values stored as scalars are expanded to the size of their class.

    # %%
    hv.Table(zip((str(idx) for idx in A.index_iter()),
                 A.indep_iter()),
             kdims=["broadcastable index"], vdims=["value"])

# %% [markdown]
# Conversely, the `flat` method will return as many times as it appears in the full tensor, as though it was called on a dense representation of that tensor (although the order will be different).
#
# `flat_index` returns the index associated to each value.

    # %%
    hv.Table(zip(A.flat_index, A.flat), kdims=["index"], vdims=["value"])

# %% [markdown]
# The number of independent components can be retrieved with the `size` attribute.

    # %%
    A.size

# %% [markdown]
# To get the size of the full tensor, we need to multiply each permutation class $\hat{σ}$ size by its multiplicity $γ_{\hat{σ}}$ (the number of times components of that class are repeated due to symmetry).

    # %%
    (math.prod(A.shape)
     == sum(A.get_class_size(σcls)*A.get_class_multiplicity(σcls)
            for σcls in A.perm_classes)
     == A.dim**A.rank
     == 1296)

# %% [markdown]
# Like the sparse arrays of *scipy.sparse*, a `SymmetricTensor` has a `.todense()` method which returns the equivalent dense NumPy array.

    # %%
    Adense = A.todense()

    # %% tags=["hide-input"]
    l1 = str(Adense[:1,:2])[:-1]
    l2 = str(Adense[1:2,:2])[1:]
    print(l1[:-1] + "\n\n   ...\n\n  " + l1[-1] + "\n\n"
          + " " + l2[:-2] + "\n\n   ...\n\n  " + l2[-2] + "\n\n...\n\n" + l2[-1])


# %% [markdown]
# ### Ingredients
#
# - [x] Function which, from a rank $r$, construct a list of permutation classes which partition the set of indices (each index belongs to exactly one class).
#   $r=3 \to \{iii, iij, ijk\}$
# - [x] Function which infers the permutation class from a tuple of actual indices.
#   + `SymmetricTensor.get_perm_class`
# - [ ] Efficient formula to compute, given a multidimensional index $I$ and its permutation class $\hat{σ}$, which is the corresponding position in that class' data vector.
#   + At present this is done by iterating all tensor indices in a class until we find a match.
#     Is it possible to do better ? (Considering that things like `factorial` can be pretty expensive, I'm not sure where the break-even point would even be between iterating and computing the index directly.)
# - [x] Efficient iterators:
#   + `indep_iter` & `index_iter` return each independent component once.
#   + `index_iter` returns indices compatible with advanced indexing, for more efficient operations on the dense tensor. (Loops over symmetric-equivalent indices are done in C.)
# - [ ] Standard `__iter__` iterator
#   + Should return symmetric sub-arrays of rank $r-1$, like `__iter__` on a dense array would.
# - [x] Per-class index iterator:
#   + `indep_iter` & `index_iter` accept an option permutation class argument.
# - [x] `tensor['iij']` returns the data vector for permutation class `'iij'`.
# - [x] Assignment to both classes or individual components
# - [ ] Assignment of callbacks (both with and without arguments) instead of values:
#   + Argument-less callbacks only as class defaults.
#   + Argument callbacks should take an index tuple as argument
#   + So that they have the same interface as arrays, callbacks are wrapped with an object which maps `__getitem__` to their arguments. Same as we do in `CumulantCollection`.
# - [x] Formula for the size of a data vector to allocate for each permutation class: Given a permutation class, how many independent tensor components does it represent.
#   + $s_{\hat{σ}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!}$, where $l$ is the number of different indices in the class and $m_n$ is the number of different indices which appear $n$ times.
#   + `SymmetricTensor.get_class_size`
# - [x] Formula for the combinatorial factor (*multiplicity*) of each class: Given an index, how many distinct tensor components are associated to that index by symmetry.
#   + $γ_{\hat{σ}} = \binom{r}{m_1,m_2,\dotsc,m_r} = \frac{r!}{m_1!m_2!\dotsb m_l!}$, where $l$ is the number of different indices in the class and $m_n$ is the number of times index $n$ appears.
#   + `SymmetricTensor.get_class_multiplicity`
# - Identities ($\hat{σ}$ denotes a permutation class):
#   + $\sum_{\hat{σ}} s_{\hat{σ}} γ_{\hat{σ}} = d^r$
#   + $\sum_{\hat{σ}} s_{\hat{σ}} = \binom{d + r - 1}{r}$

# %% [markdown]
# ## Implementation

# %% [markdown]
# > **Note**: There is a bijective map between string representations of permutation classes – `'iijk'` – and count representations – `(2,1,1)`. Public methods of `SymmetricTensor` use strings, while private methods use counts.

# %% [markdown]
# ### `_indexcounts`
# Helper function for `SymmetricTensor.perm_classes`.
# Returns iterator, which generates all unique sets of index counts in the string representation of a permutation class up to permutation. For example, the counts for class `'iij'` are represented as `[2, 1]`. The resulting output has the shape $[[c_1, \dots], [c_1', \dots], \dots]$. Each list can have variable length, depending on the number of unique indices in that set.
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
# > The particular case when `remaining_counts` = `max_count` = 0 is treated specially, to ensureit  returns `[[]]` (rather than `[[0]]`).

# %%
# NB: Index counts are also internally used as class labels
def _indexcounts(remaining_idcs, remaining_counts, max_count):
    """
    Return iterator, which generates all unique sets of index counts in the
    string representation of a permutation class up to permutation.
    """
    assert remaining_counts <= remaining_idcs * max_count
    if remaining_counts <= max_count:
        if remaining_counts == 0:
            # This can happen with a rank 0 tensor; in that case, remaining_idcs = remaining_counts = max_count = 0
            # In this case, return an empty iterator – there are no indices left
            yield []
        else:
            yield [remaining_counts]
    if remaining_idcs > 1:
        for c in range(min(remaining_counts-1, max_count), 0, -1):
            for subcounts in _indexcounts(
                  remaining_idcs-1, remaining_counts-c, max_count=c):
                yield [c] + subcounts


# %% [markdown]
# ### `multinom`
# Applies
# $$\binom{n}{k_1,k_2,\dotsc,k_m} = \binom{n}{k_1} \binom{n-k_1}{k_2} \dotsb \binom{n-\sum_{i<m} k_i}{k_m} \,.$$
# Each binomial term is computed with `math.comb`.
# The $k_i$ terms are first sorted, since $\binom{n}{k}$ is less computationally expensive when $k$ is close to 0 or $n$.

# %%
def multinom(n: int, klst: List[int]) -> int:
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


# %% [markdown]
# ### `index_iter`
#
# Given an permutation class and a dimension, `index_iter` produces a generator which yields all the *index classes* for that *permutation class*, in the order in which they would be stored as a flat vector.
# For example, if the class and dimension are `'iij'` and `3` respectively, then the generator would yield, in sequence:
#
# > `(0,0,1)`, `(0,0,2)`, `(1,1,0)`, `(1,1,2)`, `(2,2,0)`, `(2,2,1)`
#
# If instead we had class=`'iijj'` and dimension=`3`, then the indices would be
#
# > `(0,0,1,1)`, `(0,0,2,2)`, `(1,1,2,2)`
#
# Note that the index generator is aware of index equivalences due to symmetry, which is why in the latter example, `(1,1,0,0)` is not returned. In other words, only one representative from each index class is returned.
#
# The implementation iterates through index positions from left to right and exploits the fact that index class representatives are ordered from highest to lowest index multiplicity: if the multiplicities of the current ($j$) and previous ($i$) indices match, then they are permutable, and only values $j > i$ are returned. Thus, the logic for generating the indices $j$ at a single position looks as follows:
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHNbcHJldmlvdXMgaW5kZXg6IGk8YnI-Y3VycmVudCBpbmRleDogajxicj5tdWx0aXBsaWNpdHkgb2YgaTogbWk8YnI-bXVsdGlwbGljaXR5IG9mIGo6IG1qPGJyPmFzc2lnbmVkIGluZGljZXM6IEldICAtLT4gQ3ttaSA9PSBtaiA_fVxuICAgIEMgLS0-fHllc3wgRFtcInlpZWxkIGogPiBpLCBqIOKIiSBJXCJdXG4gICAgQyAtLT58bm98IEVbXCJ5aWVsZCBqIOKJoCBpLCBqIOKIiSBJXCJdIiwibWVybWFpZCI6e30sInVwZGF0ZUVkaXRvciI6ZmFsc2UsImF1dG9TeW5jIjp0cnVlLCJ1cGRhdGVEaWFncmFtIjpmYWxzZX0)](https://mermaid-js.github.io/mermaid-live-editor/edit/##eyJjb2RlIjoiZ3JhcGggTFJcbiAgICBpbnB1dHNbcHJldmlvdXMgaW5kZXg6IGk8YnI-Y3VycmVudCBpbmRleDogajxicj5tdWx0aXBsaWNpdHkgb2YgaTogbWk8YnI-bXVsdGlwbGljaXR5IG9mIGo6IG1qPGJyPmFzc2lnbmVkIGluZGljZXM6IEldICAtLT4gQ3ttaSA9PSBtaiA_fVxuICAgIEMgLS0-fHllc3wgRFtcInlpZWxkIGogPiBpLCBqIOKIiSBJXCJdXG4gICAgQyAtLT58bm98IEVbXCJ5aWVsZCBqIOKJoCBpLCBqIOKIiSBcIl0iLCJtZXJtYWlkIjoie30iLCJ1cGRhdGVFZGl0b3IiOmZhbHNlLCJhdXRvU3luYyI6dHJ1ZSwidXBkYXRlRGlhZ3JhbSI6ZmFsc2V9)
#
# > **NOTE** Each index returned is only one of possibly many equivalent permutations. The `SymmetricTensor.index_iter` method, in contrast to this function, combines all these permutations (i.e. all elements of the index class) into a single “[advanced index]”(https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing).

# %%
def _subindex_iter(mult: Tuple[int], dim: int, prev_i: int, assigned_i: Set[int]):
    """
    Implements logic in flowchart for one index position.
    :param:mult:   List of multiplicities, starting with the previous one.
    :param:dim:    Dimension along each axis
    :param:prev_i: Current index value at the previous position
    :param:assigned_i: All indices (excluding `prev_i`) which have already been
        assigned at another position.
    """
    m = mult[1]
    if mult[0] == mult[1]:
        if len(mult) == 2:
            for j in range(prev_i+1, dim):
                if j in assigned_i:
                    continue
                yield [j]*m
        else:
            for j in range(prev_i+1, dim):
                if j in assigned_i:
                    continue
                for subidx in _subindex_iter(mult[1:], dim, j, assigned_i|{prev_i}):
                    yield subidx + [j]*m  # Append is faster than prepend; we will reverse at end
    else:
        if len(mult) == 2:
            for j in range(dim):
                if j not in assigned_i|{prev_i}:
                    yield [j]*m
        else:
             for j in itertools.chain(range(0, prev_i), range(prev_i+1, dim)):
                if j in assigned_i:
                    continue
                for subidx in _subindex_iter(mult[1:], dim, j, assigned_i|{prev_i}):
                    yield subidx + [j]*m  # Append is faster than prepend; we will reverse at end


# %%
def index_iter(perm_class: Tuple[int], dim: int) -> Generator[Tuple[int]]:
    """
    Given a permutation class and a dimension, return a generator which yields all
    the indices for that class, in the order in which they would be stored as
    a flat vector.
    """
    rank = sum(perm_class)
    idx = []

    if len(perm_class) == 0:  # Rank-0 tensor
        # Single, scalar value; associated to empty tuple
        yield ()
    if len(perm_class) == 1:  # Diagonal terms
        for i in range(dim):
            yield (i,)*rank
    elif len(perm_class) > dim:
        # Cannot have more distinct indices than than there are dimensions
        # => return an empty iterator
        return
    else:
        for i in range(dim):
            m = perm_class[0]
            for subidx in _subindex_iter(perm_class, dim, i, set()):
                yield tuple(reversed(subidx + [i]*m))


# %% [markdown]
# ### Indexing utilities
#
# `_get_perm_class`: Given either a tuple of numerical indices (`(3, 0, 3)`) or string indices (`('j','i', 'j')`), return the tuple representing the permutation class it belongs to (`(2, 1)`).
# Note that a string index must be split into a tuple of single chars.

# %%
def _get_perm_class(key: Tuple[int,str]) -> Tuple[int]:
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


# %% [markdown]
# `_get_perm_class_size`: Return the number of independent components in the given permutation class.
# Computed as
# $$s_{\hat{I}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!} \,,$$
# where $r$ is the rank, $l$ is the number of different indices in the class and $m_n$ is the number of different indices which appear $n$ times in $I$. Examples:
#
# | $\hat{I}$ | $\hat{I}$ (str) | $l$ | $m$ |
# |-----------|-----------------|-----|-----|
# |`(3,2)`    | `iiijj`         | 2   | 0, 1, 1, 0, 0 |
# |`(1,1,1)`  | `ijk`           | 3   | 3, 0, 0 |

# %%
def _get_perm_class_size(perm_class: Tuple[int], dim: int) -> int:
    counts = perm_class
    rank = sum(counts)
    l = len(counts)
    permutable_groups = [counts.count(c) for c in range(1, rank+1)]
    res = (math.prod(range(dim, dim-l, -1))
           / math.prod(math.factorial(m) for m in permutable_groups))
    assert res.is_integer()
    return int(res)


# %% [markdown]
# `_get_index_representative`: Each set of indices equivalent under permutation has one representative index, given by grouping identical indices, then sorting first by repeat count than by index value. This function returns that representative.
#
# For example, given the input index `(2,1,2)`, it returns `(2,2,1)`.

# %%
def _get_index_representative(index: Tuple[int]) -> Tuple[int]:
    i_repeats = ((i, len(list(grouper))) for i, grouper in
                 itertools.groupby(sorted(index)))
    return sum(((i,)*repeats for i, repeats in
                sorted(i_repeats, key = lambda tup: tup[1], reverse=True)),
               start=())


# %% [markdown]
# `partition_list_into_two`:
#  Generate pairs of lists, with different lengths, length 1 and length 2, such that each pair contains all elements of the list. Order does not matter, but it treats the elements of the list as if they were all different. Thus a total of
#  $$ \binom{\text{length}}{\text{length } 1} $$
#  pairs is created.
#
#  Example:
#  [1,1,5] can be paired into pairs of lists with lengths 1,2 in the following ways:
#
# | length `1` |length `2`|
# |------------|----------|
# |`[1]`       | `[1,5]`  |
# |`[1]`       | `[5,1]`  |
# |`[5]`       | `[1,1]`  |

# %%
def partition_list_into_two(lst, size1, size2):
    '''
    return all pairs of sublists of lst, where the order of the elements in the sublists is ignored.
    '''
    assert size1 +size2 == len(lst), 'sizes must sum to length of lst'

    indices = range(len(lst))
    indices_1 = list(itertools.combinations(indices, size1))
    indices_2 = [list(set(indices).difference(set(idcs1))) for idcs1 in indices_1]
    lsts_1 = [[lst[i] for i in idcs1] for idcs1 in indices_1]
    lsts_2 = [[lst[i] for i in idcs2] for idcs2 in indices_2]
    return lsts_1, lsts_2


# %% [markdown]
# ### `SymmetricTensor`
#
# **Public attributes**
# - *dim*  → `int`
# - *dtype* → (e.g.) `dtype('float64')`
# - *rank* → `int`
# - *perm_classes* → (e.g.) `['iii', 'iij', 'ijk']`
# - *shape* → $(d,d,\dotsc)$
# - *size*  → `int` (total number of independent components)
# - *flat*       → Iterator over all components, including symmetric equivalents. See also *indep_iter*.
# - *flat_index* → Indices aligned with *flat*.
#
# **Public methods**
# - *get_perm_class(index: Tuple[int])*: From a specific index (`(0,0,1)`), return class string (`'iij'`).
# - *get_class_label(repeats: Tuple[int])*: Convert permutation class tuple (repeats) (`(2,1)`) to class string (e.g. `'iij'`).
# - *get_class_tuple(class_label: str)*: Convert permutation class string to class tuple (repeats).
# - *get_class_size(class_label: str)* : Number of independent components in a permutation class, i.e. the size of the storage vector.
# - *get_class_multiplicity(class_label: str)*: Number of times components in this permutation class are repeated in the full tensor.
# - *todense()*
# - *indep_iter(class_label: Optional[str])*     : Iterator over independent components (i.e. *excluding* symmetric equivalents).
# - *index_iter(class_label: Optional[str])*     : Indices aligned with *indep_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
#
# **Supported indexing**
# - `A['iij']` → 1D vector of values for this class.
# - `A[0,0,1]` → Scalar value corresponding to index `(0,0,1)`.
#
# **Supported assignment**
# - `A['iij'] = 3` – Assign the value `3` to all components in the permutation class `'iij'`.
# - `A['iij'] = [1, 2, 3, …]` – Assign different values to each component in the permutation class `'iij'`.
# - `A[0,1,2] = 3` – Assign the value `3` to the all components in the index class $\widehat{(0,1,2)}$.
#
# **Remarks**
#
# - Partial or sliced indexing is not currently supported.
#   This could certainly be done, although it would require some indexing gymnastics
# - Similarly, the `__iter__` method is not implemented.
#   To be consistent with a dense array, the produced iterator should yield
#   `SymmetricTensor` objects of rank `k-1` corresponding to partial indexing
#   along the first dimension.
# - Arithmetic operations are not currently supported.
#   Elementwise operations between tensors of the same size and rank can be trivially implemented when the need arises; other operations (like `.dot`) should be possible with some more work.

# %%
class SymmetricTensor(Serializable):
    """
    On creation, defaults to a zero tensor.
    """
    indices: ClassVar[str] = "ijklmnαβγδ"

    rank                 : int
    dim                  : int
    _dtype                : DType
    _data                : Dict[Tuple[int], Union[float, Array[float,1]]]
    _class_sizes         : Dict[Tuple[int], int]
    _class_multiplicities: Dict[Tuple[int], int]
        # NB: Internally we use the index counts instead of the equivalent
        # string to represent classes, since it is more useful for calculations

    def __init__(self, rank: int, dim: int,
                 data: Optional[Dict[Union[Tuple[int,...], str],
                                     Array[float,1]]]=None,
                 dtype: Union[None,str,DType]=None):
        self.rank = rank
        self.dim = dim
        if dtype is None:
            dtype = np.dtype('float64')
        self._dtype = dtype
        self._data = {tuple(repeats): dtype.type(0)
                      for repeats in _indexcounts(rank, rank, rank)}
        self._class_sizes = {tuple(repeats): _get_perm_class_size(repeats, self.dim)
                             for repeats in self._data}
        self._class_multiplicities = {tuple(repeats): multinom(self.rank, repeats)
                                      for repeats in self._data}
        if data:
            if isinstance(data, np.ndarray):
                raise NotImplementedError("Casting plain arrays to SymmetricTensor "
                                          "is not yet supported.")
                # TODO: Check that the array is symmetric, then extract values.
            elif not isinstance(data, dict):
                raise TypeError("If provided, `data` must be a dictionary with "
                                "the format {σ class: data vector}")
            # If `data` comes from serialized JSON; revert strings to tuples
            for key in list(data):
                if isinstance(key, str):
                    newkey = literal_eval(key)  # NB: That this is Tuple[int] is verified below
                    if newkey in data:
                        raise ValueError(f"`data` contains the key '{key}' "
                                         "twice: in both its original and "
                                         "serialized (str) form.")
                    data[newkey] = data[key]
                    del data[key]
            # Assert that the deserialized data has the right shape
            if data.keys() != self._data.keys():
                raise ValueError("`data` argument to SymmetricTensor does not "
                                 "have the expected format.\nExpected keys: "
                                 f"{sorted(self._data)}\nReceived keys:{sorted(data)}")
            for k, v in data.items():
                if isinstance(v, np.ndarray) and v.shape != (self._class_sizes[k],):
                    raise ValueError(f"Data for permutation class {self.get_class_label} "
                                     f"should have shape {(self._class_sizes[k],)}, "
                                     f"but the provided data has shape {v.shape}.")
            # Replace blank data with the provided data
            self._data = {k: v.astype(dtype) for k, v in data.items()}

    ## Pydantic serialization ##
    class Data(BaseModel):
        rank: int
        dim: int
        # NB: JSON keys must be str, int, float, bool or None, but not tuple => convert to str
        data: Dict[str, Union[float, Array[float,1]]]
        @classmethod
        def json_encoder(cls, symtensor: SymmetricTensor):
            return cls(rank=symtensor.rank, dim=symtensor.dim,
                       data={str(k): v for k,v in symtensor._data.items()})

    ## Translation functions ##
    # Mostly used internally, but part of the public API

    def copy(self):
        '''
        Return a copy of the current tensor
        '''
        return SymmetricTensor(dim = self.dim, rank = self.rank, data = self._data.copy())

    def is_equal(self, other, prec =None):
        '''
        Check current SymmetricTensor is equal to other.
        '''
        if not isinstance(other, SymmetricTensor):
            raise TypeError("Both tensors must be instances of SymmetricTensor for comparison")
        equal = True
        if prec is None:
            for idx in self.index_class_iter():
                if not self[idx] == other[idx]:
                    equal = False
                    break
        else:
            for idx in self.index_class_iter():
                if not (abs(self[idx]- other[idx])<prec).any():
                    equal = False
                    break
        return equal


    @classmethod
    def get_perm_class(cls, key: Tuple[int]) -> str:
        """
        Given an index tuple, return the string of the permutation class
        to which it belongs.

        Usage:
        >>> A = SymmetricTensor(rank=4, dim=6)
        >>> A.get_perm_class((5,0,1,0))
        'iijk'
        """
        σcls = ""
        for count, s in zip(_get_perm_class(key), cls.indices):
            σcls += s*count
        return σcls

    @classmethod
    def get_class_label(cls, index_repeats: Tuple[int]) -> str:
        """
        Convert an index from tuple (repeat) to string representation.
        """
        return ''.join((s*c for s,c in zip(cls.indices, index_repeats)))

    @classmethod
    def get_class_tuple(cls, class_label: str) -> Tuple[int]:
        """
        Convert an index from string to tuple (repeat) representation.
        """
        return tuple(sorted(
            (class_label.count(s) for s in set(class_label)),
            reverse=True))

    def get_index_repeats(self, class_label: str) -> List[int]:
        """
        Return the number of times each index is repeated.
        Always returns a list of length equal to rank, padding with zero
        if necessary.
        The sum of the returned values always equals the length of `class_label`

        Differences with `get_class_tuple`:
        - Raises `ValueError` if the length of the label doesn't match the
          tensor's rank.
        - `get_class_tuple` only returns non-zero repeats.
        """
        self._check_class_label(class_label)
        return [class_label.count(s) for s in self.indices[:self.rank]]

    def get_class_size(self, class_label: str) -> int:
        """
        Return the number of independent components in the permutation class
        associated to `class_label`.
        """
        self._check_class_label(class_label)
        return self._class_sizes[self.get_class_tuple(class_label)]

    def get_class_multiplicity(self, class_label: str) -> List[int]:
        """
        Return the number of times each component in the permutation class given
        by `class_label` is repeated in the full tensor.
        """
        self._check_class_label(class_label)
        return multinom(self.rank, self.get_index_repeats(class_label))

    ## Dunder methods ##

    def __repr__(self):
        s = f"SymmetricTensor(rank: {self.rank}, dim: {self.dim})"
        lines = [f"  {self.get_class_label(σcls)}: {value}"
                 for σcls, value in self._data.items()]
        return "\n".join((s, *lines)) + "\n"  # Lists of SymmetricTensors look better if each tensor starts on its own line

    def _convert_dense_index(self, key: Union[Tuple[int],int]
                            ) -> Tuple[Tuple[int,...], int]:
        """
        Given an index in dense format, return the class and position keys
        needed to index the matching value in `_data`.
        """
        if isinstance(key, tuple):
            # Use the standardized key so that it matches values returned by index_iter
            key = _get_index_representative(key)
            σcls = _get_perm_class(key)
            for i, idx in enumerate(index_iter(σcls, self.dim)):
                if key == idx:
                    return σcls, i
            if len(key) < self.rank:
                raise IndexError(
                    "Partial indexing (where the number of indices is less "
                    "than the rank) is inefficient with a SymmetricTensor and "
                    "not currently supported.")
            else:
                raise IndexError(f"{key} does not seem to be a valid array index, "
                               "or some of its values exceed the tensor dimensions.")
        elif isinstance(key, int):
            if self.rank == 1:
                assert len(self._data) == 1
                return (0,), key
            else:
                raise IndexError(
                    "Partial indexing (where the number of indices is less "
                    "than the rank) is inefficient with a SymmetricTensor and "
                    "not currently supported.")
        elif isinstance(key, slice):
            raise NotImplementedError(
                "Indexing with slices is not currently implemented. It is "
                "not trivial but could be done.")
        else:
            raise TypeError(f"Unrecognized index type '{type(key)}' (value: {key}).\n"
                            f"{type(self).__name__} only supports strings "
                            "and integer tuples as keys.")

    def __getitem__(self, key):
        """
        Two paths:
        - If `key` is as string (e.g. `'iij'`), return the data vector
          corresponding to that permutation class.
        - If `key` is a tuple, treat it as a tuple from a dense array and
          return the corresponding value.
        - If 'key' is an int, return a rank C of rank= self.rank -1 such that C.todense()[ = self.todense()[key,:,...,:]

        .. Note:: slices are not yet supported.
        """
        if isinstance(key, str):
            repeats = _get_perm_class(tuple(key))
            return self._data[repeats]

        elif isinstance(key, tuple):
            if any([isinstance(i,slice) for i in key]) or isinstance(key, slice):
                indices_fixed = [i for i in key if isinstance(i,int)]
                slices = [i for i in key if isinstance(i,slice)]
                #Check for subslicing
                subslicing = False
                for s in slices:
                    if any([x is not None for x in [s.start,s.step,s.stop]]):
                        subslicing = True
                        raise NotImplementedError("Indexing with subslicing (for example SymmetricTensor[1:3, 0,0]) is not"
                                                  " currently implemented. Only slices of the type"
                                                  "[i_1,...,i_n,:,...,:] with i_1,..., i_n all integers are allowed.")

                if not subslicing:
                    C = self.copy()
                    for i in indices_fixed:
                        C = C[i]
                    return C

            else:
                σcls, pos = self._convert_dense_index(key)
                vals = self._data[σcls]
                return vals if np.isscalar(vals) else vals[pos]
        elif isinstance(key, int):
            if self.dim ==1:
                σcls, pos = self._convert_dense_index(key)
                vals = self._data[σcls]
                return vals if np.isscalar(vals) else vals[pos]
            elif self.dim >1:
                B = SymmetricTensor(dim=self.dim, rank=self.rank-1)

                def σcls_subset_with_i(self, i, J, σcls):
                    '''
                    Let σcls be a perm. class of a Symmetrictensor of rank=self.rank-1 and dim = self.dim.
                    This function extracts all entries in the current tensor, which are compatible with the σcls+i, where i is an integer, and
                    the other entries come from J.
                    For example, if self.rank =3 i=0, and σcls='ij', and J=(0,1) this will extract all the entry with tensor index:
                    (0,0,1)
                    '''
                    σcls_data = self._data[self.get_class_tuple(σcls)]
                    if isinstance(σcls_data, (list,np.ndarray)):
                        return [σcls_data[k] for k, K in enumerate(self.index_class_iter(σcls))
                                                       if sorted((i, *J)) == sorted(K)]
                    else:
                        return [σcls_data for k, K in enumerate(self.index_class_iter(σcls))
                                                       if sorted((i, *J)) == sorted(K)]
                for σbcls in B.perm_classes:
                    B[σbcls] = [np.mean(np.fromiter(
                                    itertools.chain.from_iterable( (
                                        σcls_subset_with_i(self, key, J, σacls)
                                        for σacls in self.perm_classes if self.is_subσcls(σacls,σbcls) ) ),
                                    dtype=self.dtype))
                                # Use fancy indexing to retrieve multiple values simultaneously
                                # Averaging is done to symmetrize the array
                                for j, J in enumerate(B.index_class_iter(σbcls))]
                return B
        else:
            raise KeyError(f"{key}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
            repeats = _get_perm_class(tuple(key))
            if repeats not in self._data:
                raise KeyError(f"'{key}' does not match any permutation class.\n"
                               f"Permutation classes: {self.perm_classes}.")
            if np.isscalar(value):
                self._data[repeats] = value
            else:
                if len(value) != _get_perm_class_size(repeats, self.dim):
                    raise ValueError(
                        "Value must either be a scalar, or match the index "
                        f"class size.\nValue size: {len(value)}\n"
                        f"Permutation class size: {_get_perm_class_size(repeats, self.dim)}")
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                self._data[repeats] = value
        else:
            σcls, pos = self._convert_dense_index(key)
            v = self._data[σcls]
            if np.isscalar(v):
                if pos == slice(None):  # Equivalent to setting the whole permutation class
                    self._data[σcls] = value
                elif np.isscalar(value) and v == value:
                    # Value has not changed; no need to expand
                    pass
                else:
                    # Value is no longer uniform for all positions => need to expand storage from scalar to vector
                    v = v * np.ones(self._class_sizes[σcls], dtype=np.result_type(v, value))
                    v[pos] = value
                    self._data[σcls] = v
            else:
                self._data[σcls][pos] = value

    ## Numpy dispatch protocols – see NEP13, NEP18 ##

    def __array_function__(self, func, types, args, kwargs):
        # See "Implementation of the __array_function__ protocol" below
        # for construction of HANDLED_FUNCTIONS
        if func not in HANDLED_FUNCTIONS:
            return NotImplemented
        # NB: In contrast to the example in NEP18, we don't require
        #     arguments to be SymmetricTensors – ndarray is also allowed.
        if not all(issubclass(t, (SymmetricTensor, np.ndarray)) for t in types):
            return NotImplemented
        return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == "__call__":  # The "standard" ufunc, e.g. `multiply`, and not `multiply.outer`
            if ufunc in {np.add, np.multiply, np.divide, np.power}:  # Set of all ufuncs we want to support
                A, B = inputs   # FIXME: Check the shape of `inputs`. It might also be that we need `B = self` instead
                if isinstance(A, (int, float)):
                    C = self.__class__(dim=A.dim, rank=A.rank)
                    for σcls in C.perm_classes:
                        C[σcls] = ufunc(A, B[σcls])
                    return C
                elif isinstance(B, (int, float)):
                    C = self.__class__(dim=A.dim, rank=A.rank)
                    for σcls in C.perm_classes:
                        C[σcls] = ufunc(A[σcls], B)
                    return C
                elif A.dim != B.dim or A.rank != B.rank:
                    return NotImplemented
                else:
                    C = self.__class__(dim=A.dim, rank=A.rank)
                    for σcls in C.perm_classes:
                        C[σcls] = ufunc(A[σcls], B[σcls])  # This should always do the right whether, whether A[σcls] is a scalar or 1D array
                    return C
            elif ufunc in {np.exp, np.sin, np.cos, np.tan, np.cosh, np.sinh, np.tanh, np.sign, np.abs, np.sqrt, np.log}:  # Set of all ufuncs we want to support
                A, = inputs   # FIXME: Check the shape of `inputs`. It might also be that we need `B = self` instead
                C = self.__class__(dim=A.dim, rank=A.rank)
                for σcls in C.perm_classes:
                    C[σcls] = ufunc(A[σcls])  # This should always do the right whether, whether A[σcls] is a scalar or 1D array
                return C
        elif method == "outer":
            A, B = inputs
            return self.outer_product(B, ufunc = ufunc, **kwargs)
        else:
            return NotImplemented  # NB: This is different from `raise NotImplementedError`

    # Implementations of operations

    def __add__(self, other):
        return np.add(self, other)
    def __mul__(self, other):
        return np.multiply(self, other)
    def __sub__(self,other):
        C = other*(-1)
        return np.add(self, C)

    def outer_product(self, other, ufunc=np.multiply):  # SYMMETRIC ALGEBRA
        """
        Implement the outer product. Note that the outer product of two symmetric tensors is not symmetric.
        The result generated here is the symmetrized version of the outer product.
        """
        if isinstance(other, SymmetricTensor):
            if self.dim != other.dim:
                raise NotImplementedError("Currently only outer products between SymmetricTensors of the same dimension are supported.")
            else:
                C = SymmetricTensor(dim=self.dim, rank=self.rank+other.rank)
                for I in C.index_class_iter():
                    list1, list2 = partition_list_into_two(I, self.rank, other.rank)
                    C[I] = np.mean( [ufunc(self[tuple(idx1)], other[tuple(idx2)]) for idx1, idx2 in zip(list1,list2)] ).item()
                return C
        elif isinstance(other, list): 
            C = self.copy()
            for o in other:
                C = C.outer_product(o)
            return C
        elif not isinstance(other, (SymmetricTensor,list)): 
            raise TypeError( 'Argument must be SymmetricTensor or list of SymmetricTensors')

    def tensordot(self, other, axes=2):
        """
        like numpy.tensordot, but outputs are all symmetrized.
        """
        if not isinstance(other, SymmetricTensor):
            raise NotImplementedError("Currently only tensor products between SymmetricTensors are supported.")
        if self.dim != other.dim:
            raise NotImplementedError("Currently only tensor products between SymmetricTensors of the same dimension are supported.")
        if isinstance(axes,int):
            if axes ==0:
                return self.outer_product(other)
            elif axes ==1:
                # note \sum_i A_jkl..mi B_inop..z = \sum_i A_ijkl..m B_inop..z for A, B symmetric
                return np.sum( [self.__getitem__(i).outer_product(other[i]) for i in range(self.dim)])
            elif axes ==2:
                if self.rank < 2 or other.rank <2:
                    raise ValueError("Both tensors must have rank >=2")
                get_slice_index = lambda i,j,rank: (i,j,) +(slice(None,None,None),)*(rank-2)
                C = np.sum([np.multiply.outer(self[get_slice_index(i,j,self.rank)],
                                     other[get_slice_index(i,j,other.rank)]) for i in range(self.dim) for j in range(other.dim)])
                return C
        elif isinstance(axes, tuple):
            axes1 ,axes2 = axes
            if isinstance(axes1, tuple):
                assert isinstance(axes2, tuple), 'axes must be either int, tuple of length 2, or tuple of tuples'
                assert len(axes1) == len(axes2), '# dimensions to sum over must match'
                rank_deduct = len(axes1)
                get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-rank_deduct)
                C = np.sum([np.multiply.outer(self[get_slice_index(idx,self.rank)],
                                     other[get_slice_index(idx,other.rank)]) for idx in itertools.product(range(self.dim),repeat = rank_deduct)])
                return C
            elif isinstance(axes1,int):
                assert isinstance(axes2,int),  'axes must be either int, tuple of length 2, or tuple of tuples'
                return self.tensordot(other, axes =1)
        else:
            raise NotImplementedError("Tensordot with more axes than two is currently not implemented.")
            
    def contract_all_indices(self,W): 
        '''
        compute the contraction over all indices with a non-symmetric matrix, e.g. 
        
        C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}
        
        if current tensor has rank 3.
        '''
        
        C = SymmetricTensor(rank = self.rank, dim = self.dim)
        
        def _index_perm_prod_sum(W, idx_fixed, idx_permute):
            '''
            For index_fixed = (j_1, ... j_r)
            \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
            
            
            where σ(idx_permute) are all unique permutations. 
            '''
            idx_repeats = _get_perm_class(idx_permute) # number of repeats of indices
            permutations_of_identical_idx = np.prod([math.factorial(r) for r in idx_repeats])
            return np.sum( [np.prod([W[i,j] for i,j in zip(σidx,idx_fixed)])
                          for σidx in itertools.permutations(idx_permute)] ) /permutations_of_identical_idx
            
        for σcls in self.perm_classes:
            C[σcls] = [ np.sum([_index_perm_prod_sum(W, idx_fixed, idx_permute)*self[idx_permute] 
                      for idx_permute in self.index_class_iter()]) for idx_fixed in self.index_class_iter(class_label= σcls) ]
            
        return C
    
    def contract_tensor_list(self, tensor_list, n_times =1): 
        '''
        Do the following contraction: 
        
        out_{i_1,i_2,..., i_(r-n_times), j_1, j_2, ...j_m, k_1, k_2, ... k_m, ...} 
        = Symmetrize( \sum_{i_{r-n_times+1}, ..., i_r} outer( self_{i_1,i_2,.. i_r}, tensor_list[i_{r-n_times+1}]_{j_1,j_2,...j_m}, 
        
        Important: The tensors in tensor_list must be symmetric. 
        This is essentially a way to do a contraction between a symmetric and quasi_symmetric tensor \chi. Let 
        
        \chi_{i,j_1,j_2,...,j_m} = tensor_list[i]_{j_1,j_2,...j_m}
        
        Then even if \chi is not symmetric under exchange of the first indices with the rest, but the subtensors \chi_i,... 
        for fixed i are, we can do a contraction along the first index. 
        '''
        
        assert n_times <= self.rank, f"n_times is {n_times}, but cannot do more contractions than {self.rank} with tensor of rank {self.rank}"
        for list_entry in tensor_list: 
            assert isinstance(list_entry, SymmetricTensor), "tensor:list entries must be instances of SymmetricTensor"
        
        get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-n_times)
        indices = itertools.product(range(self.dim), repeat = n_times)
        chi_rank = tensor_list[0].rank
        C = SymmetricTensor(dim = self.dim, rank = self.rank +(chi_rank-1)*n_times) #one dimension used for contraction
        for idx in indices: 
            slice_idx = get_slice_index(idx, self.rank)
            C += self[slice_idx].outer_product([tensor_list[i] for i in idx])
        return C

    ## Array creation, copy, etc. ##

    def __array__(self):
        warn(f"Converting a SymmetricTensor to a dense NumPy array of shape {self.shape}.")
        return self.todense()

    def asarray(self, dtype=None, order=None):
        new_data = {k: v.asarray(dtype, order) if isinstance(v, np.ndarray)
                       else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                       else v.astype(dtype)
                    for k, v in self._data.items()}
        if all(self._data[k] is new_data[k] for k in new_data):
            # We received copy=False & all sub arrays were successfully not copied
            return self
        else:
            return SymmetricTensor(self.rank, self.dim, new_data)

    def asanyarray(self, dtype=None, order=None):
        new_data = {k: v.asanyarray(dtype, order) if isinstance(v, np.ndarray)
                       else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                       else v.astype(dtype)
                    for k, v in self._data.items()}
        if all(self._data[k] is new_data[k] for k in new_data):
            # We received copy=False & all sub arrays were successfully not copied
            return self
        else:
            return SymmetricTensor(self.rank, self.dim, new_data)

    def astype(self, dtype, order, casting, subok=True, copy=True):
        new_data = {k: v.astype(dtype, order, casting, subok, copy) if isinstance(v, np.ndarray)
                       else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                       else v.astype(dtype)
                    for k, v in self._data.items()}
        if all(self._data[k] is new_data[k] for k in new_data):
            # We received copy=False & all sub arrays were successfully not copied
            return self
        else:
            return SymmetricTensor(self.rank, self.dim, new_data)

    ## Public attributes & API ##

    @property
    def perm_classes(self) -> List[str]:
        """
        Return all permutation class strings as a list.
        """
        return [self.get_class_label(σcls) for σcls in self._data.keys()]

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int,...]:
        return (self.dim,)*self.rank

    @property
    def size(self) -> int:
        """
        Return the number of independent components in the tensor.
        This may be more than the number of stored values, if some values
        are stored as scalars.
        """
        return sum(self._class_sizes.values())

    def todense(self) -> Array:
        A = np.empty(self.shape, self.dtype)
        for idx, value in zip(self.index_iter(), self.indep_iter()):
            A[idx] = value
        return A

    @classmethod
    def is_subσcls(self, σcls: str, subσcls: str) -> bool:  # Could be considered public
        '''
        Check if subσcls is a sub permutation class of σcls.
        Both σcls and subσcls are strings where the entries denote the number of repetitions of a single index.
        For example 'iiij' is the permutation class of indices of a fourth order tensor where all indices are equal but one.
        Then 'iij' is a subσcls of σcls,
        but 'ijk' is not a subσcls of 'iiij', because for 'ijk', there have to be at least three different indices.
        '''
        return self._is_subσcls(self.get_class_tuple(σcls),
                           self.get_class_tuple(subσcls))

    ## Iterators ##

    @property
    def flat(self):
        """
        Return an iterator which yields each independent component *once*.
        Can be zipped with `flat_index` to get one (of the generally
        multiple) associated indices in the symmetric tensor.

        .. Note:: At present, in contrast to NumPy's `flat`, it is not possible
           to set values with this iterator (since it is an iterator rather
           than a view).
        """
        for v, size, mult in zip(self._data.values(),
                                 self._class_sizes.values(),
                                 self._class_multiplicities.values()):
            if np.isscalar(v):
                yield from itertools.repeat(v, size*mult)
            else:
                for vi in v:
                    yield from itertools.repeat(vi, mult)

    @property
    def flat_index(self):
        """
        Return an iterator which yields the index of each tensor component
        exactly once.
        Can be zipped with `flat` to also get the component values.

        .. Note:: For looping over an entire dense tensor, the advanced index
           returned by `index_iter` should in general be more efficient than
           this one.
        """
        for repeats in self._data:
            for index in index_iter(repeats, self.dim):
                # TODO? Is there a better way than `set` to keep only unique permutations ?
                yield from sorted(set(itertools.permutations(index)))

    def __iter__(self):
        raise NotImplementedError("Standard iteration, as with a dense tensor, "
                                  "would require extra work but could be supported.")

    def indep_iter(self, class_label: str=None) -> Generator:
        """
        Return a generator which yields values for the independent components
        in the class associated to `class_label`, in the order in which they
        are stored as a flat vector.
        Values stored as a scalar (when all components in a permutation class
        have the same value) are returned multiple times, as many as the size
        of that class. The output thus does not depend on whether values
        are stored as scalars or arrays.

        Parameters
        ---------
        class_label: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: Can be combined with `index_iter`, `index_class_iter` and
           `mult_iter`.
        """
        if class_label is None:
            for v, size in zip(self._data.values(), self._class_sizes.values()):
                if np.isscalar(v):
                    yield from itertools.repeat(v, size)
                else:
                    yield from v
        else:
            self._check_class_label(class_label)
            repeats = self.get_class_tuple(class_label)
            v = self._data[repeats]
            if np.isscalar(v):
                size = self._class_sizes[repeats]
                yield from itertools.repeat(v, size)
            else:
                yield from v

    def index_iter(self, class_label: str=None
                  ) -> Generator[Tuple[List[int],...]]:
        """
        Return a generator which yields all the indices in the class associated
        to `class_label`, in the order in which they are stored as a flat vector.
        Equivalent permutations are returned together, as a single “advanced
        index” tuple, such that they can be used to simultaneously get or set values
        in a dense tensor.

        Parameters
        ---------
        class_label: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: To construct the advanced index, permuted indices are
           collated. So for example, the index `(0,1,2)` has six permutations,
           which are collated as
           `([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [2, 1, 2, 0, 1, 0])`
           (For more information on the integer array indexing, see section in the
           `NumPy docs <https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing>`_.)
        """
        if class_label is None:
            for label in self.perm_classes:
                yield from self.index_iter(label)
        else:
            self._check_class_label(class_label)
            repeats = self.get_class_tuple(class_label)
            for index in index_iter(repeats, self.dim):
                # NB: dict.fromkeys is used to keep only unique permutations
                yield tuple(list(σindex) for σindex in  # Convert from tuple to list to trigger advanced indexing
                            zip(*dict.fromkeys(itertools.permutations(index)).keys()))

    def index_class_iter(self, class_label: str=None
                  ) -> Generator[Tuple[int]]:
        """
        Return a generator which yields one representative for each index class
        *I* in the permutation class given by `class_label`, in the order in
        which they are stored as a flat vector.

        Parameters
        ---------
        class_label: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: In contrast to `index_iter`, this does not return all
           equivalent permutations of a index. In general, `index_class_iter`
           is more suited to operations involving only symmetric tensors,
           while `index_iter` is more suited to operations involving also
           dense tensors.
        """
        if class_label is None:
            for label in self.perm_classes:
                yield from self.index_class_iter(label)
        else:
            self._check_class_label(class_label)
            repeats = self.get_class_tuple(class_label)
            yield from index_iter(repeats, self.dim)

    def mult_iter(self):
        """
        Return the multiplicity of each class.
        In contrast to simply iterating over `self.perm_classes` and calling
        `get_class_multiplicity`, multiplicities are returned per *index*
        class (i.e., they line up with independent components).
        This makes this iterator appropriate for zipping with `indep_iter`
        and `index_iter`.
        """
        for σcls in self.perm_classes:
            γ = self.get_class_multiplicity(σcls)
            s = self.get_class_size(σcls)
            yield from itertools.repeat(γ, s)

    ## Private utilities

    def _check_class_label(self, class_label: str) -> None:
        """
        Raises `ValueError` if the `class_label` has a different length as
        the rank.
        """
        if len(class_label) != self.rank:
            raise ValueError(f"A tensor of rank {self.rank} expects a class "
                             f"label with {self.rank} indices. Received the "
                             f"label '{class_label}'.")

    @staticmethod
    def _is_subσcls(σcls: Tuple[int], subσcls: Tuple[int]) -> bool:
        '''
        Check if subσcls is a sub permutation class of σcls.
        Both σcls and subσcls are tuples where the entries denote the number of repetitions of a single index.
        For example σcls = (3,1) corresponds to ther σcls string 'iiij'. Then (2,1) <-> 'iij' is a subσcls of σcls,
        but (1,1,1) <-> 'ijk' is not a subσcls of 'iiij' <-> (3,1).
        '''
        return len(σcls) >= len(subσcls) and all(a >= b for a, b in zip(σcls, subσcls))


# %%
if __name__=="__main__": 
    dim = 4
    for dim in [2,3,4,5]: #not tpo high dimensionality, because dense tensor operations
        test_tensor = SymmetricTensor(rank =3, dim = dim)
        test_tensor['iii'] = np.random.rand(dim)
        test_tensor['ijk'] = np.random.rand(int(dim*(dim-1)*(dim-2)/6))
        test_tensor['iij'] = np.random.rand(int(dim*(dim-1)))
        
        tensor_list = []
        chi_dense = np.zeros( (dim,)*3)
        def get_random_symtensor_rank2(dim): 
            tensor = SymmetricTensor(rank=2, dim =dim)
            tensor['ii'] = np.random.rand(dim)
            tensor['ij'] = np.random.rand(int((dim**2 -dim)/2))
            return tensor
        for i in range(dim): 
            random_tensor = get_random_symtensor_rank2(dim)
            tensor_list += [random_tensor]
            chi_dense[i,:,:] = random_tensor.todense()

        contract_1 = test_tensor.contract_tensor_list( tensor_list, n_times =1)
        contract_2 = test_tensor.contract_tensor_list( tensor_list, n_times =2)

        print( (contract_1.todense() == np.einsum('ija, akl -> ijkl', test_tensor.todense(), chi_dense)).all())
        #assert (contract_2.todense() == np.einsum('iab, ajk, blm  -> ijklm', test_tensor.todense(), chi_dense, chi_dense)).any()
        #print(contract_1.todense()- np.einsum('ija, akl -> ijkl', test_tensor.todense(), chi_dense))

# %% [markdown]
# ### Implementation of the `__array_function__` protocol
#
# Support for *non*-universal functions should be added here, following the pattern provided in [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html#example-for-a-project-implementing-the-numpy-api).
# Universal functions (`np.add`, `np.exp`, etc.) [should use](https://numpy.org/neps/nep-0018-array-function-protocol.html#specialized-protocols) the more specialized `__array_ufunc__` protocol defined in [NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html).
#
# Additional references/remarks:
# - There has been a lot of discussion regarding dispatching mechanisms for NumPy duck arrays – see [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html), [NEP 22](https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html), [NEP 30](https://numpy.org/neps/nep-0030-duck-array-protocol.html), [NEP 31](https://numpy.org/neps/nep-0031-uarray.html), [NEP 35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html), [NEP 47](https://numpy.org/neps/nep-0047-array-api-standard.html). Of these, only NEP 18 and NEP 35 have actually been adopted; NEP 47 seems to be where this will go in the future, but it's likely to be a few years still before this becomes implemented.
# - `tensordot` is often used as a motivating example in these cases, so whenever this matures, it likely will address the use cases we have here.
# - There is already an [open issue](https://github.com/numpy/numpy/issues/11506) for supporting `einsum_path` with non-NumPy arrays on NumPy's GitHub.
#
# **Implemented array functions**
#
# Implementation allows the standard numpy functions to work as expected, e.g. `np.tensordot(A, B)` where `A`, `B` are `SymmetricTensors` will work.
#
# - `asarray()`
# - `asanyarray()`
# - `tensordot()`
# - `einsum_path()`
# - `einsum()` [**TODO**]

# %% [markdown]
# HANDLED_FUNCTIONS = {}
#
# def implements(numpy_function):
#     """Register an __array_function__ implementation for SymmetricTensor objects."""
#     def decorator(func):
#         HANDLED_FUNCTIONS[numpy_function] = func
#         return func
#     return decorator
#
# @implements(np.asarray)
# def asarray(a, dtype=None, order=None):
#     return a.asarray(dtype, order=order)
#
# @implements(np.asanyarray)
# def asanyarray(a, dtype=None, order=None):
#     return a.asanyarray(dtype, order=order)
#
# @implements(np.tensordot)
# def tensordot(a, b, axes=2):
#     return a.tensordot(b, axes=2)
#
# @implements(np.einsum_path)
# def einsum_path(*operands, optimize='greedy', einsum_call=False):
#     with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#         return np.core.einsumfunc.einsum_path.__wrapped__(
#             *operands, optimize=optimize, einsum_call=einsum_call)
#
# # TODO
# #@implements(np.einsum)
# #def einsum(*operands, dtype=None, order='K', casting='safe', optimize=False):
# #    # NB: Can't used the implementation in np.core.einsumfunc, because that calls
# #    #     C code which requires true arrays
# #    ...

# %% [markdown]
# ### Bypassing coercion to ndarray
#
# (Note: the `make_array_like` context manager is not specific to `SymmetricTensor`, and could be moved to a *utils* module*.)
#
# A lot of the provided NumPy functions use `asarray` or `asanyarray` to ensure their inputs are array-like (and not, say, a list). Unfortunately this also coerces inputs into NumPy arrays, which we absolutely want to avoid with `SymmetricTensor`. The problem as that these functions are used for two different purposes:
# - When it is required that arguments truly be NumPy arrays;
# - When it is required that arguments be array-like, and implement parts of the NumPy array API (so-called “duck arrays”).
# The current solution is to use the keyword `like` (see [NEP 35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html)) when a duck array will suffice; this has the added benefit of being explicit about which duck array type is expected (since different types implement different subsets of the array API).
#
# This however doesn't address the use of `asarray` within NumPy functions, where it is mostly called without the `like` keyword. (Which makes sense – to use the keyword, the function would have to know the duck type desired by the user.) As a workaround, we provide below the `make_array_like` context manager: within the context, and only for the specified modules, the functions `asarray` and `asanyarray` are modified to inject a specified type as the default value for the `like` keyword. For example, within `np.einsum_path`, `asanyarray` is called on the array inputs. To ensure that this call returns a `SymmetricTensor` rather than an `ndarray`, we first determine the module in which this call is made (in this case, *np.core.einsumfunc*). Then
#
# ```python
# A = SymmetricTensor(3,4)
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
# Note: Rather than requiring the user to wrap calls with `make_array_like`, a better approach is to include those calls in the type-specific dispatch code, so that `np.einsum_path` always works as expected. See the implementation of `einsum_path` in *statGLOW/stats/symmetric_tensor.py* for an example.

# %% [markdown]
# from collections.abc import Iterable
# from contextlib import contextmanager
# from numpy.core import numeric as _numeric
# _make_array_like_patched_modules = set()  # Used in case of nested contexts
# @contextmanager
# def make_array_like(like, modules=()):  # UTILS
#     """
#     Monkey patch NumPy so that the type of `like` is recognized as an array.
#     Within this context, and within `module`, the default signature of `asarray(x)`
#     and `asanyarray(x)` is changed to include `like=like` (instead of `like=None`).
#
#     .. Note:: Like must be an *instance* (not a type), and must implement the
#        __array_function__ protocol. See NEP35 and NEP18.
#
#     .. Caution:: This as hack of the ugliest kind. Please use sparingly, and
#        only when no better solution is available.
#     """
#     if isinstance(modules, Iterable):
#         modules = set(modules)
#     else:
#         modules = {modules}
#     #if any(mod is np for mod in modules):
#     #    raise ValueError("`make_array_like` doesn't support overriding "
#     #                     "methods in the base 'numpy' module.")
#     # Open context: Monkey-patch Numpy function
#     def asarray(a, dtype=None, order=None, *, like=like):
#         if isinstance(a, type(like)):  # Without this, will break on normal arrays
#             return _numeric.asanyarray(a, dtype, order, like=like)
#         else:
#             return _numeric.asanyarray(a, dtype, order)
#     def asanyarray(a, dtype=None, order=None, *, like=like):
#         if isinstance(a, type(like)): # Without this, will break on normal arrays
#             return _numeric.asanyarray(a, dtype, order, like=like)
#         else:
#             return _numeric.asanyarray(a, dtype, order)
#     new_funcs = {'asarray': asarray,
#                  'asanyarray': asanyarray}
#     old_funcs = {'asarray': _numeric.asarray,
#                  'asanyarray': _numeric.asanyarray}
#     # NB: Because most NumPy modules alias these functions when they use them,
#     #     it's not sufficient to redefine np.asarray in _numeric: we need to
#     #     replace the aliases in the modules.
#     #     (assumption: aliases use the same function name)
#     for mod in modules:
#         if mod in _make_array_like_patched_modules:
#             # Already patched by an outer context
#             modules.remove(mod)
#             continue
#         for nm, f in new_funcs.items():
#             if nm in mod.__dict__:
#                 #import pdb; pdb.set_trace()
#                 setattr(mod, nm, f)
#     # Return control to code inside context
#     try:
#         yield None
#     # Close context: Undo the monkey patching
#     except Exception:
#         raise
#     finally:
#         for mod in modules:
#             for nm in old_funcs:
#                 # Iterating over .items() for some reason doesn't return the right values
#                 if nm in mod.__dict__:
#                     setattr(mod, nm, old_funcs[nm])

# %% [markdown]
# ## Memory footprint
# Memory required to store tensors, as a percentage of an equivalent tensor with dense storage.

# %%
if __name__ == "__main__":
    import sys
    curves = []
    for rank in [1, 2, 3, 4, 5, 6]:
        points = []
        for dim in [1, 2, 3, 4, 6, 8, 10, 20, 40, 80, 100]:
            # Dense size (measured in bytes; assume 64-bit values)
            dense = dim**rank*8 + 104  # 104: fixed overhead for arrays
            # SymmetricTensor size
            A = SymmetricTensor(rank=rank, dim=dim)
            # For dicts, getsizeof oly counts the number of keys/value pairs,
            # not the size of either of them.
            sym = sys.getsizeof(A)
            sym += sum(sys.getsizeof(k) + sys.getsizeof(v) for k,v in A.__dict__.items())
            sym += sum(sys.getsizeof(k) for k in A._data)
            sym += sum(A.get_class_size(σcls)*8 + 104 for σcls in A.perm_classes)
            # Append data point
            points.append((dim, sym/dense))
        curves.append(hv.Curve(points, kdims=['dimensions'], vdims=['size (rel. to dense)'],
                               label=f"rank = {rank}"))
        if rank > 4:
            curves[-1].opts(muted=True)
    fig = hv.Overlay(curves) * hv.HLine(1)
    fig.opts(hv.opts.HLine(color='grey', line_dash='dashed', line_width=2, alpha=0.5),
             hv.opts.Curve(width=500, logy=True, logx=True),
             hv.opts.Overlay(legend_position='right'))
    #display(fig)

# %% [markdown]
# ## Tests

# %%
if __name__ == "__main__":
    import pytest
    from statGLOW.utils import does_not_warn
    from collections import Counter
    def test_tensors() -> Generator:
        for d, r in itertools.product([2, 3, 4, 6, 8], [2, 3, 4, 5, 6]):
            yield SymmetricTensor(rank=r, dim=d)
    assert SymmetricTensor(rank=4, dim=3).perm_classes == \
        ['iiii', 'iiij', 'iijj', 'iijk', 'ijkl']

# %% [markdown]
# ### Combinatorics
#
# The tests below confirm that the permutation classes form a partition of tensor indices: the sum of their class sizes equals the total number of independent components in a symmetric tensor, which is given by
# $$\binom{d + r - 1}{r}\,.$$
# This is a well-known expression; it can be found for example [here](http://www.physics.mcgill.ca/~yangob/symmetric%20tensor.pdf).
# This test gives us good confidence that the methods `perm_classes` and `get_class_size` are correctly implemented.

    # %%
    for A in test_tensors():
        r = A.rank
        d = A.dim
        assert (sum(A.get_class_size(σcls) for σcls in A.perm_classes)
                == math.prod(range(d, d+r)) / math.factorial(r))

# %% [markdown]
# We now check if class multiplicities are correctly evaluated, to validate `get_class_multiplicity`. We do this by checking the identity
# $$\sum_\hat{σ} s_\hat{σ} = d^r\,.$$
# (Here $\hat{σ}$ is a permutation class and $s_{\hat{σ}}$ the size of that class.)

    # %%
    for A in test_tensors():
        r = A.rank
        d = A.dim
        assert (sum(A.get_class_size(σcls) * A.get_class_multiplicity(σcls)
                    for σcls in A.perm_classes)
                == d**r)

# %% [markdown]
# ### Iteration
#
# Test the index iterator, including examples given in the description of `index_iter`.

# %%
if __name__ == "__main__":
    assert list(index_iter((3,), 3))  == [(0,0,0), (1,1,1), (2,2,2)]
    assert list(index_iter((2,1), 2)) == [(0,0,1), (1,1, 0)]
    assert list(index_iter((2,1), 3)) == [(0,0,1), (0,0,2), (1,1,0), (1,1,2), (2,2,0), (2,2,1)]
    assert list(index_iter((2,2), 3)) == [(0,0,1,1), (0,0,2,2), (1,1,2,2)]

# %% [markdown]
# Test iteration.
#
# - `SymmetricTensor` gets initialized as a zero tensor, storing only one scalar per class.
# - Iteration still returns either $\binom{d + r - 1}{r}\,.$ or $d^r$ values (depending on whether it returns permutations of symmetric terms).
# - The `flat` iterators start by returning all diagonal components.
# - `index_class_iter` returns a unique index tuple (index class representative)
# - `index_class_iter` returns exactly one index $I$ for each index class, and each $I$ cannot be obtained from another via permutation.

    # %%
    for A in test_tensors():
        assert all(len(list(A.index_iter(σcls))) == len(list(A.indep_iter(σcls)))
                   for σcls in A.perm_classes)
        assert len(list(A.index_iter())) == len(list(A.indep_iter())) == A.size
        assert len(list(A.flat)) == len(list(A.flat_index)) == A.dim**A.rank
        # index_class_iter returns a unique index tuple
        I = next(A.index_class_iter())
        assert isinstance(I, tuple) and all(isinstance(i, int) for i in I)
        # index_class_iter returns a unique index I for each index class
        len({str(Counter(sorted(idx))) for idx in A.index_class_iter()}) == A.size

# %% [markdown]
# ### Indexing
#
# Test permutation standardization (computation of class representatives).

    # %%
    assert _get_index_representative((2,1,2))         == (2,2,1)
    assert _get_index_representative((1,1,2))         == (1,1,2)
    assert _get_index_representative((0,0))           == (0,0)
    assert _get_index_representative((5,4,3,3,2,1))   == (3,3,1,2,4,5)

# %% [markdown]
# Test indexing.

    # %%
    A = SymmetricTensor(3, 5)
    assert A['iij'] is A._data[(2,1)]

    b = 0
    sizes = [A.get_class_size(σcls) for σcls in A.perm_classes]
    for σcls, size in zip(A.perm_classes, sizes):
        if σcls == "iii":
            # Test indexing with both scalar and list entries
            A[σcls] = 0
        else:
            A[σcls] = np.arange(b, b+size)
        b += size

    assert A[1, 1, 1] == A['iii']
    assert A[0, 0, 3] == A['iij'][2]   # Preceded by: (0,0,1),(0,0,2)
    assert A[2, 2, 3] == A['iij'][10]  # Preceded by: (0,0,1–4),(1,1,0),(1,1,2–4),(2,2,0–1)
    assert A[1, 2, 3] == A['ijk'][6]   # Preceded by: (0,1,2—4),(0,2,3–4),(0,3,4)



# %%
if __name__ == "main":
    #subtensor generation with 1 index
    for A in test_tensors():
        for i in range(A.dim):
            assert (A[i].todense() == A.todense()[i]).any()
    #subtensor generation with multiple indices
    dim = 4
    rank = 4
    #test is_equal
    diagonal = np.random.rand(dim)
    odiag1 = np.random.rand()
    odiag2 = np.random.rand()
    A = SymmetricTensor(rank = rank, dim =dim)
    A['iiii'] = diagonal
    A['iiij'] = odiag1
    A['iijj'] = odiag2

    assert (A[0,1,:,:].todense() == A.todense()[0,1,::]).any()
    assert (A[0,1,:,:]).is_equal(A[1,0,:,:])
    assert (A[0,1,1,:]).is_equal(A[1,1,0,:])
    assert all([A[0,0,0,:][i] == A[0,0,0,i] for i in range(dim)])


    # %%
    #outer product
    A = next(test_tensors())
    B = next(test_tensors())
    Ad = A.todense()
    Bd = B.todense()
    assert (np.multiply.outer(A,B).todense() == np.multiply.outer(Ad,Bd)).any()

    # %%
    # Rank 3
    assert SymmetricTensor.is_subσcls('iii', 'ii')
    assert SymmetricTensor.is_subσcls('iij', 'ij')
    assert SymmetricTensor.is_subσcls('ijk', 'ij')
    assert not SymmetricTensor.is_subσcls('iii', 'ij')
    assert not SymmetricTensor.is_subσcls('ijk', 'ii')
    # Rank 4
    assert SymmetricTensor.is_subσcls('iiii', 'iii')
    assert SymmetricTensor.is_subσcls('iiii', 'ii')
    assert SymmetricTensor.is_subσcls('iijj', 'ij')
    assert SymmetricTensor.is_subσcls('iijj', 'iij')
    assert SymmetricTensor.is_subσcls('iijk', 'iij')
    assert SymmetricTensor.is_subσcls('iijk', 'ijk')
    assert SymmetricTensor.is_subσcls('iijk', 'ij')
    assert SymmetricTensor.is_subσcls('iijk', 'i')
    assert not SymmetricTensor.is_subσcls('iiii', 'ij')
    assert not SymmetricTensor.is_subσcls('iiii', 'ijk')
    assert not SymmetricTensor.is_subσcls('iijj', 'ijk')
    assert not SymmetricTensor.is_subσcls('iijk', 'ijkl')
    assert not SymmetricTensor.is_subσcls('iijk', 'iijj')
    assert not SymmetricTensor.is_subσcls('iijj', 'iii')

# %% [markdown]
# ### Assignement
#
# Test assignement: Assigning one value modifies all associated symmetric components.

    # %%
    A = SymmetricTensor(3, 3)
    A[1, 2, 0] = 1
    assert np.all(
        A.todense() ==
        np.array([[[0., 0., 0.],
                   [0., 0., 1.],
                   [0., 1., 0.]],

                  [[0., 0., 1.],
                   [0., 0., 0.],
                   [1., 0., 0.]],

                  [[0., 1., 0.],
                   [1., 0., 0.],
                   [0., 0., 0.]]])
    )

# %% [markdown]
# ### Serialization

# %% [markdown]
#     class Foo(BaseModel):
#         A: SymmetricTensor
#         class Config:
#             json_encoders = {Serializable: Serializable.json_encoder}
#     foo = Foo(A=A)
#     foo2 = Foo.parse_raw(foo.json())
#     assert foo2.json() == foo.json()

# %% [markdown]
# ### Avoiding array coercion
#
# `asarray` works as one would expect (converts to dense array by default, does not convert if `like` argument is used).

# %% [markdown]
# if __name__ == "__main__":
#     A = SymmetricTensor(rank=2, dim=3)
#     B = SymmetricTensor(rank=2, dim=3)
#     with pytest.warns(UserWarning):
#         assert type(np.asarray(A)) is np.ndarray
#     # `like` argument is supported and avoids the conversion to dense array
#     with does_not_warn(UserWarning):
#         assert type(np.asarray(A, like=SymmetricTensor(0,0))) is SymmetricTensor

# %% [markdown]
# Test that the `make_array_like` context manager correctly binds custom functions to `asarray`, and cleans up correctly on exit.

# %% [markdown]
#     # Context manager works as expected…
#     with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#         assert "<locals>" in str(np.core.einsumfunc.asanyarray)   # asanyarray has been substituted…
#         np.einsum('iij', np.arange(8).reshape(2,2,2))  # …and einsum still works
#         np.asarray(np.arange(3))                       # Plain asarray is untouched and still works
#     # …and returns the module to its clean state on exit…
#     assert "<locals>" not in str(np.core.einsumfunc.asanyarray)
#     with pytest.warns(UserWarning):
#         assert type(np.asarray(A)) is np.ndarray
#     # …even when an error is raised within the context.
#     try:
#         with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#             assert "<locals>" in str(np.core.einsumfunc.asanyarray)
#             raise ValueError
#     except ValueError:
#         pass
#     assert "<locals>" not in str(np.core.einsumfunc.asanyarray)

# %% [markdown]
# Test dispatched array functions which use the `make_array_like` decorator to avoid coercion.

# %% [markdown]
# with does_not_warn(UserWarning):
#     np.einsum_path("ij,ik", A, B)
#     np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))
#
# with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#     with does_not_warn(UserWarning):
#         np.einsum_path("ij,ik", A, B)
#         np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))

# %% [markdown]
# ### WIP
#
# *Ordering permutation classes.*
# At some point I thought I would need a scheme for ordering permutation classes (for implementing a hierarchy, where e.g. `'ijkl'` can be used as a default for `'iijk'`). I save it here in case it turns out to be useful after all.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCLigZ1cIl1cbiAgXG4gICAgc3R5bGUgTiBmaWxsOm5vbmUsIHN0cm9rZTpub25lIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/edit##eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCJcdOKBnVwiXVxuICBcbiAgICBzdHlsZSBOIGZpbGw6bm9uZSwgc3Ryb2tlOm5vbmUiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)

# %% [markdown]
# ## Arithmetic

# %%
if __name__ == "__main__":
    def transpose(A, axes):
        return np.transpose(A, axes)
    from itertools import permutations

    def symmetrize(dense_tensor):
        D = dense_tensor.ndim
        n = np.prod(range(1,D+1))  # Factorial – number of permutations
        return sum(transpose(dense_tensor, σaxes) for σaxes in permutations(range(D))) / n
    rank = 4
    dim = 2
    #test addition
    test_tensor_1 = SymmetricTensor(rank=rank, dim=dim)
    test_tensor_1['iiii'] = np.random.rand(2)
    test_tensor_2 = np.add(test_tensor_1,1.0)
    test_tensor_3 = SymmetricTensor(rank=rank, dim=dim)
    for σcls in test_tensor_3.perm_classes:
                test_tensor_3[σcls] = 1.0
    test_tensor_4 =  test_tensor_2 - test_tensor_3
    print(test_tensor_1, test_tensor_4)
    assert test_tensor_4.is_equal(test_tensor_1, prec =1e-10)
    test_tensor_5 = np.multiply(test_tensor_2, -1)
    test_tensor_6 = np.multiply(test_tensor_5, -1)
    #test multiplication
    assert test_tensor_6.is_equal(test_tensor_2, prec =1e-10)
    test_tensor_7 = np.exp(test_tensor_2)
    test_tensor_8 = np.log(test_tensor_7)
    #test log, exp
    assert test_tensor_8.is_equal(test_tensor_2, prec =1e-10)

# %% [markdown]
# ### Tensordot

# %%
if __name__ == "__main__":
    #outer product
    

    test_tensor_1d = test_tensor_1.todense()
    test_tensor_2d = test_tensor_2.todense()
    test_tensor_3d = test_tensor_3.todense()
    prec =1e-10
    test_tensor_8 = np.multiply.outer(test_tensor_2,test_tensor_3)
    assert (abs(test_tensor_8.todense()- symmetrize(np.multiply.outer(test_tensor_2d,test_tensor_3d)))<prec).all()
    test_tensor_9 = np.multiply.outer(test_tensor_1,test_tensor_3)
    assert (abs(test_tensor_9.todense() - symmetrize(np.multiply.outer(test_tensor_1d,test_tensor_3d)))<prec).all()

    test_tensor_10 = SymmetricTensor(rank=1, dim=2)
    test_tensor_10['i'] = [1,0]
    test_tensor_11 = SymmetricTensor(rank=1, dim=2)
    test_tensor_11['i'] = [0,1]
    test_tensor_12 = np.multiply.outer(test_tensor_10,test_tensor_11)
    assert test_tensor_12[0,0] ==0 and test_tensor_12[1,1] ==0
    assert test_tensor_12['ij'] == 0.5



    # %%
    #outer product with tensordot
    def test_tensordot(tensor_1, tensor_2, prec =1e-10):
        test_tensor_13 = tensor_1.tensordot(tensor_2, axes =0)
        assert test_tensor_13.is_equal(np.multiply.outer(tensor_1,tensor_2))

        #Contract over first and last indices:
        test_tensor_14 =  tensor_1.tensordot(tensor_2, axes =1)
        dense_tensor_14 = symmetrize(np.tensordot(tensor_1.todense(),
                                                  tensor_2.todense(),
                                                  axes =1 ))
        assert (abs(test_tensor_14.todense() - dense_tensor_14) <prec).any()
        test_tensor_141 =  tensor_1.tensordot(tensor_2, axes =(0,1))
        assert test_tensor_14.is_equal(test_tensor_141, prec = prec)

        #Contract over two first and last indices:
        test_tensor_15 =  tensor_1.tensordot(tensor_2, axes =2)
        dense_tensor_15 = symmetrize(np.tensordot(tensor_1.todense(),
                                                  tensor_2.todense(),
                                                  axes =2 ))
        if isinstance(test_tensor_15, SymmetricTensor):
            assert (abs(test_tensor_15.todense() - dense_tensor_15) <prec).all()
        else:
            assert test_tensor_15 == dense_tensor_15

        if tensor_1.rank >2 and tensor_2.rank >2:
            test_tensor_16 =  tensor_1.tensordot(tensor_2, axes =((0,1,2),(0,1,2)))
            dense_tensor_16 = symmetrize(np.tensordot(tensor_1.todense(),
                                                  tensor_2.todense(),
                                                  axes =((0,1,2),(0,1,2)) ))
            dense_tensor_161 = symmetrize(np.tensordot(tensor_1.todense(),
                                                  tensor_2.todense(),
                                                  axes =((0,1,2),(2,1,0)) ))
            dense_tensor_162 = symmetrize(np.tensordot(tensor_1.todense(),
                                                  tensor_2.todense(),
                                                  axes =((0,1,2),(2,0,1)) ))
            assert (abs(test_tensor_16.todense() - dense_tensor_16) <prec).all()
            assert (abs(test_tensor_16.todense() - dense_tensor_161) <prec).all()
            assert (abs(test_tensor_16.todense() - dense_tensor_162) <prec).all()

    for A in [test_tensor_1, test_tensor_2, test_tensor_3, test_tensor_4,test_tensor_5,test_tensor_6, test_tensor_7, test_tensor_8]:
        for B in [test_tensor_1, test_tensor_2, test_tensor_3, test_tensor_4,test_tensor_5,test_tensor_6, test_tensor_7, test_tensor_8]:
            if A.rank +B.rank <= 8: #otherwise we can't convert to dense
                test_tensordot(A,B)



# %% [markdown]
# ## Contraction with matrix along all indices

# %%
if __name__ == "__main__": 
    
    A = SymmetricTensor(rank = 3, dim=3)
    A[0,0,0] =1
    A[0,0,1] =-12
    A[0,1,2] = 0.5
    A[2,2,2] = 1.0
    A[0,2,2] = -30
    A[1,2,2] = 0.1
    W = np.random.rand(3,3)
    W1 = np.random.rand(3,3)
    assert (A.contract_all_indices(W).todense() == symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W))).all()
    assert (A.contract_all_indices(W1).todense() == symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1))).all()

# %% [markdown]
# ## Copying and Equality

# %%
if __name__ == "__main__":
    rank = 4
    dim = 10
    #test is_equal
    diagonal = np.random.rand(dim)
    odiag1 = np.random.rand()
    odiag2 = np.random.rand()
    A = SymmetricTensor(rank = rank, dim =dim)
    B = SymmetricTensor(rank = rank, dim =dim)
    A['iiii'] = diagonal
    B['iiii'] = diagonal
    A['iiij'] = odiag1
    B['iiij'] = odiag1
    A['iijj'] = odiag2
    B['iijj'] = odiag2
    assert A.is_equal(B)

    #test copying
    C = A.copy()
    assert C.is_equal(A)
