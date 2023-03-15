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
# # Symmetric tensors stored as permutation classes

# %%
from __future__ import annotations

# %%
from ast import literal_eval
from warnings import warn
from collections import Counter
import itertools
import more_itertools
import re  # Used in PermClsSymmetricTensor.Data.decode
from numbers import Number

import numpy as np

from symtensor import utils

from typing import Union, ClassVar, Any, Iterator, Generator, Dict, List, Tuple, Set
from scityping import Serializable
from scityping.numpy import Array, DType
from scityping.pydantic import dataclass

# %% [markdown] tags=[]
# **TODO** Include `TimeThis` (or equivalent) in symtensor.testing.utils

# %% [markdown]
# Notebook only imports

# %% tags=["active-ipynb", "remove-input"]
# from symtensor.base import SymmetricTensor,_elementwise_compare, _array_compare
# from symtensor import base
# from symtensor import utils
#
# import holoviews as hv
# hv.extension('bokeh')

# %% [markdown]
# Script only imports

# %% tags=["active-py", "remove-cell"]
from .base import SymmetricTensor, array_function_dispatch
from . import base
from . import utils

# %%
__all__ = ["PermSymmetricTensor"]


# %% [markdown]
# ## Rationale
#
# If we naively store tensors as high-dimensional arrays, they quickly grow quite large. However, reality does not need to be so bad, for two main reasons:
# 1. Since tensors must be symmetric, a fully specified tensor contains a lot of redundant information.
# 2. In practice, certain terms might be more important than others – e.g. the diagonal elements $A_{ii\dotsb}$.
#
# We want to avoid storing each element of the tensor, while still being efficient with those elements we do store. In particular, “similar” terms (like diagonal terms) should be stored as arrays, so that looping remains efficient and certain vectorized operations remain possible.
#
# While all memory layouts for symmetric tensors are concerned with (1), the layout described here, in terms of *permutation classes*, is meant to also address (2). It has the potential for huge reduction in memory use for highly structured tensors, but is probably less efficient for generic tensors.

# %% [markdown]
# ## Usage hints
#
# The `PermClsSymmetricTensor` class provides an outward interface is similar to an array: `A[1,2,3]` retrieves the tensor component $A_{123}$.
#
# Internally, components are grouped into permutation classes based on the form of their index. For example, all components with indices of the form `'iiii'` are grouped into one array; similarly for `'iij'`, etc. Instead of an array, a permutation class can also be associated to a single value. For example, if components of the form `'ijkl'` should all be zero, we simply associate zero to that class, instead of allocating a potentially large array of size $\frac{d(d-1)(d-2)(d-3)}{4!}$ elements.

# %% [markdown]
# ### Situations where a `PermClsSymmetricTensor` may be well suited
#
# In general, with a `PermClsSymmetricTensor` one gains a much reduced memory footprint for large tensors and very efficient operations on permutation classes, which are paid for by less efficient operations on the tensor as a whole.
#
# :::{margin}
# Triangles are comparison hints to other possible symmetric tensor implementations:
# ▲ – completely new feature; △ – partial improvement of feature;
# ▼ – complete loss of feature; ▽ – partial loss of feature.
# :::
#
# **(△) Construction of symmetric tensors.**
# ~ Allocation in terms of permutation classes makes it extremely convenient to assign to all entries of a particular form, e.g. `'iijj'`.
#
# **(△) Storing symmetric tensors in the most memory-efficient manner.**
# ~ When permutation classes can be represented as scalars, this is even more efficient than other symmetric storage layouts.
#
# **Computing combinatorics**
# ~ Number of equivalent permutations.
# ~ Number of independent components within a *permutation class*.
#
# **(▲) Iterating over particular permutation classes**
# ~ &nbsp;
#
# **Iterating over the entire tensor, when the order does not matter**
# ~ &nbsp;
#
# **(▲) Broadcasted operations on permutation classes.**
# ~ &nbsp;
#
# **(▽) Random access of a specific entry.**
# ~ Since there is no formula for directly computing memory position, the current implementation relies on a lookup dictionary. However it still benefits from the much lower memory size compared to a dense array.
#
# **Broadcasted operations with scalars operands.**
# ~ &nbsp;
#
# **(▽) Contractions with arrays or symmetric tensors**
# ~ Because of the compressed memory layout, dot products or contractions require looping over indices in non-trivial order. Blocked layouts (as proposed by Schatz et al (2014)), or a fully lexicographic layout, should allow for much faster memory-aligned operations, at least in some cases.
#
# ### Situations for which a `PermClsSymmetricTensor` may be less well suited:
#
# **Slicing a tensor as a normal array (per axis).**
# ~ It might be possible to implement this, although with additional overhead compared to normal arrays.
#
# **Broadcasted operations on the whole tensor with non-scalar operands.**
# ~ While it might possible to get this to work, it likely would not be much faster than just looping.
#
# ### Supported array operations.
#
# See [below](#Implementation-of-the-__array_function__-protocol).

# %% [markdown]
# ### Further development
#
# Not all functionality one would expect from an array-like object is yet available; features are implemented as they become needed.
#
# By design, operations on `PermClsSymmetricTensor`s involve iterating over values in non-trivial order, and cannot be expressed as simple, already-optimized dot products. Optimizing these operations is critical to applying our theory to more than low-dimensional toy examples; work in this regard, and justifications for some of the design decisions, can be found in the [developer docs](../developers/symmetric_tensor_algdesign.ipynb)

# %% [markdown]
# ### Notation summary
#
# | Symbol        | Desc              | Examples                             |
# |---------------|-------------------|--------------------------------------|
# | $d$           | dimension         | 2                                    |
# | $r$           | rank              | 4                                    |
# | $I$           | Multi-index       | $1010$<br>$1011$                     |
# | $\hat{I}$     | Index class       | $\widehat{0011}$<br>$\widehat{1110}$ |
# | $\hat{σ}$     | Permutation class | `iijj`<br>`iiij`                     |
# | $γ_{\hat{σ}}$ | multiplicity      | 6<br>4                               |
# | $s_{\hat{σ}}$ | size              | 1<br>2                               |
# | $l$           | Given $\hat{σ}$, number of different indices | 2<br>2    |
# | $m_n$         | Given $\hat{σ}$, number of indices repeated $n$ times  | See below |
# | $n_k$         | Given $\hat{σ}$, number of times index $k$ is repeated | See below |
#
#
# More examples:
#
# | $\hat{I}$ | $\hat{I}$ (str) | $l$ | $m$           | $n$     |
# |-----------|-----------------|-----|---------------|---------|
# |`(3,2)`    | `iiijj`         | 2   | 0, 1, 1, 0, 0 | 3, 2    |
# |`(1,1,1)`  | `ijk`           | 3   | 3, 0, 0       | 1, 1, 1 |
#
# NB: In code, we usually just write $σ$ instead of $\hat{σ}$.
#
# For more details, see the [documentation for the SymmetricTensor base class](./base.py).

# %% [markdown]
# ### Identities
#
# - $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} γ_{\hat{σ}} = d^r$
# - $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} = \binom{d + r - 1}{r}$
# - $\displaystyle s_{\hat{σ}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $m_n$ is the number of different indices which appear $n$ times.
# - $\displaystyle γ_{\hat{σ}} = \binom{r}{m_1,m_2,\dotsc,m_r} = \frac{r!}{n_1!n_2!\dotsb n_l!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $n_k$ is the number of times index $\hat{I}_k$ appears.

# %% [markdown]
# ### Storage format
#
# Tensors are stored as dictionaries, containing one flat array per permutation class. Additionally, if all entries associated to a permutation are equal, we allow storing that value as a scalar. For example, a rank 3, dimension 3 tensor $A$ would be stored as three arrays:
#
# | $\hat{σ}$ | values |
# |-----------|--------|
# |`iii` | $$(A_{000}, A_{111}, A_{222})$$ |
# |`iij` | $$(A_{001}, A_{002}, A_{110}, A_{112}, A_{220}, A_{221})$$ |
# |`ijk` | $$(A_{012})$$ |
#
# Each array value is associated to an index class. Helper functions are provided to iterate over index classes in a consistent, predictable order, so that the position in an array suffices to associate a value to an index class. (See [`σindex_iter`](#σindex_iter).)
#
# This design has the following advantages:
#
# - The number of different arrays one needs to maintain is determined only by the rank of the matrix, and remains modest. For example, a rank-4 tensor requires only 4 arrays. The overhead due to the additional Python structures should thus be manageable, and large dimensional tensors maximally benefit from NumPy's memory efficiencies since they are stored as a few long, flat arrays.
# - Tensors with additional structure (e.g. diagonal tensors) can be efficiently represented by assigning a scalar (typically 0) to potentially large permutation classes.
# - Sums over a tensor can conceivably be performed by a vectorized operation on the array of a permutation class, then multiplying the result by the multiplicity of that class.
#

# %% [markdown]
# ### Usage

# %% tags=["hide-output", "active-ipynb"]
#     # Indented code below is executed only when run in a notebook
#     from symtensor.permcls_symtensor import PermClsSymmetricTensor

# %% [markdown]
# When created, `SymmetricTensors` default to being all zero. Note that only one scalar value is saved per “permutation class”, making this especially space efficient.

# %% tags=["active-ipynb"]
#     A = PermClsSymmetricTensor(rank=4, dim=6)
#     display(A)

# %% [markdown]
# (▲) Each permutation class can be assigned either a scalar, or an array of the same length as the number of independent components of that class. For example, to make a tensor with 1 on the diagonal and different non-zero values for the double paired terms `'iijj'`, we can do the following.
# Note the use of `utils.get_permclass_size` to avoid having to compute how many independent `'iijj'` terms there are.

# %% tags=["active-ipynb"]
#     A['iiii'] = 1
#     A['iijj'] = np.arange(utils.get_permclass_size('iijj', A.dim))
#     display(A)

# %% [markdown]
# The `indep_iter` and `indep_iter_index` methods can be used to obtain a list of values where *each independent component appears exactly once*. (▲) Note that component values stored as scalars are expanded to the size of their class.

# %% tags=["active-ipynb"]
#     hv.Table(zip((str(idx) for idx in A.indep_iter_index()),
#                  A.indep_iter()),
#              kdims=["broadcastable index"], vdims=["value"])

# %% [markdown]
# Conversely, the `flat` method will return as many times as it appears in the full tensor, as though it was called on a dense representation of that tensor (although the order will be different).
#
# `flat_index` returns the index associated to each value.

# %% tags=["active-ipynb"]
#     hv.Table(zip(A.flat_index, A.flat), kdims=["index"], vdims=["value"])

# %% [markdown]
# The number of independent components can be retrieved with the `size` attribute.

# %% tags=["active-ipynb"]
#     A.size

# %% [markdown]
# To get the size of the full tensor, we need to multiply each permutation class $\hat{σ}$ size by its multiplicity $γ_{\hat{σ}}$ (the number of times components of that class are repeated due to symmetry).

# %% tags=["active-ipynb"]
#     (math.prod(A.shape)
#      == sum(utils.get_permclass_size(σcls, A.dim) * utils.get_permclass_multiplicity(σcls)
#             for σcls in A.perm_classes)
#      == A.dim**A.rank
#      == 1296)

# %% [markdown]
# Like the sparse arrays of *scipy.sparse*, a `SymmetricTensor` has a `.todense()` method which returns the equivalent dense NumPy array.

# %% tags=["active-ipynb"]
#     Adense = A.todense()

# %% tags=["hide-input", "active-ipynb"]
#     l1 = str(Adense[:1,:2])[:-1]
#     l2 = str(Adense[1:2,:2])[1:]
#     print(l1[:-1] + "\n\n   ...\n\n  " + l1[-1] + "\n\n"
#           + " " + l2[:-2] + "\n\n   ...\n\n  " + l2[-2] + "\n\n...\n\n" + l2[-1])

# %% [markdown]
# ## Implementation

# %% [markdown]
# > **Note**: There is a bijective map between string representations of permutation classes – `'iijk'` – and count representations – `(2,1,1)`. Public methods of `SymmetricTensor` use strings, while private methods use counts.

# %% [markdown]
# ### `σindex_iter`
#
# Given an permutation class and a dimension, `σindex_iter` produces a generator which yields all the *index classes* for that *permutation class*, in the order in which they would be stored as a flat vector.
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
# > **NOTE** Each index returned is only one of possibly many equivalent permutations. The `SymmetricTensor.indep_iter_index` method, in contrast to this function, combines all these permutations (i.e. all elements of the index class) into a single “[advanced index]”(https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing).

# %%
def _sub_σindex_iter(mult: Tuple[int], dim: int, prev_i: int, assigned_i: Set[int]):
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
                for subidx in _sub_σindex_iter(mult[1:], dim, j, assigned_i|{prev_i}):
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
                for subidx in _sub_σindex_iter(mult[1:], dim, j, assigned_i|{prev_i}):
                    yield subidx + [j]*m  # Append is faster than prepend; we will reverse at end


# %%
def σindex_iter(σcls: Tuple[int], dim: int) -> Generator[Tuple[int]]:
    """
    Given a permutation class and a dimension, return a generator which yields all
    the indices for that class, in the order in which they would be stored as
    a flat vector.
    """
    rank = sum(σcls)
    idx = []

    if len(σcls) == 0:  # Rank-0 tensor
        # Single, scalar value; associated to empty tuple
        yield ()
    if len(σcls) == 1:  # Diagonal terms
        for i in range(dim):
            yield (i,)*rank
    elif len(σcls) > dim:
        # Cannot have more distinct indices than than there are dimensions
        # => return an empty iterator
        return
    else:
        for i in range(dim):
            m = σcls[0]
            for subidx in _sub_σindex_iter(σcls, dim, i, set()):
                yield tuple(reversed(subidx + [i]*m))

# %% [markdown]
# ### WIP
#
# *Ordering permutation classes.*
# At some point I thought I would need a scheme for ordering permutation classes (for implementing a hierarchy, where e.g. `'ijkl'` can be used as a default for `'iijk'`). I save it here in case it turns out to be useful after all.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCLigZ1cIl1cbiAgXG4gICAgc3R5bGUgTiBmaWxsOm5vbmUsIHN0cm9rZTpub25lIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/edit##eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCJcdOKBnVwiXVxuICBcbiAgICBzdHlsZSBOIGZpbGw6bm9uZSwgc3Ryb2tlOm5vbmUiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)


# %% [markdown]
# ### Indexing utilities

# %% [markdown]
# #### `get_index_representative`
# Each set of indices equivalent under permutation has one representative index;
# this is the index returned by `*repindex` iterators.
# For a `PermClsSymmetricTensor`, this is given by grouping identical indices, then sorting first by repeat count then by index value.
#
# For example, given the input index `(2,1,2)`, it returns `(2,2,1)`.
#
# :::{note}
#
# Different conventions for index representatives may work best for different memory layouts, therefore each module defining a layout must also define a `get_index_representative` function. This function defines the convention for that layout.
# :::

# %%
def get_index_representative(index: Tuple[int]) -> Tuple[int]:
    "Return the representative for the index class to which `index` belongs."
    i_repeats = ((i, len(list(grouper))) for i, grouper in
                 itertools.groupby(sorted(index)))
    return sum(((i,)*repeats for i, repeats in
                sorted(i_repeats, key = lambda tup: tup[1], reverse=True)),
               start=())


# %% [markdown]
# #### Fast Indexing
#
# We need to be able to quickly retrieve entries in a SymmetricTensor. Given an index $(i,j,k,l,m)$, it might be fast to find the corresponding permulation class $\hat{σ}$, but not where in the array $\texttt{data}[\hat{σ}]$ this index is stored. (To our knowledge, there is no formula which is faster than simply iterating through all indices in $\texttt{data}[\hat{σ}]$ until $(i,j,k,l,m)$ is found.)
#
# To speed this up, we store {key: index} mappings in dictionaries and use them as a lookup table; this *position registry* is constructed the first time a SymmetricTensor is indexed into. Since Python dictionary lookups are reasonably fast and $\mathcal{O}(1)$, this should not scale catastrophically with tensor dimension. Moreover, since we only store one integer per stored SymmetricTensor element, the memory cost is no more than instantiating a second SymmetricTensor. (And possibly less, if the integers require less space.)
#
# Structure of `PosRegistry`:
#
#     {'iii': {(0,0,0): 0,
#              (1,1,1): 1
#              (2,2,2): 2
#             },
#      'iij': {(0,0,1): 3,
#              (0,0,2): 4,
#              (1,1,2): 5
#             },
#      'ijk': {(0,1,2): 6
#             }
#     }
#
# The current implementation has not been profiled; possibilities for speeding it up further would include:
# - Use the smallest possible integer type for positions, based on the size of the tensor.
# - Using strings instead of int tuples for position keys (e.g `"1_1_4_5_10"` instead of `(1,1,4,5,10)`) -> Hashes (and therefore lookups) on strings are faster, but this may be subsumed by the cost of conversion.
#   + It might be possible to write a low-level function and/or use unsafe casts to perform very fast conversions to raw bytes, which are as efficient as strings for hashing.
# - Subclassing `dict` directly instead of `UserDict` (probably negligible)
# - Precompute the position registry when the symmetric tensor is created, so we get rid of the `try:... except` statements. (probably negligible benefit, and in some cases unnecessary computation)
# - Low-level optimizations of the dictionary (hash table size, hash collision resolution — benefit uncertain but possibly substantial, would not be able to use Python dictionaries)
#
# Alternatively, we could get rid of the whole dictionary lookup and solve the problem this way:
# - Design a rule which, for each index, assigns a unique integer. This integer is then the index in a lookup table, which stores the data storage index.
#   This is *almost* like solving the problem of inverting the position -> index problem, except that a) the order of integers produced by the rule doesn't need to match storage order, and b) the rule doesn't need to produce all integers. If it makes the rule simpler/faster, the lookup table can have invalid positions, for integers the rule would never return.
#   AR: If we actually want to spend time making the `PosRegistry` faster, IMO this would be the way to go. It is possible to construct a rule for lexicographic ordering; the lookup table would then just be a translation from lexicographic to (perm class, index)
#

# %%
from collections import UserDict

class PosRegistry(UserDict):  #  TODO?: Make singleton ?
    @staticmethod
    def create_pos_dict(rank, dim):
        pos_dict = {}
        for perm_class in utils._perm_classes(rank):
            idx_pos_dict = {}
            for i,idx in enumerate(σindex_iter(perm_class,dim)):
                #could also convert index to string for faster evaluation ( something like: (1,1,4,5,10) -> '1_1_4_5_10'
                idx_pos_dict[idx] = i
            pos_dict[perm_class] = idx_pos_dict
        return pos_dict

    def __getitem__(self, key):
        try:
            return self.data[key]
        except KeyError:
            if len(key) != 2:
                raise KeyError(f"Received '{key}', but position dict only takes keys of length two, key = (rank,dimension)")
            pos_dict = self.create_pos_dict(*key)
            self.data[key] = pos_dict
            return pos_dict


pos_dict = PosRegistry()

# %%
def _convert_dense_index(rank, dim, key: Union[Tuple[int],int]
                        ) -> Tuple[Tuple[int,...], int]:
    """
    Given an index in dense format, return the class and position keys
    needed to index the matching value in `_data`.
    """
    if isinstance(key, tuple):
        # Use the standardized key so that it matches values returned by indep_iter_index
        key = get_index_representative(key)
        σcls = utils._get_permclass(key)
        i = pos_dict[(rank, dim)][σcls][key]
        return σcls, i
        if len(key) < rank:
            raise IndexError(
                "Partial indexing (where the number of indices is less "
                "than the rank) is inefficient with a PermClsSymmetricTensor and "
                "not currently supported.")
        else:
            raise IndexError(f"{key} does not seem to be a valid array index, "
                           "or some of its values exceed the tensor dimensions.")
    elif isinstance(key, int):
        if rank == 1:
            return (0,), key
        else:
            raise IndexError(
                "Partial indexing (where the number of indices is less "
                "than the rank) is inefficient with a PermClsSymmetricTensor and "
                "not currently supported.")
    else:
        raise TypeError(f"Unrecognized index type '{type(key)}' (value: {key}).\n"
                        "SymmetricTensor only supports strings "
                        "and integer tuples as keys.")


# %% [markdown]
# ### `PermClsSymmetricTensor`
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
# - *get_permclass(index: Tuple[int])*: From a specific index (`(0,0,1)`), return class string (`'iij'`).
# - *permclass_counts_to_label(counts: Tuple[int])*: Convert permutation class tuple (counts) (`(2,1)`) to class string (e.g. `'iij'`).
# - *permclass_label_to_counts(σcls: str | Tuple[int])*: Convert permutation class string to class tuple (counts).
# - [moved to utils] *get_permclass_size(σcls: str)* : Number of independent components in a permutation class, i.e. the size of the storage vector.
# - [moved to utils] *get_permclass_multiplicity(σcls: str)*: Number of times components in this permutation class are repeated in the full tensor.
# - *todense()*
# - *indep_iter()*     : Iterator over independent components (i.e. *excluding* symmetric equivalents).
# - *indep_iter_index()*     : Indices aligned with *indep_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
# - *permcls_indep_iter(σcls: str)*     : Iterator over independent components (i.e. *excluding* symmetric equivalents) within a permutation class.
# - *permcls_index_iter(σcls: str)*     : Indices aligned with *indep_permcls_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
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
# **Supported NumPy functions**
# - Use `@PermClsSymmetricTensor.implements` decorator to define support for NumPy functions like `np.transpose`.
# - See [below](#Implementation-of-the-__array_function__-protocol).
#
# **Remarks**
#
# - Partial or sliced indexing is not currently supported.
#   This could certainly be done, although it would require some indexing gymnastics
# - Similarly, the `__iter__` method is not implemented. **[TODO]**
#   To be consistent with a dense array, the produced iterator should yield
#   `PermClsSymmetricTensor` objects of rank `k-1` corresponding to partial indexing
#   along the first dimension.
# - Arithmetic operations are not currently supported.
#   Elementwise operations between tensors of the same size and rank can be trivially implemented when the need arises; other operations (like `.dot`) should be possible with some more work.

# %% [markdown]
# **Selected method dependencies**
#
# A → B: A depends on B.
#
# [![](https://mermaid.ink/img/pako:eNqNVMlugzAQ_RXkc_MDHHrocq0iJbe4shx7SCx5QV6UpohTP7C_VBtCSFhCOSAz8-a9GeZBhZjhgHJUSHNiR2p9tn3BGussXr8_QnP4IsKD7WIu7A-WlsdsDVa9Src5KwXeCrYF7YxtQVlWSOpvz6Rhqqr-XNddPj2WjQixUF6AE8G-oozaTDoyWfkgOak5FlxWm5ea1ZlnazOg-egdBy-k6wpJqiRMUuegDx5KtsOIHMDf5DH6vALcZUHf0PY2r_YR1Prc1ekyrWqXblcuzRcY0jTemL5nZtReaOqF0Y6chD-mVUjKQIG--iOGoHdLmiL4tmSglJrJVqvnm4reT03i3rGj9JC738GAdrjQJj1awCRq2MI01dRWJgy7qHsHHEqPrTYYc86__2RbGujBlzhSQE9IRTgVPP6MqlSNkT9Gn2CUxyOHggbpMcK6jtBQcurhnQtvLMoLKh08IRq82Zw1Q7m3ATrQm6DRnOqCqv8AjvXHJA)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNqNVMlugzAQ_RXkc_MDHHrocq0iJbe4shx7SCx5QV6UpohTP7C_VBtCSFhCOSAz8-a9GeZBhZjhgHJUSHNiR2p9tn3BGussXr8_QnP4IsKD7WIu7A-WlsdsDVa9Src5KwXeCrYF7YxtQVlWSOpvz6Rhqqr-XNddPj2WjQixUF6AE8G-oozaTDoyWfkgOak5FlxWm5ea1ZlnazOg-egdBy-k6wpJqiRMUuegDx5KtsOIHMDf5DH6vALcZUHf0PY2r_YR1Prc1ekyrWqXblcuzRcY0jTemL5nZtReaOqF0Y6chD-mVUjKQIG--iOGoHdLmiL4tmSglJrJVqvnm4reT03i3rGj9JC738GAdrjQJj1awCRq2MI01dRWJgy7qHsHHEqPrTYYc86__2RbGujBlzhSQE9IRTgVPP6MqlSNkT9Gn2CUxyOHggbpMcK6jtBQcurhnQtvLMoLKh08IRq82Zw1Q7m3ATrQm6DRnOqCqv8AjvXHJA)

# %%
class PermClsSymmetricTensor(SymmetricTensor):
    """
    On creation, defaults to a zero tensor.
    """
    #rank       : int
    #dim        : int
    #_dtype     : DType
    data_format : ClassVar[str]="PermCls"
    _data       : Dict[Tuple[int,...], Union[float, Array[float,1]]]

    def __init__(self, rank: Optional[int]=None, dim: Optional[int]=None,
                 data: Optional[Dict[Union[Tuple[int,...], str],
                                     Array[float,1]]]=np.float64(0),
                 dtype: Union[None,str,DType]=None, 
                 symmetrize: bool=False,):
        """
        Parameters
        ----------
        data: If provided, should be a dictionary of {σcls: 1d array} pairs.
        dtype: If both `data` and `dtype` are provided, the dtype of the former
           should match the latter.
           If only `data` is provided, dtype is inferred from the data.
           If only `dtype` is provided, it determines the data dtype.
        """
        super().__init__(rank=rank, dim=dim, data=data, dtype=dtype, symmetrize = symmetrize)
        # Sets rank, dim
        # Calls _validate_data
        # Sets _dtype
        # Calls _init_data

    def _validate_data(self,
                       data: Optional[Dict[Union[Tuple[int,...], str],
                                      Array[Any,1]]],
                       symmetrize=False
                       ) -> Tuple[Dict[Tuple[int,...], Array[Any,1]],
                                       np.dtype,
                                       Union[Tuple[int], None] ]:
        """
        {{base_docstring}}

        For the case of PermClsSymmetricTensor, this specifically means
        - Standardizing the `data` argument to a dict of {σ counts: array} pairs
        - Converting any scalars to 0-d arrays
        - Asserting that all array dtypes are numeric
        - Infer the dtype by applying type promotion on data dtypes
        """
        # Overview:
        # - 3 cases for data: scalar, dense array, mapping (matching internal σcls storage)
        # - For each case, we need to determine
        #   + Internal representation of data ({σcls: 1D-array})
        #   + dtype, as inferred from the data
        #   + shape, as inferred from the data
        # - These are returned as `data`, `datatype`, `datashape`
        # - `SymmetricTensor.__init__` compares these values with self.rank & self.dim,
        #   inferring for missing rank or dim and ensuring that all inferred values are consistent
        if isinstance(data, Number):
            arr_data = self._validate_dataarray(data)  # Normalizes to either Numpy or Torch array-like
            data = {σcls: arr_data for σcls in utils._perm_classes(self.rank)}
            datadtype = arr_data.dtype
            datashape = None  # SymmetricTensor.__init__ discards shape for scalar init values
        elif isinstance(data, np.ndarray):
            rank = np.ndim(data) if self.rank is None else self.rank
            dim = (self.dim if self.dim is not None
                   else max(*data.shape) if data.shape
                   else 1)  # Last line for scalars, which have an empty shape tuple
            datashape = (dim,)*rank
            datadtype = data.dtype
            broadcasted_data = np.broadcast_to(data, datashape)
            if rank == 0:
                data = {(): data}
            else:
                if symmetrize:
                    broadcasted_data = utils.symmetrize(broadcasted_data)
                elif not utils.is_symmetric(broadcasted_data):
                    raise ValueError("Data array is not symmetric.")
                data = {σcls: self._validate_dataarray(
                            broadcasted_data[tuple(np.array(idcs)
                                for idcs in zip(*σindex_iter(σcls, dim)))])
                        for σcls in utils._perm_classes(rank)
                        if len(σcls) <= dim}  # This condition guards against situations where rank > dim
        elif isinstance(data, dict):
            if len(data) == 0:
                raise NotImplementedError("Initializating with empty data is not implemented")
            # NB: If data is passed as a valid dict, it is by construction symmetric
            # If `data` comes from serialized JSON; revert strings to tuples
            # TODO: Better do the str -> tuple deserialization in Data.decode
            for key in list(data):
                if isinstance(key, str):
                    newkey = literal_eval(key)  # NB: That this is Tuple[int] is verified below
                    if newkey in data:
                        raise ValueError(f"`data` contains the key '{key}' "
                                         "twice: possibly in both its original "
                                         "and serialized (str) form.")
                    data[newkey] = data[key]
                    del data[key]

            # Infer the data dtype
            data = {k: self._validate_dataarray(v) for k, v in data.items()}
            datadtype = np.result_type(*data.values())

            rank = self.rank
            if rank is None:
                raise NotImplementedError("To instantiate with a mapping, we need to implement code which infers rank and dim from the mapping itself.")
            dim = self.dim
            if dim is None:
                dims = set(sum(counts) for counts in data.keys())
                if len(dims) > 1:
                    raise ValueError("Data dict inconsistent: keys don't all have the same dimension.")
                dim = next(iter(dims))

            datashape = (dim,)*rank

        else:
            raise TypeError("If provided, `data` must be a dictionary with "
                            "the format {σ class: data vector}")

        return data, datadtype, datashape

    def _init_data(self, data:  Dict[Tuple[int,...], Array[Any,1]], symmetrize: bool):
        # Assert that the data has the right shape
        # Convert scalars to 0-d arrays
        # Ensure dtypes are numeric
        if not data.keys() <= set(utils._perm_classes(self.rank)):  # NB: Allow setting only some σ-classes
            raise ValueError("`data` argument to PermClsSymmetricTensor does not "
                             "have the expected format.\nExpected keys to be a subset "
                             f"of: {sorted(utils._perm_classes(self.rank))}\n"
                             f"Received keys:{sorted(data)}")
        self._data = {k: utils.empty_array_like(self, (0,), dtype=self._dtype)
                      for k in utils._perm_classes(self.rank)}
        for k, v in data.items():
            # if np.isscalar(v):
            #     data[k] = v = np.array(v)
            v = self._validate_dataarray(v)
            if v.ndim > 0 and v.shape != (utils._get_permclass_size(k, self.dim),):
                raise ValueError(f"Data for permutation class {utils.permclass_counts_to_label(k)} "
                                 f"should have shape {(utils._get_permclass_size(k, self.dim),)}, "
                                 f"but the provided data has shape {v.shape}.")
            if v.dtype != self._dtype:
                v = v.astype(self._dtype)
            self._data[k] = v

    def _set_raw_data(self, key, arr):
        dataarr = self._data.get(key)
        assert key is not None, "Provided key does not match a dictionary entry."
        self._data[key] = arr

    ## Serialization ##
    @dataclass
    class Data(SymmetricTensor.Data):
        #rank: int
        #dim: int
        data: Dict[str, Array]  # NB: JSON keys cannot be tuples => convert to str
        _symtensor_type: ClassVar[Optional[type]]="PermClsSymmetricTensor"  # NB: Data.decode expects a string, in order resolve the forward ref

        @staticmethod
        def encode(symtensor: SymmetricTensor):
            return (symtensor.rank, symtensor.dim, {str(k): v for k,v in symtensor.items()})
        @classmethod
        def decode(cls, data: "SymmetricTensor.Data"):
            # Determine the SymmetricTensor name from the string
            try:
                import sys
                symtensor_type = getattr(sys.modules[cls.__module__], cls._symtensor_type)
            except AttributeError:
                raise NameError(f"Could not find '{cls._symtensor_type}' in module '{cls.__module__}'.")
            # Invert the conversion tuple -> str that was done in `encode`
            data_dict = {tuple(int(k) for k in re.findall(r"\d+", key_str)): arr
                         for key_str, arr in data.data.items()}
            # Instantiate the expected tensor
            return symtensor_type(data.rank, data.dim, data_dict)

    ## Dunder methods ##

    # def __str__(self)

    def __repr__(self):
        s = f"{type(self).__qualname__}(rank: {self.rank}, dim: {self.dim})"
        data = getattr(self, "_data", None)
        if data:
            lines = [f"  {utils.permclass_counts_to_label(σcls)}: {value}"
                     for σcls, value in self._data.items()]
        else:
            lines = []
        return "\n".join((s, *lines)) + "\n"  # Lists of SymmetricTensors look better if each tensor starts on its own line

    def __getitem__(self, key):
        """
        {{base_docstring}}

        .. Note:: slices with bounds (e.g. [3:8]) are not yet supported.
        """
        if isinstance(key, str):
            # str index => select perm class
            counts = utils.permclass_label_to_counts(key)
            return self._data[counts]

        elif isinstance(key, tuple):
            if any([isinstance(i,slice) for i in key]) or isinstance(key, slice):
                # Index involves a slice => convert to integer indices
                indices_fixed = tuple(i for i in key if isinstance(i,int))
                slices = [i for i in key if isinstance(i,slice)]
                assert len(indices_fixed) + len(slices) == len(key), "SymmetricTensor index should contain only integers and slices."
                #Check for subslicing
                for s in slices:
                    if s != slice(None):
                        raise NotImplementedError("Indexing with subslicing (for example SymmetricTensor[1:3, 0,0]) is not"
                                                  " currently implemented. Only slices of the type"
                                                  "[i_1,...,i_n,:,...,:] with i_1,..., i_n all integers are allowed.")

                key = indices_fixed  # Order does not matter => Just drop the `:` slices from the key
            # NB: Current implementation allows only `:` slices
            if len(key) < self.rank:
                # Fewer indices than rank => return a lower rank Symtensor
                new_rank = self.rank - len(key)
                C = self.__class__(rank=new_rank, dim=self.dim)  # __class__ avoids the need to reimplement for different backends
                for idx in C.indep_iter_repindex():
                    C[idx] = self[idx + key]
                return C
            else:
                # As many indices as rank => return scalar
                σcls = utils._get_permclass(key)
                vals = self._data[σcls]
                if np.ndim(vals) == 0:
                    return vals
                else:
                    σcls, pos = _convert_dense_index(self.rank, self.dim, key)
                    return vals[pos]

        elif self.rank==1 and isinstance(key,int): #special rules for vectors
            vals = self._data[(1,)]
            return vals if np.isscalar(vals) else vals[key]

        elif self.rank > 1 and isinstance(key, int):
            if self.dim == 1:
                σcls, pos = _convert_dense_index(self.rank, self.dim, key)
                vals = self._data[σcls]
                return vals if np.isscalar(vals) else vals[pos]
            elif self.dim > 1:
                B = PermClsSymmetricTensor(rank = self.rank-1, dim = self.dim)
                for idx in B.indep_iter_repindex():
                    B[idx] = self[idx+(key,)]

                return B

        else:
            raise KeyError(f"{key}")

    def __setitem__(self, key, value):
        ## Special, short-circuited branch if `value` is data-aligned ##
        #  (Skips a potentially costly cast to a dense array)
        if (  isinstance(value, PermClsSymmetricTensor)
              and self.data_alignment == value.data_alignment
              and key == slice(None)  ):
            for k, v in self._data.items():
                if isinstance(v, np.ndarray) and v.ndim > 0:
                    v[:] = self._validate_dataarray(value._data[k])
                else:
                    # Scalars need to be overwritten
                    self._data[k] = self._validate_dataarray(value._data[k])
            return  # EARLY EXIT

        if isinstance(value, SymmetricTensor):
            raise NotImplementedError(
                "For lack of a use case, assignment of SymmetricTensor into "
                "another SymmetricTensor is only defined when doing a full "
                "assignment with data-aligned tensors (e.g. A[:] = B).")

        ## Normal generic branch ##
        #value = np.asarray(value).astype(self.dtype)
        value = self._validate_dataarray(value)
        if key == slice(None):
            # Special case: we allow replacing the entire data
            dict_value, _, shape = self._validate_data(value)
            if shape != self.shape or not dict_value.keys() <= self._data.keys():
                raise ValueError("Cannot assign to SymmetricTensor: value has an incompatible shape.")
            for k, v in dict_value.items():
                # NB: User expects in-place assignment, so make sure we do this
                if np.ndim(self._data[k]) == 0:
                    self._data[k] = v
                else:
                    self._data[k][:] = v
        elif isinstance(key, str):
            counts = utils._get_permclass(tuple(key))
            if counts not in self._data:
                raise KeyError(f"'{key}' does not match any permutation class.\n"
                               f"Permutation classes: {self.perm_classes}.")
            if np.ndim(value) == 0:
                self._data[counts] = value
            else:
                if len(value) != utils._get_permclass_size(counts, self.dim):
                    raise ValueError(
                        "Value must either be a scalar, or match the index "
                        f"class size.\nValue size: {len(value)}\n"
                        f"Permutation class size: {utils._get_permclass_size(counts, self.dim)}")
                if isinstance(value, (list, tuple)):
                    value = np.array(value)
                self._data[counts] = value
        else:
            if self.rank==1 and isinstance(key,int): #special rules for vectors
                σcls = (1,)
                pos = key
            else:
                σcls, pos = _convert_dense_index(self.rank, self.dim, key)
            v = self._data[σcls]
            if np.ndim(v) == 0:
                if pos == slice(None):  # Equivalent to setting the whole permutation class
                    self._data[σcls] = value
                elif np.ndim(value) == 0 and v == value:
                    # Value has not changed; no need to expand
                    pass
                else:
                    # Value is no longer uniform for all positions => need to expand storage from scalar to vector
                    # v = v * np.ones(utils._get_permclass_size(σcls, self.dim),
                    #                 dtype=np.result_type(v, value))
                    v = v * self._validate_dataarray(  # Not as clean as creating the ones with the correct dtype immediately, but
                        np.ones(utils._get_permclass_size(σcls, self.dim), dtype="int8"))  # this works with the Torch backend too
                    v[pos] = value
                    self._data[σcls] = v
            else:
                self._data[σcls][pos] = value

    def __iter__(self):
        raise NotImplementedError("Standard iteration, as with a dense tensor, "
                                  "would require extra work but could be supported.")

    ## Translation functions ##
    # Mostly used internally, but part of the public API

    # def copy(self) -> PermClsSymmetricTensor

    ## Public attributes & API ##

    # @property
    # def dtype(self) -> np.dtype

    # @property
    # def shape(self) -> Tuple[int,...]

    @property
    def size(self) -> int:
        # `size` must return the maximum number of allocated elements.
        # For permutation-class storage, this is exactly the number of independent components
        return self.indep_size

    def todense(self) -> Array:
        A = np.empty(self.shape, self.dtype)
        for idx, value in zip(self.indep_iter_index(), self.indep_iter()):
            A[idx] = value
        return A

    ## Iterators ##
    # To facilitate comparisons, empty values are standardized: although empty
    # permutation classes can be equivalently be stored as empty arrays or
    # scalars of any value, these iterators always return empty arrays

    def keys(self):
        return self._data.keys()
    def values(self):
        dim = self.dim
        return [v if len(k) <= dim else self._validate_dataarray(np.array([]))
                for k, v in self._data.items()]
    def items(self):
        return [(k, v) for k,v in zip(self.keys(), self.values())]

    @property
    def flat(self):
        """
        {{base_docstring}}

        .. Note:: At present, in contrast to NumPy's `flat`, it is not possible
           to set values with this iterator (since it is an iterator rather
           than a view).
        """
        for σcls, v in self._data.items():
            mult = utils.get_permclass_multiplicity(σcls)
            if np.ndim(v) == 0:
                size = utils._get_permclass_size(σcls, self.dim)
                yield from itertools.repeat(v, size*mult)
            else:
                for vi in v:
                    yield from itertools.repeat(vi, mult)

    @property
    def flat_index(self):
        """
        {{base_docstring}}

        .. Note:: For looping over an entire dense tensor, the advanced index
           returned by `indep_iter_index` should in general be more efficient than
           this one.
        """
        for counts in self._data:
            for index in σindex_iter(counts, self.dim):
                yield from sorted(more_itertools.distinct_permutations(index))

    def indep_iter(self) -> Generator:
        """
        {{base_docstring}}

        Values stored as a scalar (when all components in a permutation class
        have the same value) are returned multiple times, as many as the size
        of that class. The output thus does not depend on whether values
        are stored as scalars or arrays.
        """
        try:
            for σcls, v in self._data.items():
                if np.ndim(v) == 0:
                    size = utils._get_permclass_size(σcls, self.dim)
                    yield from itertools.repeat(v, size)
                else:
                    yield from v
        except AttributeError:
            if self._data is None:
                raise RuntimeError("Symmetric tensor was initialized empty. Cannot create the `indep_iter` iterator.")
            else:
                raise

    # def indep_iter_index(cls) -> Generator[Tuple[List[int],...]]:

    def indep_iter_repindex(self) -> Generator:
        for counts in utils._perm_classes(self.rank):
            yield from σindex_iter(counts, self.dim)  # Inlined permcls_indep_iter_repindex

    def permcls_indep_iter(self, σcls: Union[str, Tuple[int,...]]) -> Generator:
        if isinstance(σcls, str):
            σcls = utils.permclass_label_to_counts(σcls)
        v = self._data[σcls]
        if np.ndim(v) == 0:
            size = utils.get_permclass_size(σcls, self.dim)
            yield from itertools.repeat(v, size)
        else:
            yield from v

    # def permcls_indep_iter_index(cls, σcls: Union[str, Tuple[int]]
    #                             ) -> Generator[Tuple[List[int],...]]:

    def permcls_indep_iter_repindex(self, σcls: Union[str, Tuple[int]]
                                   ) -> Generator[Tuple[int]]:
        if isinstance(σcls, str):
            σcls = utils.permclass_label_to_counts(σcls)
        return σindex_iter(σcls, self.dim)


# %% [markdown]
# ### Implementation of the `__array_function__` protocol
#

# %%
@PermClsSymmetricTensor.implements(np.einsum_path)
def einsum_path(*operands, optimize='greedy', einsum_call=False):
    with utils.make_array_like(PermClsSymmetricTensor(0,0), np.core.einsumfunc):
        return np.core.einsumfunc.einsum_path.__wrapped__(
            *operands, optimize=optimize, einsum_call=einsum_call)

# %% [markdown]
# #### `array_equal()`
#
# Overriden to allow for scalars in the underlying arrays: underlying arrays not having the same shape is fine

# %%
from symtensor import base

@PermClsSymmetricTensor.implements(np.array_equal)
def array_equal(a, b) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are equal. C.f. `numpy.array_equal`.
    """
    return np.shape(a) == np.shape(b) and base._array_compare(
        lambda x, y: np.all(x == y), a , b)

# TODO
#@PermClsSymmetricTensor.implements(np.einsum)
#def einsum(*operands, dtype=None, order='K', casting='safe', optimize=False):
#    # NB: Can't used the implementation in np.core.einsumfunc, because that calls
#    #     C code which requires true arrays
#    ...

# %% [markdown]
# ### Symmetrized operations
#
# #These functions replace standard NumPy ones, making them symmetric.
#
# **TODO**: Provide implementations of `symmetric_outer` and `symmetric_tensordot` that don’t depend on the inefficient generic implementation in *symalg.py* (which just converts to dense).
#
# @SymmetricTensor.implements_ufunc.outer(np.add, np.sub, np.multiply)
# def symmetric_outer(self, other, ufunc=np.multiply):
#     """
#     Implement the outer product. Note that the outer product of two symmetric tensors is not symmetric.
#     The result generated here is the symmetrized version of the outer product.
#     """
#     if isinstance(other, SymmetricTensor):
#         if self.dim != other.dim:
#             raise NotImplementedError("Currently only outer products between SymmetricTensors of the same dimension are supported.")
#         else:
#             C = SymmetricTensor(dim=self.dim, rank=self.rank+other.rank)
#             for I in C.permcls_indep_iter_repindex():
#                 list1, list2, L = partition_list_into_two(I, self.rank, other.rank)
#                 C[I] = sum( ufunc(self[tuple(idx1)], other[tuple(idx2)]) for idx1, idx2 in zip(list1,list2) )/L
#             return C
#     elif isinstance(other, list):
#         C = self.copy()
#         for o in other:
#             C = C.outer_product(o)
#         return C
#     elif not isinstance(other, (SymmetricTensor,list)):
#         raise TypeError( 'Argument must be SymmetricTensor or list of SymmetricTensors')
#
# @PermClsSymmetricTensor.implements(np.tensordot)
# def symmetric_tensordot(self, other, axes=2):
#     """
#     Like numpy.tensordot, but outputs are all symmetrized.
#     """
#     if not isinstance(other, SymmetricTensor):
#         # "Currently only tensor products between SymmetricTensors are supported."
#         return NotImplemented
#     if self.dim != other.dim:
#         # "Currently only tensor products between SymmetricTensors of the same dimension are supported."
#         return NotImplemented
#     if isinstance(axes,int):
#         if axes == 0:
#             return self.outer_product(other)
#         elif axes == 1:
#             # note: \sum_i A_jkl..mi B_inop..z = \sum_i A_ijkl..m B_inop..z for A, B symmetric
#             if other.rank == 1 and self.rank ==1:
#                 return np.dot(self['i'],other['i'])
#             elif other.rank ==1 and self.rank >1:
#                 return sum((self[i]*other[i] for i in range(self.dim)),
#                            start=SymmetricTensor(self.rank -1, self.dim))
#             elif other.rank >1 and self.rank ==1:
#                 return sum((self[i]*other[i] for i in range(self.dim)),
#                            start=SymmetricTensor(other.rank -1, self.dim))
#             else:
#                 return sum((self[i].outer_product(other[i]) for i in range(self.dim)),
#                        start=SymmetricTensor(self.rank + other.rank - 2, self.dim))
#         elif axes == 2:
#             if self.rank < 2 or other.rank < 2:
#                 raise ValueError("Both tensors must have rank >=2")
#             get_slice_index = lambda i,j,rank: (i,j,) +(slice(None,None,None),)*(rank-2)
#             if self.rank ==2 or other.rank==2:
#                  C = sum((np.multiply(
#                         self[get_slice_index(i,j,self.rank)],
#                         other[get_slice_index(i,j,other.rank)])
#                      for i in range(self.dim) for j in range(other.dim)),
#                     start=SymmetricTensor(self.rank + other.rank - 4, self.dim))
#             else:
#                 C = sum((np.multiply.outer(
#                             self[get_slice_index(i,j,self.rank)],
#                             other[get_slice_index(i,j,other.rank)])
#                          for i in range(self.dim) for j in range(other.dim)),
#                         start=SymmetricTensor(self.rank + other.rank - 4, self.dim))
#             return C
#         else:
#             raise NotImplementedError("tensordot is currently implemented only for 'axes'= 0, 1, 2. "
#                                       f"Received: {axes}")
#     elif isinstance(axes, tuple):
#         axes1 ,axes2 = axes
#         if isinstance(axes1, tuple):
#             if not isinstance(axes2, tuple):
#                 raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
#                                 f"Received: {axes}")
#             if len(axes1) != len(axes2):
#                 raise ValueError("# dimensions to sum over must match")
#             rank_deduct = len(axes1)
#             get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-rank_deduct)
#             C = sum((np.multiply.outer(self[get_slice_index(idx,self.rank)],
#                                        other[get_slice_index(idx,other.rank)])
#                      for idx in itertools.product(range(self.dim),repeat = rank_deduct)),
#                     start=SymmetricTensor(self.rank + other.rank - 2*rank_deduct, self.dim))
#             return C
#         elif isinstance(axes1,int):
#             if not isinstance(axes2,int):
#                 raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
#                                 f"Received: {axes}")
#             return self.tensordot(other, axes = 1)
#         else:
#             raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
#                             f"Received: {axes}")
#     else:
#         raise NotImplementedError("Tensordot with more axes than two is currently not implemented. "
#                                   f"Received: axes={axes}")

# %% [markdown]
# ### Additional linear algebra operations
#
#
# def _index_perm_prod_sum(W, idx_fixed, idx_permute):
#     """
#     For index_fixed = (j_1, ... j_r)
#     \sum_{(i_1, ... i_r) in σ(idx_permute)} W_{i_1,j_1} ... W_{i_n, j_n}
#     where σ(idx_permute) are all unique permutations.
#     """
#     idx_counts = _get_permclass(idx_permute) # number of repeats of indices
#     permutations_of_identical_idx = np.prod([math.factorial(r) for r in idx_counts])
#     matr = np.array([[W[i,j] for i,j in zip(σidx, idx_fixed)]
#                       for σidx in itertools.permutations(idx_permute)])
#     return np.sum( np.prod(matr, axis=1)) /permutations_of_identical_idx
#
# @PermClsSymmetricTensor.implements(symalg.contract_all_indices_with_matrix)
# def contract_all_indices_with_matrix(self,W):
#     """
#     compute the contraction over all indices with a non-symmetric matrix, e.g.
#
#     C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}
#
#     if current tensor has rank 3.
#     """
#
#     C = PermClsSymmetricTensor(rank = self.rank, dim = self.dim)
#
#     for σcls in self.perm_classes:
#         C[σcls] = [ np.sum([_index_perm_prod_sum(W, idx_fixed, idx_permute)*self[idx_permute]
#                       for idx_permute in self.permcls_indep_iter_repindex()]) for idx_fixed in self.permcls_indep_iter_repindex(σcls= σcls) ]
#
#     return C

# %% [markdown]
#
# The following methods are supported via the default implementation in symalg.py
#
# - `contract_all_indices_with_vector`
# - `contract_tensor_list`

# %% [markdown]
# ## Memory footprint
# Memory required to store tensors, as a percentage of an equivalent tensor with dense storage.

# %% tags=["active-ipynb"]
# import sys
# curves = []
# for rank in [1, 2, 3, 4, 5, 6]:
#     points = []
#     for dim in [1, 2, 3, 4, 6, 8, 10, 20, 40, 80, 100]:
#         # Dense size (measured in bytes; assume 64-bit values)
#         dense = dim**rank*8 + 104  # 104: fixed overhead for arrays
#         # SymmetricTensor size
#         A = PermClsSymmetricTensor(rank=rank, dim=dim)
#         # For dicts, getsizeof oly counts the number of keys/value pairs,
#         # not the size of either of them.
#         sym = sys.getsizeof(A)
#         sym += sum(sys.getsizeof(k) + sys.getsizeof(v) for k,v in A.__dict__.items())
#         sym += sum(sys.getsizeof(k) for k in A._data)
#         sym += sum(utils.get_permclass_size(σcls, A.dim)*8 + 104 for σcls in A.perm_classes)
#         # Append data point
#         points.append((dim, sym/dense))
#     curves.append(hv.Curve(points, kdims=['dimensions'], vdims=['size (rel. to dense)'],
#                            label=f"rank = {rank}"))
#     if rank > 4:
#         curves[-1].opts(muted=True)
# fig = hv.Overlay(curves) * hv.HLine(1)
# fig.opts(hv.opts.HLine(color='grey', line_dash='dashed', line_width=2, alpha=0.5),
#          hv.opts.Curve(width=500, logy=True, logx=True),
#          hv.opts.Overlay(legend_position='right'))
# display(fig)
#
#
#

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
#     # Context manager works as expected…
#     with utils.make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#         assert "<locals>" in str(np.core.einsumfunc.asanyarray)   # asanyarray has been substituted…
#         np.einsum('iij', np.arange(8).reshape(2,2,2))  # …and einsum still works
#         np.asarray(np.arange(3))                       # Plain asarray is untouched and still works
#     # …and returns the module to its clean state on exit…
#     assert "<locals>" not in str(np.core.einsumfunc.asanyarray)
#     with pytest.warns(UserWarning):
#         assert type(np.asarray(A)) is np.ndarray
#     # …even when an error is raised within the context.
#     try:
#         with utils.make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#             assert "<locals>" in str(np.core.einsumfunc.asanyarray)
#             raise ValueError
#     except ValueError:
#         pass
#     assert "<locals>" not in str(np.core.einsumfunc.asanyarray)

# %% [markdown]
# Test dispatched array functions which use the `make_array_like` decorator to avoid coercion.

# %% [markdown]
#     with does_not_warn(UserWarning):
#         np.einsum_path("ij,ik", A, B)
#         np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))
#
#     with utils.make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#         with does_not_warn(UserWarning):
#             np.einsum_path("ij,ik", A, B)
#             np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))


# %% [markdown]
# ## Timings

# %% [markdown]
# ### Slicing
# Some tests to see where slowness could come from:

# %% tags=["active-ipynb"]
# TimeThis.on= True
# with TimeThis("check slicing speed", lambda name, Δ: None):
#     D = A[0]


# %% [markdown]
# ### Outer product:

# %% tags=["active-ipynb"]
# for rank in [3]:
#     for dim in [50]:
#         vect = SymmetricTensor(rank=1, dim=dim)
#         vect['i'] = np.random.rand(dim)
#         print('rank = ', rank)
#         print('dim = ', dim)
#         with TimeThis('pos_dict_creation', output_last_Δ=lambda name, Δ: None):
#             x = pos_dict[rank,dim]
#         with TimeThis('outer product', output_last_Δ=lambda name, Δ: None):
#             # vect x vect x vect ... x vect
#             A = np.multiply.outer([vect,vect], [vect,vect])
#             # Old: vect.outer_product([vect,]*(rank-1))


# %% [markdown]
# ## WIP
#
# *Ordering permutation classes.*
# At some point I thought I would need a scheme for ordering permutation classes (for implementing a hierarchy, where e.g. `'ijkl'` can be used as a default for `'iijk'`). I save it here in case it turns out to be useful after all.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCLigZ1cIl1cbiAgXG4gICAgc3R5bGUgTiBmaWxsOm5vbmUsIHN0cm9rZTpub25lIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/edit##eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCJcdOKBnVwiXVxuICBcbiAgICBzdHlsZSBOIGZpbGw6bm9uZSwgc3Ryb2tlOm5vbmUiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)
