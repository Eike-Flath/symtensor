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
import time
from tqdm.auto import tqdm
from pydantic import BaseModel

import math  # For operations on plain Python objects, math can be 10x faster than NumPy
import numpy as np

from mackelab_toolbox.utils import TimeThis
import statGLOW
from statGLOW.utils import does_not_warn
import statGLOW.stats.symtensor.symtensor.utils as utils

from typing import Union, ClassVar, Any, Iterator, Generator, Dict, List, Tuple, Set
from scityping import Serializable, Array, DType

# %% tags=["hide-input", "active-ipynb"]
# # Notebook only imports
# import holoviews as hv
# hv.extension('bokeh')

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
# **(△) Storing symmetric tensors in the _most_ memory-efficient manner.**
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
# ~ Because of the compressed memory layout, dot products or contractions require looping over indices in non-trivial order. Blocked layouts (as proposed by Schatz et al (2014)), or a fully lexicographic layout, should allow for much faster memory-aligned operations at least in some cases.
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
# By design, operations on `PermClsSymmetricTensor`s involve iterating over values in non-trivial order, and cannot be expressed as simple, already-optimized dot products. Optimizing these operations is critical to applying our theory to more than low-dimensional toy examples; work in this regard, and justifications for some of the design decisions, can be found in the [developer docs](../../docs/developers/SymmetricTensor/symmetric_tensor_algdesign.ipynb)

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
#     from permcls_symmetric_tensor import PermClsSymmetricTensor

# %% [markdown]
# When created, `SymmetricTensors` default to being all zero. Note that only one scalar value is saved per “permutation class”, making this especially space efficient.

# %% tags=["active-ipynb"]
#     A = PermClsSymmetricTensor(rank=4, dim=6)
#     display(A)

# %% [markdown]
# (▲) Each permutation class can be assigned either a scalar, or an array of the same length as the number of independent components of that class. For example, to make a tensor with 1 on the diagonal and different non-zero values for the double paired terms `'iijj'`, we can do the following.
# Note the use of `get_permclass_size` to avoid having to compute how many independent `'iijj'` terms there are.

# %% tags=["active-ipynb"]
#     A['iiii'] = 1
#     A['iijj'] = np.arange(A.get_permclass_size('iijj'))
#     display(A)

# %% [markdown]
# The `indep_iter` and `index_iter` methods can be used to obtain a list of values where *each independent component appears exactly once*. (▲) Note that component values stored as scalars are expanded to the size of their class.

# %% tags=["active-ipynb"]
#     hv.Table(zip((str(idx) for idx in A.index_iter()),
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
#      == sum(A.get_permclass_size(σcls)*A.get_permclass_multiplicity(σcls)
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
# > **NOTE** Each index returned is only one of possibly many equivalent permutations. The `SymmetricTensor.index_iter` method, in contrast to this function, combines all these permutations (i.e. all elements of the index class) into a single “[advanced index]”(https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing).

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
        # Use the standardized key so that it matches values returned by index_iter
        key = _get_index_representative(key)
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
# - *get_permclass_size(σcls: str)* : Number of independent components in a permutation class, i.e. the size of the storage vector.
# - *get_permclass_multiplicity(σcls: str)*: Number of times components in this permutation class are repeated in the full tensor.
# - *todense()*
# - *indep_iter()*     : Iterator over independent components (i.e. *excluding* symmetric equivalents).
# - *index_iter()*     : Indices aligned with *indep_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
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
    #rank                 : int
    #dim                  : int
    #_dtype                : DType
    _data                : Dict[Tuple[int], Union[float, Array[float,1]]]

    def __init__(self, rank: int, dim: int,
                 data: Optional[Dict[Union[Tuple[int,...], str],
                                     Array[float,1]]]=None,
                 dtype: Union[None,str,DType]=None):
        """
        Parameters
        ----------
        data: If provided, should be a dictionary of {σcls: 1d array} pairs.
        dtype: If both `data` and `dtype` are provided, the dtype of the former
           should match the latter.
           If only `data` is provided, dtype is inferred from the data.
           If only `dtype` is provided, it determines the data dtype.
        """
        super().__init__(rank=rank, dim=dim, data=data, dtype=dtype)
        # Sets rank, dim, device, _σclass_sizes, _σclass_multiplicities
        # Calls _validate_data
        # Sets _dtype
        # Calls _init_data

    @classmethod
    def _validate_data(cls,
                       data: Optional[Dict[Union[Tuple[int,...], str],
                                      Array[Any,1]]]
                       ) -> Dict[Tuple[int,...], Array[Any,1]]:
        """
        {{base_docstring}}

        For the case of PermClsSymmetricTensor, this specifically means
        - Standardizing the `data` argument to a dict of {σ counts: array} pairs
        - Converting any scalars to 0-d arrays
        - Asserting that all array dtypes are numeric
        - Infer the dtype by applying type promotion on data dtypes
        """
        if isinstance(data, np.ndarray):
            raise NotImplementedError("Casting plain arrays to PermClsSymmetricTensor "
                                      "is not yet supported.")
            data, datadtype = DenseSymmetricTensor._validate_data(data, symmetrize)
            # TODO: 1. Remove dependency on self.perm_classes (through permcls_indep_iter_repindex)
            #       2. Test
            if not utils.is_symmetric(data):
                raise ValueError("Data array is not symmetric.")
            data = {σcls: data[list(self.permcls_indep_iter_repindex(σcls))]
                    for σcls in utils._perm_classes(self.rank)}
        elif isinstance(data, dict):
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

            # Infer the data dtype
            datadtype = np.result_type(data.values())
            if not np.issubdtype(datadtype, np.number):
                    raise TypeError("Data should have numeric dtypes; received "
                                    f"{[np.result_type(v) for v in data.values()]}.")

            # Assert that the deserialized data has the right shape
            # Convert scalars to 0-d arrays
            # Ensure dtypes are numeric
            if not data.keys() <= set(utils._perm_classes(self.rank)):  # NB: Allow setting only some σ-classes
                raise ValueError("`data` argument to PermClsSymmetricTensor does not "
                                 "have the expected format.\nExpected keys to be a subset "
                                 f"of: {sorted(utils._perm_classes(self.rank))}\n"
                                 f"Received keys:{sorted(data)}")
            for k, v in data.items():
                if np.isscalar(v):
                    data[k] = v = np.array(v)
                if v.ndim > 0 and v.shape != (self._σclass_sizes[k],):
                    raise ValueError(f"Data for permutation class {self.permclass_counts_to_label(k)} "
                                     f"should have shape {(self._σclass_sizes[k],)}, "
                                     f"but the provided data has shape {v.shape}.")

        else:
            raise TypeError("If provided, `data` must be a dictionary with "
                            "the format {σ class: data vector}")

        return data, datadtype

    def _init_data(self, data:  Dict[Tuple[int,...], Array[Any,1]]):
        self._data = {k: v.astype(self._dtype) if v.dtype != self._dtype
                      for k, v in data.items()}

    ## Dunder methods ##

    # def __str__(self)

    def __repr__(self):
        s = f"{type(self).__qualname__}(rank: {self.rank}, dim: {self.dim})"
        lines = [f"  {self.permclass_counts_to_label(σcls)}: {value}"
                 for σcls, value in self._data.items()]
        return "\n".join((s, *lines)) + "\n"  # Lists of SymmetricTensors look better if each tensor starts on its own line

    def __getitem__(self, key):
        """
        {{base_docstring}}

        .. Note:: slices are not yet supported.
        """
        if isinstance(key, str):
            counts = self.permclass_label_to_counts(key)
            return self._data[counts]

        elif isinstance(key, tuple):
            if any([isinstance(i,slice) for i in key]) or isinstance(key, slice):
                indices_fixed = tuple(i for i in key if isinstance(i,int))
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
                    new_rank = self.rank -len(indices_fixed)
                    C = PermClsSymmetricTensor(rank = new_rank, dim = self.dim)
                    for idx in C.permcls_indep_iter_repindex():
                        C[idx] = self[idx +indices_fixed]
                    return C

            else:
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
        elif self.rank >1 and isinstance(key, int):
            if self.dim ==1:
                σcls, pos = self._convert_dense_index(self.rank, self.dim, key)
                vals = self._data[σcls]
                return vals if np.isscalar(vals) else vals[pos]
            elif self.dim >1:
                B = PermClsSymmetricTensor(rank = self.rank-1, dim = self.dim)
                for idx in B.permcls_indep_iter_repindex():
                    B[idx] = self[idx+(key,)]

                return B
        else:
            raise KeyError(f"{key}")

    def __setitem__(self, key, value):
        if isinstance(key, str):
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
                σcls, pos = self._convert_dense_index(self.rank, self.dim, key)
            v = self._data[σcls]
            if np.ndim(v) == 0:
                if pos == slice(None):  # Equivalent to setting the whole permutation class
                    self._data[σcls] = value
                elif np.ndim(value) == 0 and v == value:
                    # Value has not changed; no need to expand
                    pass
                else:
                    # Value is no longer uniform for all positions => need to expand storage from scalar to vector
                    v = v * np.ones(self._σclass_sizes[σcls], dtype=np.result_type(v, value))
                    v[pos] = value
                    self._data[σcls] = v
            else:
                self._data[σcls][pos] = value

    def __iter__(self):
        raise NotImplementedError("Standard iteration, as with a dense tensor, "
                                  "would require extra work but could be supported.")

    ## Translation functions ##
    # Mostly used internally, but part of the public API

    # def device(self) -> Union[None, "torch.device"]:

    # def copy(self) -> PermClsSymmetricTensor

    # def is_equal(self, other, prec =None)

    ## Public attributes & API ##

    # @property
    # def dtype(self) -> np.dtype

    # @property
    # def shape(self) -> Tuple[int,...]

    # @property
    # def size(self) -> int

    def todense(self) -> Array:
        A = np.empty(self.shape, self.dtype)
        for idx, value in zip(self.index_iter(), self.indep_iter()):
            A[idx] = value
        return A

    ## Iterators ##

    def keys(self):
        return self._data.keys()
    def values(self):
        return self._data.values()
    def items(self):
        return self._data.items()

    @property
    def flat(self):
        """
        {{base_docstring}}

        .. Note:: At present, in contrast to NumPy's `flat`, it is not possible
           to set values with this iterator (since it is an iterator rather
           than a view).
        """
        for v, size, mult in zip(self._data.values(),
                                 self._σclass_sizes.values(),
                                 self._σclass_multiplicities.values()):
            if np.isscalar(v):
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
        for v, size in zip(self._data.values(), self._σclass_sizes.values()):
            if np.isscalar(v):
                yield from itertools.repeat(v, size)
            else:
                yield from v

    # def indep_iter_index(cls) -> Generator[Tuple[List[int],...]]:

    def indep_iter_repindex(self) -> Generator:
        for counts in utils._perm_classes(self.rank):
            yield from σindex_iter(counts, self.dim)  # Inlined permcls_indep_iter_repindex

    def permcls_indep_iter(self, σcls: Union[str, Tuple[int,...]]) -> Generator:
        if isinstance(σcls, str):
            σcls = self.permclass_label_to_counts(σcls)
        v = self._data[σcls]
        if np.isscalar(v):
            size = self._σclass_sizes[σcls]
            yield from itertools.repeat(v, size)
        else:
            yield from v

    # def permcls_indep_iter_index(cls, σcls: Union[str, Tuple[int]]
    #                             ) -> Generator[Tuple[List[int],...]]:

    def permcls_indep_iter_repindex(self, σcls: Union[str, Tuple[int]]
                                   ) -> Generator[Tuple[int]]:
        if isinstance(σcls, str):
            σcls = self.permclass_label_to_counts(σcls_label)
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

# TODO
#@PermClsSymmetricTensor.implements(np.einsum)
#def einsum(*operands, dtype=None, order='K', casting='safe', optimize=False):
#    # NB: Can't used the implementation in np.core.einsumfunc, because that calls
#    #     C code which requires true arrays
#    ...

# %% [markdown]
# ### Symmetrized operations
#
# These functions replace standard NumPy ones, making them symmetric.
#
# @SymmetricTensor.implements_ufunc.outer(np.add, np.sub, np.multiply)
# def symmetric_outer(self, other, ufunc=np.multiply):  # SYMMETRIC ALGEBRA
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
# def symmetric_tensordot(self, other, axes=2):  # SYMMETRIC ALGEBRA
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
            A = PermClsSymmetricTensor(rank=rank, dim=dim)
            # For dicts, getsizeof oly counts the number of keys/value pairs,
            # not the size of either of them.
            sym = sys.getsizeof(A)
            sym += sum(sys.getsizeof(k) + sys.getsizeof(v) for k,v in A.__dict__.items())
            sym += sum(sys.getsizeof(k) for k in A._data)
            sym += sum(A.get_permclass_size(σcls)*8 + 104 for σcls in A.perm_classes)
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
    display(fig)

# %% [markdown]
# ## Tests

# %%
if __name__ == "__main__":
    SymmetricTensor = PermClsSymmetricTensor  # This makes it easier to write reusable tests for different SymmetricTensor formats

    # %%
    import pytest
    from collections import Counter
    from statGLOW.utils import does_not_warn
    from statGLOW.stats.symtensor.symtensor.utils import symmetrize

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
# This test gives us good confidence that the methods `perm_classes` and `get_permclass_size` are correctly implemented.

    # %%
    for A in test_tensors():
        r = A.rank
        d = A.dim
        assert (sum(A.get_permclass_size(σcls) for σcls in A.perm_classes)
                == math.prod(range(d, d+r)) / math.factorial(r))

# %% [markdown]
# We now check if class multiplicities are correctly evaluated, to validate `get_permclass_multiplicity`. We do this by checking the identity
# $$\sum_\hat{σ} s_\hat{σ} = d^r\,.$$
# (Here $\hat{σ}$ is a permutation class and $s_{\hat{σ}}$ the size of that class.)

    # %%
    for A in test_tensors():
        r = A.rank
        d = A.dim
        assert (sum(A.get_permclass_size(σcls) * A.get_permclass_multiplicity(σcls)
                    for σcls in A.perm_classes)
                == d**r)

# %% [markdown]
# ### Iteration
#
# Test the index iterator, including examples given in the description of `σindex_iter`.

# %%
if __name__ == "__main__":
    assert list(σindex_iter((3,), 3))  == [(0,0,0), (1,1,1), (2,2,2)]
    assert list(σindex_iter((2,1), 2)) == [(0,0,1), (1,1, 0)]
    assert list(σindex_iter((2,1), 3)) == [(0,0,1), (0,0,2), (1,1,0), (1,1,2), (2,2,0), (2,2,1)]
    assert list(σindex_iter((2,2), 3)) == [(0,0,1,1), (0,0,2,2), (1,1,2,2)]

# %% [markdown]
# Test iteration.
#
# - `SymmetricTensor` gets initialized as a zero tensor, storing only one scalar per class.
# - Iteration still returns either $\binom{d + r - 1}{r}\,.$ or $d^r$ values (depending on whether it returns permutations of symmetric terms).
# - The `flat` iterators start by returning all diagonal components.
# - `permcls_indep_iter_repindex` returns a unique index tuple (index class representative)
# - `permcls_indep_iter_repindex` returns exactly one index $I$ for each index class, and each $I$ cannot be obtained from another via permutation.

    # %%
    for A in test_tensors():
        assert all(len(list(A.indep_iter_index(σcls))) == len(list(A.indep_iter(σcls)))
                   for σcls in A.perm_classes)
        assert len(list(A.indep_iter_index())) == len(list(A.indep_iter())) == A.size
        assert len(list(A.flat)) == len(list(A.flat_index)) == A.dim**A.rank
        # permcls_indep_iter_repindex returns a unique index tuple
        I = next(A.permcls_indep_iter_repindex())
        assert isinstance(I, tuple) and all(isinstance(i, int) for i in I)
        # permcls_indep_iter_repindex returns a unique index I for each index class
        len({str(Counter(sorted(idx))) for idx in A.permcls_indep_iter_repindex()}) == A.size

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
    sizes = [A.get_permclass_size(σcls) for σcls in A.perm_classes]
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
# ### Copying and Equality

# %%
if __name__ == "__main__":
    rank = 4
    dim = 50
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

# %% [markdown]
# ### Serialization

    # %%
    from statGLOW.smttask_ml import scityping
    class Foo(BaseModel):
        A: SymmetricTensor
        class Config:
            json_encoders = scityping.json_encoders  # Includes Serializable encoder
    foo = Foo(A=A)
    foo2 = Foo.parse_raw(foo.json())
    assert foo2.json() == foo.json()

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
# ### Arithmetic

# %%
import math

# %%
# %timeit math.prod(range(1, 100+1))
# %timeit np.prod(range(1, 100+1))

# %%
if __name__ == "__main__":

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
    TimeThis.on = False

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



    # %% tags=[]
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
# ### Contraction with matrix along all indices

# %%
if __name__ == "__main__":

    A = SymmetricTensor(rank = 3, dim=3)
    A[0,0,0] =1
    A[0,0,1] =-12
    A[0,1,2] = 0.5
    A[2,2,2] = 1.0
    A[0,2,2] = -30
    A[1,2,2] = 0.1
    A[1,1,1] =-0.3
    A[0,1,1] = 13
    A[2,1,1] = -6
    W = np.random.rand(3,3)
    W1 = np.random.rand(3,3)
    W2 = np.random.rand(3,3)
    assert np.isclose(A.contract_all_indices_with_matrix(W).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W))).all()
    assert np.isclose(A.contract_all_indices_with_matrix(W1).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1))).all()
    assert np.isclose(A.contract_all_indices_with_matrix(W2).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W2,W2,W2))).all()

    B = SymmetricTensor(rank = 4, dim =4)
    B['iiii'] = np.random.rand(4)
    B['ijkl'] =12
    B['iijj'] = np.random.rand(6)
    B['ijkk'] =-0.5
    W = np.random.rand(4,4)
    C = B.contract_all_indices_with_matrix(W)
    W1 = np.random.rand(4,4)
    W2 = np.random.rand(4,4)
    assert np.isclose(C.contract_all_indices_with_matrix(W).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W,W,W,W))).all()
    assert np.isclose(C.contract_all_indices_with_matrix(W1).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W1,W1,W1,W1))).all()
    assert np.isclose(C.contract_all_indices_with_matrix(W2).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W2,W2,W2,W2))).all()


# %% [markdown]
# ### Contraction with list of SymmetricTensors

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

        contract_1 = test_tensor.contract_tensor_list( tensor_list, n_times =1, rule ='all')
        contract_2 = test_tensor.contract_tensor_list( tensor_list, n_times =2, rule ='all')

        assert  np.isclose(contract_1.todense(), symmetrize(np.einsum('ija, akl -> ijkl', test_tensor.todense(), chi_dense))).all()
        assert  np.isclose(contract_2.todense(), symmetrize(np.einsum('iab, ajk, blm -> ijklm', test_tensor.todense(), chi_dense,chi_dense))).all()

# %% [markdown]
# ### Contraction with vector

# %%
if __name__ == "__main__":
    A = SymmetricTensor(rank = 3, dim=3)
    A[0,0,0] =1
    A[0,0,1] =-12
    A[0,1,2] = 0.5
    A[2,2,2] = 1.0
    A[0,2,2] = -30
    A[1,2,2] = 0.1
    x = np.random.rand(3)
    x1 = np.random.rand(3)
    x2 = np.zeros(3)
    assert np.isclose(A.contract_all_indices_with_vector(x), np.einsum('abc, a,b,c -> ', A.todense(), x,x,x))
    assert np.isclose(A.contract_all_indices_with_vector(x1), np.einsum('abc, a,b,c -> ', A.todense(), x1,x1,x1))
    #assert np.isclose(A.contract_all_indices_with_vector(x2), 0)


# %%
if __name__ == "__main__":
    print(A.contract_all_indices_with_vector(x2))

# %% [markdown]
# ## Timings

# %% [markdown]
# ### Slicing
# Some tests to see where slowness could come from:

# %%
if __name__=="__main__":
    TimeThis.on= True
    with TimeThis("check slicing speed"):
        D = A[0]


# %% [markdown]
# ### Outer product:

# %%
if __name__=="__main__":
    for rank in [3]:
        for dim in [50]:
            vect = SymmetricTensor(rank=1, dim=dim)
            vect['i'] = np.random.rand(dim)
            print('rank = ', rank)
            print('dim = ', dim)
            with TimeThis('pos_dict_creation'):
                x = pos_dict[rank,dim]
            with TimeThis('outer product'):
                # vect x vect x vect ... x vect
                A = vect.outer_product([vect,]*(rank-1))

# %% [markdown]
# ### Contractions

    # %%
    A = SymmetricTensor(rank = 3, dim=3)

    A[0,0,0] =1
    A[0,0,1] =-12
    A[0,1,2] = 0.5
    A[2,2,2] = 1.0
    A[0,2,2] = -30
    A[1,2,2] = 0.1

    W = np.random.rand(3,3)
    with TimeThis('permcls_indep_iter_repindex'):
        li = [[W[0,a] for a in σidx] for σidx in itertools.permutations((0,1,2))]
    W1 = np.random.rand(3,3)
    assert np.isclose(A.contract_all_indices_with_matrix(W).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W))).all()
    assert np.isclose(A.contract_all_indices_with_matrix(W1).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1))).all()

# %% [markdown]
# ## WIP
#
# *Ordering permutation classes.*
# At some point I thought I would need a scheme for ordering permutation classes (for implementing a hierarchy, where e.g. `'ijkl'` can be used as a default for `'iijk'`). I save it here in case it turns out to be useful after all.
#
# [![](https://mermaid.ink/img/eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCLigZ1cIl1cbiAgXG4gICAgc3R5bGUgTiBmaWxsOm5vbmUsIHN0cm9rZTpub25lIiwibWVybWFpZCI6eyJ0aGVtZSI6ImRlZmF1bHQifSwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)](https://mermaid-js.github.io/mermaid-live-editor/edit##eyJjb2RlIjoiZ3JhcGggVERcbiAgICBBW1wiQSA9IGlqa-KAplwiXSAtLT4gQ3t7XCJuQSA6PSAjIGRpZmZlcmVudCBpbmRpY2VzIGluIEE8YnI-bkIgOj0gIyBkaWZmZXJlbnQgaW5kaWNlcyBpbiBCPGJyPkUuZy4gaWlpaSA8IGlpampcIn19XG4gICAgQltcIkIgPSBpamvigKZcIl0gLS0-IENcbiAgICBDIC0tPnxuQSA8IG5CfCBEW0EgPCBCXVxuICAgIEMgLS0-fG5BID4gbkJ8IEVbQSA-IEJdXG4gICAgQyAtLT58bkEgPSBuQnwgRnt7XCJjQSA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQTxicj5jQiA6PSAjIGRpZmZlcmVudCBpbmRleCBjb3VudHMgaW4gQjxicj5FLmcuIGlpamogPCBpaWlqXCJ9fVxuICAgIEYgLS0-fGNBIDwgY0J8IEdbQSA8IEJdXG4gICAgRiAtLT58Y0EgPiBjQnwgSFtBID4gQl1cbiAgICBGIC0tPnxjQSA9IGNCfCBJe3tcIm1BIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBBPGJyPm1CIDo9IGxvd2VzdCBpbmRleCBjb3VudCBpbiBCPGJyPkUuZy4gaWlpamogPCBpaWlpalwifX1cbiAgICBJIC0tPnxtQSA8IG1CfCBKW0EgPCBCXVxuICAgIEkgLS0-fG1BID4gbUJ8IEtbQSA-IEJdXG4gICAgSSAtLT58bUEgPSBtQnwgTXt7XCJzZWNvbmQgbG93ZXN0IGluZGV4IGNvdW50XCJ9fVxuICAgIE0gLS0-IE5bXCJcdOKBnVwiXVxuICBcbiAgICBzdHlsZSBOIGZpbGw6bm9uZSwgc3Ryb2tlOm5vbmUiLCJtZXJtYWlkIjoie1xuICBcInRoZW1lXCI6IFwiZGVmYXVsdFwiXG59IiwidXBkYXRlRWRpdG9yIjpmYWxzZSwiYXV0b1N5bmMiOnRydWUsInVwZGF0ZURpYWdyYW0iOmZhbHNlfQ)
