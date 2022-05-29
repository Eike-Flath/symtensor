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
# # `DenseSymmetricTensor`

# %% [markdown]
# ## Situations where a `DenseSymmetricTensor` may be well suited
#
# Since a `DenseSymmetricTensor` simply wraps a normal NumPy array with the `SymmetricTensor` API, it requires as much memory as storing a dense array. Still, there are some circumstances where it may be useful:
#
# **Computing combinatorics**  
# ~ Number of equivalent permutations.  
# ~ Number of independent components within a *permutation class*.
#
# **Use as a reference implementation**
# ~ &nbsp;

# %%
from __future__ import annotations
import statGLOW.typing

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
import itertools
from reprlib import repr
from numbers import Number as Number_, Integral as Integral_
from typing import KeysView, ValuesView, ItemsView
import numpy as np

# %%
if exenv in {"jbook", "notebook"}:
    from statGLOW.stats.symtensor.symtensor.base import SymmetricTensor
    from statGLOW.stats.symtensor import utils
else:
    from .base import SymmetricTensor
    from . import utils

# %%
from typing import Generator
from statGLOW.smttask_ml.scityping import Number, Array, DType


# %% [markdown]
# ## Selected method dependencies
#
# A → B: A depends on B.
#
# [![](https://mermaid.ink/img/pako:eNqVVM2OwiAQfpWGs76Ahz1s3OtmE73JhiCMlqRAA0Nct-m7L4i1_rR29WBg5vuZGSZtiLASyILsKnsQJXdYrN-poaaIPx-2e8frsliC8bA6ag3olFjHm3UZUhS7iuP1mSkj4adp-nPbdvl0rZlCcMxBfQYOBHtGDU6LyrNB5pPkoOej4bTbuNWoz7hazoCRDxMOqCrfEVliMlFx76EP7muxoYTtAVlWjmlKvi95f36fX8iljZt9Bv117HimTi-1SX8XLSMnFFIzaG1fsrB6qwxHZY1nB4VleomKC9BgLusRQ9AvS2oiYKbcOaViivn87VxcF8p95URfYT_eU2ZgGW5RVyqvYAfnOyBwAk-O43E3TrwnGz3MmirwKeN1t5sm7zX_w0lbTGZERxhXMn53msSiBMs4GkoW8Shhx0OFlFDTRmioJUf4kAqtIwt0AWaEB7SroxHdPWOWisft1DnY_gEXxr-x)](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNqVVM2OwiAQfpWGs76Ahz1s3OtmE73JhiCMlqRAA0Nct-m7L4i1_rR29WBg5vuZGSZtiLASyILsKnsQJXdYrN-poaaIPx-2e8frsliC8bA6ag3olFjHm3UZUhS7iuP1mSkj4adp-nPbdvl0rZlCcMxBfQYOBHtGDU6LyrNB5pPkoOej4bTbuNWoz7hazoCRDxMOqCrfEVliMlFx76EP7muxoYTtAVlWjmlKvi95f36fX8iljZt9Bv117HimTi-1SX8XLSMnFFIzaG1fsrB6qwxHZY1nB4VleomKC9BgLusRQ9AvS2oiYKbcOaViivn87VxcF8p95URfYT_eU2ZgGW5RVyqvYAfnOyBwAk-O43E3TrwnGz3MmirwKeN1t5sm7zX_w0lbTGZERxhXMn53msSiBMs4GkoW8Shhx0OFlFDTRmioJUf4kAqtIwt0AWaEB7SroxHdPWOWisft1DnY_gEXxr-x)

# %% [markdown]
# ## Implementation

# %% [markdown]
# ### Indexing utilities

# %% [markdown]
# #### `get_index_representative`
# Each set of indices equivalent under permutation has one representative index; this is the index returned by `*repindex` iterators.
# For a `DenseSymmetricTensor`, this is given by sorting index values *lexicographically*.
#
# This function converts an arbitrary index into its class' representative.
# For example, given the input index `(2,1,2)`, it returns `(1,2,2)`.
#
# :::{note}
#
# Different conventions for index representatives may work best for different memory layouts, therefore each module defining a layout must also define a `get_index_representative` function. This function defines the convention for that layout.  
# :::

# %%
def get_index_representative(index: Tuple[int]) -> Tuple[int]:
    "Return the representative for the index class to which `index` belongs."
    return tuple(sorted(index))


# %% [markdown]
# ### `DenseSymmetricTensor`

# %% tags=["remove-output"]
class DenseSymmetricTensor(SymmetricTensor):
    """
    A `SymmetricTensor` storing its data as a single dense NumPy array.
    Provides no memory improvements over NumPy arrays, but exposes an API
    consistent with other `SymmetricTensors`.
    
    On creation, defaults to a zero tensor.
    """
    _data                : Union[Array]
        
    # _validate_data mostly depends on the data format – overridden in subclasses
    def _validate_data(self, data: Union[dict, "array-like"]) -> Tuple[Array, DType]:
        # DEVNOTE: Implementations in subclasses can assume that self.rank and
        #    self.dim are set to the value of arguments (but NOT self._dtype).
        #    They can also assume that `data` is not None.
        #    If both rank/dim and data are provided, subclasses SHOULD check
        #    that they are compatible, and raise ValueError otherwise.
        
        if isinstance(data, dict):
            # Special case: to have a standard mapping API, we emulate data stored as {(): array}
            if data.keys() == {()}:
                data = data[()]
            else:
                raise TypeError("Data should be passed as either a normal NumPy array, "
                                "or the dictionary {(): data} with a single NumPy array.\n"
                                f"Received: {repr(data)}")

        # Now that we've ensured that `data` is not a dict, make it an array, then inspect it to get dtype
        arraydata = self._validate_dataarray(data)
        datadtype = arraydata.dtype
        datashape = arraydata.shape

        # Return
        return arraydata, datadtype, datashape

    def _init_data(self, data, symmetrize: bool):
        # DEVNOTE: Implementations in subclasses can assume that
        #          self.rank, self.dim and self._dtype are available
        if data.dtype != self._dtype:  # Only cast if necessary
            data = data.astype(self._dtype)
        if data.shape == (self.dim,)*self.rank:
            self._data = data
        else:
            self._data = np.empty((self.dim,)*self.rank, dtype=self._dtype)
            self._data[:] = data
        if symmetrize:
            self._data = utils.symmetrize(self._data)
        elif not utils.is_symmetric(self._data):
            raise ValueError("Data are not symmetric.")

    ## Dunder methods ##

    def __getitem__(self, key):
        if isinstance(key, str):
            # repindex -> Only return one value per independent component
            return self._data[tuple(zip(*self.permcls_indep_iter_repindex(key)))]

        elif isinstance(key, tuple):
            new_data = self._data[key]
            if np.ndim(new_data) == 0:
                return new_data
            else:
                return DenseSymmetricTensor(
                    rank=new_data.ndim, dim=self.dim, data=self._data[key])
        else:
            return DenseSymmetricTensor(
                rank=self.rank-1, dim=self.dim, data=self._data[key])

    def __setitem__(self, key, value):
        if isinstance(key, str):
            if np.ndim(value) == 0:
                # If rank=3, then permcls_indep_iter_index yields 3-tuples of lists
                # zip(* …) then concatenates those lists, into one 3-tuple of longer lists
                self._data[tuple(zip(*self.permcls_indep_iter_index(key)))] = value
            else:
                for idx, v in zip(self.permcls_indep_iter_index(key), value):
                    self._data[idx] = v

        elif key == slice(None):
            self._data[:] = value
            if self.rank <= 2 or not utils.is_symmetric(self._data):
                # For rank ≤ 2, symmetrizing is generally faster than checking for symmetry
                utils.symmetrize(self._data, out=self._data)
        
        elif len(key) == self.rank and all(isinstance(i, Integral_) for i in key):
            # Fully specified index: All we need to do is assign to all symmetry-equivalent positions
            self._data[utils.symmetrize_index(key)] = value
            
        else:
            raise NotImplementedError(
                "DenseSymmetricTensor currently only supports the following index assignments:\n"
                "  A['iij'] = 5\n  A[0,1,2] = 5\n  A[:] = 5")
            # Note that we can't reuse indexing logic in self._data – i.e.
            # assign to _data[key] and symmetrize – since that would give
            # incorrect results: If I do ``A[0,1] = 6``, I expect that afterwards
            # ``A[0,1] == A[1,0] == 6``. Doing as above would assign instead 3
            # to both those values.

    def __iter__(self):
        return iter(self._data)

    ## Public attributes & API ##

    def todense(self) -> Array:
        if self._data is None:
            raise RuntimeError("No data was allocated for this SymmetricTensor")
        return self._data
    
    @property
    def size(self) -> int:
        return self.dim**self.rank

    ## Iterators ##

    def keys(self) -> KeysView:
        return ({():None} if hasattr(self, '_data') else {}).keys()
    def values(self) -> ValuesView:
        return ({():self._data} if hasattr(self, '_data') else {}).values()
    def items(self) -> ItemsView:
        return ({():self._data} if hasattr(self, '_data') else {}).items()

    @property
    def flat(self):
        """
        {{base_docstring}}

        .. Note:: The implementation of this method for `DenseSymmetricTensor`
           actually does follow the order of NumPy's `flat`.
        """
        return self._data.flat

    @property
    def flat_index(self):
        return np.ndindex(*self.shape)

    def indep_iter(self) -> Iterator:
        # DEVNOTE: Elementwise access of unique indices is >100x faster than
        # iterating over an ndindex and filtering to keep only lexicographically
        # ordered indices; see :doc:`docs/developers/SymmetricTensor/timings.py`
        A = self._data
        for idx in self.indep_iter_repindex():
            yield A[idx]

    #def indep_iter_index(self) -> Generator:

    def indep_iter_repindex(self) -> Iterator:
        return itertools.combinations_with_replacement(range(self.dim), self.rank)

    def permcls_indep_iter(self, σcls: str=None) -> Generator:
        A = self._data
        for idx in self.permcls_indep_iter_repindex(σcls):
            yield A[idx]

    # def permcls_indep_iter_index(self, σcls: str=None
    #    ) -> Generator[Tuple[List[int],...]]:

    def permcls_indep_iter_repindex(self, σcls: str=None
        ) -> Generator[Tuple[int]]:
        if isinstance(σcls, str):
            σcls = utils.permclass_label_to_counts(σcls)
        _get_permclass = utils._get_permclass
        for idx in self.indep_iter_repindex():
            if σcls == _get_permclass(idx):
                yield idx
        # Timings (laptop):
        # _get_permclass: 900–1700 ns / loop
        # combinations_with_replacement: 600 ns (initial call) + <1 ns / loop
        # indep_iter_repindex: 28 ns / loop (function overhead)
        # comparison (tuple): 90–105 ns / loop
        # comparison (str): 62 ns / loop

# %% [markdown]
# ### Array functions

# %%
# Implementations of symmetric algebra functions
#
# Inherited from SymmetricTensor:
# - (symmetric) outer (add, sub, multiply)
# - (symmetric) tensordot
# - contract_all_indices_with_matrix
# - contract_all_indices_with_vector
# - contract_tensor_list

# %%
if exenv in {"notebook", "jbook"}:
    from mackelab_toolbox.utils import TimeThis
    from statGLOW.stats.symtensor import symalg

    A = DenseSymmetricTensor(rank = 3, dim=3)

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
    assert np.isclose(symalg.contract_all_indices_with_matrix(A, W).todense(), utils.symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W))).all()
    assert np.isclose(symalg.contract_all_indices_with_matrix(A, W1).todense(), utils.symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1))).all()

# %% [markdown]
# ## Memory footprint
# Memory required to store tensors, relative to an equivalent dense NumPy array of the same shape.

# %% tags=["active-ipynb"]
# if exenv in {"notebook", "jbook"}:
#     import holoviews as hv
#     hv.extension('bokeh')
#     
#     fig = utils.compare_memory(
#         DenseSymmetricTensor,
#         ranks_dims = itertools.product(
#             [1, 2, 3, 4, 5, 6, 7, 8],
#             [1, 2, 3, 4, 6, 8, 10, 20, 40, 80, 100]),
#         dtype=np.float64
#     )
#     fig.select(rank=[1,3,5,7]).opts(hv.opts.Curve(muted=True))  # Make even rank curves more prominent
#     display(fig)
