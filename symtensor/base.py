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
# # Abstract base class for symmetric tensors

# %% tags=["remove-cell"]
from __future__ import annotations

# %% tags=["remove-input"]
from warnings import warn
import logging
from ast import literal_eval
from abc import ABC, abstractmethod
from collections import Counter, ChainMap
from dataclasses import dataclass
from numbers import Number as Number_
from functools import partial
import time
import itertools
from itertools import chain, combinations_with_replacement
import more_itertools
import math  # For operations on plain Python objects, math can be 10x faster than NumPy
import textwrap
import inspect

import numpy as np

from numpy.core.overrides import array_function_dispatch as _array_function_dispatch

from mackelab_toolbox.utils import total_size_handler

from typing import (Union, ClassVar, Any, Type, Iterator, Generator, KeysView,
                    Dict, List, Tuple, Set)
from scityping import Number, Serializable
from scityping.numpy import Array, DType
from scityping.pydantic import dataclass

# %% tags=["active-ipynb"]
# # Notebook only imports
# #from symtensor import utils #CM:Does not work for me.... 
# import utils as utils

# %% tags=["active-py"]
# Script only imports
from . import utils

# %%
__all__ = ["SymmetricTensor"]

# %% tags=["remove-input"]
logger = logging.getLogger(__name__)


# %% [markdown]
# ## Rationale
#
# If we naively store tensors as high-dimensional arrays, they quickly grow quite large. However, reality does not need to be so bad:
# 1. Since tensors must be symmetric, a fully specified tensor contains a lot of redundant information.
#
# We want to avoid storing each element of the tensor, while still being efficient with those elements we do store.

# %% [markdown]
# ## Usage hints
#
# The `SymmetricTensor` class provides an outward interface is similar to an array: `A[1,2,3]` retrieves the tensor component $A_{123}$.
#
# `SymmetricTensor` itself as an [*abstract base class*](https://docs.python.org/3/library/abc.html); it defines the public interface, but leaves the implementation to subclasses. Subclasses may use different memory layouts, or different backends for the underlying arrays (currently NumPy or PyTorch).
#
# | Layout →<br>Backend ↓ | Dense                       | σ-classes | Blocked arrays |
# |-----------------------|-----------------------------|---------------------|---|
# | Numpy                 | [`DenseSymmetricTensor`](./dense_symtensor.py)      | [`PermClsSymmetricTensor`](./permcls_symtensor.py) | ✘ |
# | [PyTorch](./torch_symtensor) <br>(supports GPU) |  `DenseTorchSymmetricTensor` | `TorchSymmetricTensor` | ✘ |
# | tlash[^tlash]                      | ✘                                   | ✘                                                   | (planned)      |
#
# Default implementations for a particular layout are usually written for the NumPy backends. Additional backends are supported via multiple inheritance; so for example, the definition of `TorchDenseSymmetricTensor` is to a good approximation just two lines:
#
# ```python
# class TorchDenseSymmetricTensor(DenseSymmetricTensor, TorchSymmetricTensor):
#    _data                : Union[TorchTensor]
# ```
#
# :::{caution} We occasionally use “format” as a synonym for “layout”, when we want to differentiate between the “layout” or “data alignment” of a tensor with specific rank and dimension, and a generic layout “format” with undetermined rank and dimension.
# :::
#
# This approach allows to support a multiple layouts and backends with minimal code repetition.
#
#
# [^tlash]: [GitHub](https://github.com/mdschatz/tlash)
# %% [markdown]
# ## Further development
#
# Not all functionality one would expect from an array-like object is yet available; features are implemented as they become needed.
#
# We plan to have a few different memory layout implementations, in order to determine which is most efficient for our needs. Possibilities include:
#
# - Blocked storage (using code from Martin Schatz)
# - Lexicographic storage

# %% [markdown]
# ## Notation summary
#
# (Reproduced from the [*Getting started* guide](../Getting_started.md). See that page for more details.)
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
# More examples:
#
# | $\hat{I}$ | $\hat{I}$ (str) | $l$ | $m$           | $n$     |
# |-----------|-----------------|-----|---------------|---------|
# |`(3,2)`    | `iiijj`         | 2   | 0, 1, 1, 0, 0 | 3, 2    |
# |`(1,1,1)`  | `ijk`           | 3   | 3, 0, 0       | 1, 1, 1 |
#
# NB: In code, we usually just write $σ$ instead of $\hat{σ}$.

# %% [markdown]
# ### Identities
#
# - $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} γ_{\hat{σ}} = d^r$
# - $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} = \binom{d + r - 1}{r}$
# - $\displaystyle s_{\hat{σ}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $m_n$ is the number of different indices which appear $n$ times.
# - $\displaystyle γ_{\hat{σ}} = \binom{r}{m_1,m_2,\dotsc,m_r} = \frac{r!}{n_1!n_2!\dotsb n_l!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $n_k$ is the number of times index $\hat{I}_k$ appears.

# %% [markdown]
# ### Storage
#
# (Implementation specific)
#

# %% [markdown]
# ## Package dependencies
#
# :::{figure-md}  package-deps-symtensor
#
# ![Dependencies between symalg, dense_symtensor, permcls_symtensor, base, utils.](https://mermaid.ink/img/pako:eNpdj0sOgzAMRK-CvIYLsOiiak_Q7khVucRApHxQ4qhCiLs3IFLRbuzJ-MkTz9A6SVBDp927HdBzcb4LWxQvDLR2STZQs9VnmAwn4fxjnYzkTatDs_e_aXqh7lcVWekgbF5aVNUpexnbvJx4sLbY7y9-qD314EEJJpmoZDpnXhkBPJAhAXWSkjqMmgUIuyQ0jhKZrlKx81B3qAOVgJHdbbIt1OwjZeiisPdodmr5AL6nank)
#
# `A --> B` means that `A` depends on `B`.
# [[Edit]](https://mermaid-js.github.io/mermaid-live-editor/edit#pako:eNpdj0sOgzAMRK-CvIYLsOiiak_Q7khVucRApHxQ4qhCiLs3IFLRbuzJ-MkTz9A6SVBDp927HdBzcb4LWxQvDLR2STZQs9VnmAwn4fxjnYzkTatDs_e_aXqh7lcVWekgbF5aVNUpexnbvJx4sLbY7y9-qD314EEJJpmoZDpnXhkBPJAhAXWSkjqMmgUIuyQ0jhKZrlKx81B3qAOVgJHdbbIt1OwjZeiisPdodmr5AL6nank)
#
# :::

# %% [markdown]
# -------------------------------------------------
#
# ## Implementation

# %% [markdown]
# > **Note**: There is a bijective map between string representations of permutation classes – `'iijk'` – and count representations – `(2,1,1)`. Public methods of `SymmetricTensor` use strings, while private methods use counts.
#
# > **Note**: The underlying `_data` object must either be an array, or a dictionary of arrays. Or at least, they must support views, so that they may be used as the 'out' parameter to ufuncs.

# %% [markdown]
# ### `SymmetricTensor`
#
# **Public attributes**
# - *dim*  → `int`
# - *dtype* → (e.g.) `dtype('float64')`
# - *rank* → `int`
# - *perm_classes* → (e.g.) `['iii', 'iij', 'ijk']`
# - *shape* → $(d,d,\dotsc)$
# - *data_format*  → (e.g.) `"PermCls"`
# - *data_alignement*  → (e.g.) `"PermCls_3_4"`
# - *size*  → `int` (number of allocated components)
# - *indep_size*  → `int` (number of independent components)
# - *dense_size*  → `int` (number of components of the dense tensor)
# - *flat*       → Iterator over all components, including symmetric equivalents. See also *indep_iter*.
# - *flat_index* → Indices aligned with *flat*.
#
# **Public methods**
# - *keys()*: Return an iterator yielding keys matching how data is stored internally.
# - *values()*: Return an iterator yielding the data, exactly how it is stored. *The returned object can be modified in place to update the tensor.*
# - *items()*: Return an iterator yielding (key, value) tuples.
# - *todense()*
# - *indep_iter()*     : Iterator over independent components (i.e. *excluding* symmetric equivalents).
# - *indep_iter_index()*     : Indices aligned with *indep_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
# - *permcls_indep_iter(σcls: str)*     : Iterator over independent components (i.e. *excluding* symmetric equivalents) within a permutation class.
# - *index_permcls_iter(σcls: str)*     : Indices aligned with *permcls_indep_iter*. Each index includes all symmetric components, such that equivalent components of a dense tensor can be set or retrieved simulatneously.
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
# - Partial or sliced indexing is not fully supported.
#   This could certainly be done, although it would require some indexing gymnastics
# - Arithmetic operations are not currently supported.
#   Elementwise operations between tensors of the same size and rank can be trivially implemented when the need arises; other operations (like `.dot`) should be possible with some more work.

# %% [markdown]
# :::{hint}
# A good rule of thumb is the following:
# - *Concrete subclasses* of `SymmetricTensor` implementing a new *data format* should (re)define *abstract* methods of `SymmetricTensor`;
# - *Abstract subclasses* of `SymmetricTensor` implementing a new *array backend* should redefine some of the *concrete* methods of `SymmetricTensor`.
#
# :::

# %% [markdown]
# :::{admonition} List of abstract methods
#
# For convenience, here is the list `SymmetricTensor`'s abstract methods and attributes:
# - *data_format*  (class attribute)
# - *_validate_data*
# - *_init_data*
# - *\_\_getitem\_\_*
# - *\_\_setitem\_\_*
# - *size*
# - *todense*
# - *keys*
# - *values*
# - *flat*
# - *flat_index*
# - *indep_iter*
# - *indep_iter_repindex*
# - *permcls_indep_iter*
# - *permcls_indep_iter_repindex*
#
# :::

# %% [markdown]
# :::{margin} `implements_ufunc` interface
# :::

# %% tags=["hide-input"]
@dataclass
class HandledUfuncsInterface:
    """
    Register an ufunc implementation for SymmetricTensor objects.

    Examples
    >>> @SymmetricTensor.implements_ufunc.outer(np.add, np.multiply)
        def symmetric_outer(ufunc, a, b, **kwargs):
            ...
    >>> @PermClsSymmetricTensor.implements_ufunc(np.dot)  # Sets the "__call__" variant
        def symmetric_dot(a, b, **kwargs):
            ...
    >>> PermClsSymmetricTensor.does_not_implement_ufunc(np.modf)

    .. Important:: When the decorator is used for more than one ufunc (like in
       the first example above), its first argument should be named ``ufunc``.
       This argument will receive the base ufunc (``add``, ``multiply``, etc.).
       When the decorator is used with only one ufunc, having a ``ufunc``
       argument is optional.
    """
    cls: type
    implements: bool=True
    _methods: ClassVar[str]=["reduce", "accumulate", "reduceat", "outer", "at", "__call__"]
    def __dir__(self):
        attrs = super().__dir__()
        return attrs + [m for m in self._methods if m not in attrs]
    def __call__(self, *ufuncs):
        return self.__getattr__("__call__")(*ufuncs)
    def __getattr__(self, attr):
        if attr in self._methods:
            return self.get_ufunc_registry_interface(attr)
        else:
            raise AttributeError(f"{type(self)} does not define an attribute '{attr}'.")
    def get_ufunc_registry_interface(self, method):
        "Return a function which can add handlers for `method` for 1 or many ufuncs."
        # 'method': __call__, accumulate, outer, ...
        # 'ufunc': add, mean, log1p, ...
        def ufunc_registry_interface(*ufuncs):
            if self.implements:
                def decorator(func):
                    sig = inspect.signature(func)
                    if 'ufunc' in sig.parameters:
                        if next(iter(sig.parameters)) == "ufunc":
                            # ufunc is the 1st parameter
                            def bind(f, ufunc):
                                return partial(f, ufunc)
                        else:
                            logger.warning("Ufunc handlers should have 'ufunc' as their first argument.")
                            def bind(f, ufunc):
                                return partial(f, ufunc=ufunc)
                    elif len(ufuncs) > 1:
                        raise TypeError("When adding a handler for multiple ufuncs at the same time, "
                                        "the handler’s first argument must be 'ufunc'.")
                    else:
                        def bind(f, ufunc):
                            return f
                    for ufunc in ufuncs:
                        self.cls._HANDLED_UFUNCS[method][ufunc] = bind(func, ufunc)
                    return func
                return decorator
            else:
                for ufunc in ufuncs:
                    self.cls._HANDLED_UFUNCS[method][ufunc] = None
        return ufunc_registry_interface

# %% [markdown]
# :::{margin} abstract class `SymmetricTensor`
# :::

# %%
class SymmetricTensor(Serializable, np.lib.mixins.NDArrayOperatorsMixin, ABC):
    # NDArrayOperatorsMixin adds methods like __add__, __ge__, by using __array_ufunc_. See https://numpy.org/devdocs/reference/generated/numpy.lib.mixins.NDArrayOperatorsMixin.html

    # Registry used for __array_function__ protocol. Each subclass creates a
    # child ChainMap of its parent's registry, so that it is also searched.
    # Thus only newly supported or unsupported functions need to be added.
    _HANDLED_FUNCTIONS: ClassVar[ChainMap] = ChainMap()
    # Registry used for __array_ufunc__ protocol
    # For the list of ufunc methods, see https://numpy.org/doc/stable/reference/generated/numpy.ufunc.html
    _HANDLED_UFUNCS: ClassVar[Dict[str,ChainMap]] = {
        "__call__": ChainMap(), "reduce": ChainMap(), "reduceat": ChainMap(),
        "outer": ChainMap(), "at": ChainMap()}
    # DEVNOTE: Most ufuncs have signature ()->() or (),()->(), and for those,
    #   their default '__call__' method is already supported.
    #   Functions only need to be added to _HANDLED_UFUNCS if:
    #   - They have a different signature and we want to support them.
    #   - We want to support other methods, like 'outer' or 'reduce'
    #   - We want to override the default implementation
    #   - We want to explicitely *disallow* that ufunc.
    #     This can be done by defining it as `None`.

    rank        : int
    dim         : int
    _dtype      : DType
    data_format : ClassVar[str]="None"
    array_type  : ClassVar[type]=np.ndarray  # Type for undelying arrays. Typically changed by abstract subclasses to change the backend. At present not really used, but since we anticipate it will be useful, we preemptively standardize its name

    # DEVNOTE: Subclasses must define the following class annotations:
    # _data                : Union[Array, Number]

    def __init__(self, rank: Optional[int]=None, dim: Optional[int]=None,
                 data: Union[Array, Number]=np.float64(0),
                 dtype: Union[str,DType]=None,
                 symmetrize: bool=False):
        """
        Parameters
        ----------
        rank:
        dim:  Rank and dimension of the tensor.
            If `data` is provided, a ValueError is raised if it is incompatible
            with the given `rank` and `dim.

        data: May take one of the following forms:
            - None: Skip initializing _data altogether; don't even allocate `_data`.
                Intended to allow subclasses to initialize their own data.
            - Scalar: Created array will be filled with this value.
            - Array-like, which is broadcastable to the shape of the tensor.
            - Mapping (key: array-like), matching the underlying data format.

        dtype: If both `data` and `dtype` are provided, the dtype of the former
           should match the latter.
           If only `data` is provided, dtype is inferred from the data.
           If only `dtype` is provided, it determines the data dtype.

        symmetrize: If data is provided, whether to symmetrize it.
           If this is `False` and the data are not symmetric, subclasses
           should raise a `ValueError`.

        .. Note:: `rank` and `dim` may be inferred from `data`, but only when
           it is array-like. Rank and dim are not inferred from scalar data.
           In particular, **true scalars and 0d arrays are treated differently**:
           only the latter allow inferring `rank`. To infer `dim`, data must be
           at least 1d.

        Raises
        ------
        TypeError:
          - If `rank` or `dim` are not provided.
          - If data dtype is neither numeric nor bool.
        ValueError:
          - If `data` format is unexpected.
          - If `rank` or `dim` don’t match `data`.
        """
        self.rank = rank
        self.dim = dim

        # Validate data
        if data is None:
            datadtype = None
            datashape = None
        elif np.isscalar(data):  # True only for true scalars, but not 0d arrays
            data, datadtype, _ = self._validate_data(data)
            if dtype is None: dtype     = datadtype  # self._dtype is set below
        else:
            if isinstance(data, dict):
                # If `data` is a dict, it may be serialized data, in which case
                # the keys will be strings (see `Data.json_encoder`)
                data = {literal_eval(k) if isinstance(k, str) else k: v  # By design, `literal_eval` is relatively s
                        for k, v in data.items()}
            data, datadtype, datashape = self._validate_data(data)
            assert isinstance(datashape, tuple) and all(isinstance(s, int) for s in datashape), f"{type(self).__qualname__}._validate_data did not return data shape as expected."

            ## Consistency checks and inferences requiring `data` be passed ##

            # Check that array data can be cast to a square tensor
            if len(set(datashape) - {1, 0}) > 1:  # Number of size values, except 1 – to be square-broadcastable, all these values must be the same
                raise ValueError(f"Data of shape {datashape} cannot be broadcast "
                                 "to a symmetric array: all axes must have the same "
                                 "size (or 1).")

            # Infer data dim and rank from data shape
            datarank = len(datashape)
            if datarank == 0:  # Scalar data
                datadim = 0  # Could also use `None` as sentinel, but this allows `datadim` to be used in a comparison
            else:
                datadim = max(datashape)

            # Check that rank, dim & data are consistent
            if rank is not None and dim is not None:
                target_shape = (dim,)*rank
            elif rank is not None:
                if rank < datarank:
                    raise ValueError("Data has more dimensions that the specified rank: "
                                     f"Cannot create a rank {rank} tensor with "
                                     f"data that has rank {datashape}.")
                if datadim:
                    target_shape = (datadim,)*rank
                else:
                    target_shape = None
            elif dim is not None:
                if datadim > 1 and datadim != dim:
                    raise ValueError(f"Cannot create a tensor of dimension {dim}: "
                                     "data dimension differs from the specified dimension. "
                                     f"Data shape: {datashape}")
                target_shape = None
            else:
                target_shape = None

            if target_shape is not None:
                err = False
                try:
                    broadcasted_shape = np.broadcast_shapes(datashape, target_shape)
                except ValueError:
                    err = True
                else:
                    if broadcasted_shape != target_shape:
                        err = True
                if err:
                    raise ValueError(f"Cannot broadcast data (shape: {datashape}) "
                                     f"to shape {target_shape}.")

            # Set rank, dim & dtype based on args & inferred values
            if self.rank is None: self.rank = datarank
            if self.dim is None and datarank >= 1 : self.dim = datadim  # With 0-rank, everything is a scalar, no matter dim => can’t infer dim
            if dtype is None: dtype     = datadtype  # self._dtype is set below

        # Ensure that 'rank' and 'dim' are set
        def missing_error(origval, valname):
            raise TypeError(f"'{valname}' was not provided to {type(self).__qualname__}, "
                            "and it was not possible to infer it from `data`.")
        if self.rank is None: missing_error(rank, "rank")
        if self.dim is None: missing_error(dim, "dim")

        # Set σ-classes
        self.perm_classes = tuple(utils.permclass_counts_to_label(counts)
                                  for counts in utils._perm_classes(self.rank))

        # Set dtype
        self._set_dtype(dtype)

        # Set data
        if data is not None:
            self._init_data(data, symmetrize)  # Uses the set values self.rank, self.dim, self._dtype

    # _validate_dataarray mostly depends on the backend – overridden in abstract subclasse
    def _validate_dataarray(self, array: "array-like") -> Array:
        """
        Given the data for a single underlying data object, validate it.
        Typically this means casting to `ndarray`, and checking that is it
        either of numeric or bool type.
        """
        # Cast to array if necessary
        if not isinstance(array, np.ndarray):
            # Reproduce the same range of standardizations NumPy has: Python bools & ints, NumPy types, tuples, lists, etc.
            array = np.asanyarray(array)

        # Validate dtype
        if array.dtype == object:
            raise TypeError(f"Initialization of {type(self).__qualname__} doesn’t "
                            f"support arguments of type {type(array)}.")            
        elif not any((np.issubdtype(array.dtype, np.number),
                     np.issubdtype(array.dtype, bool))):
            raise TypeError(f"Data type is neither numeric nor bool: {arra.dtype}.")
        
        return array

    # _validate_data mostly depends on the data format – overridden in concrete subclasses
    @abstractmethod
    def _validate_data(cls, data: Union[dict, "array-like"]) -> Tuple[Any, DType, Tuple[int,...]]:
        # DEVNOTE: In subclasses, replace 'Any' by the type(s) actually returned
        """
        Do five things:
        - Convert `data` from any of the expected formats to a standard one.
          This may include JSON deserialization.
        - Raise `ValueError` if `data` is in an unexpected format,
          or `TypeError` if its dtype is neither numeric nor bool.
        - Ensure that the returned `data` is symmetric.
        - Try to infer the dtype from the data.
        - Try to infer the shape from the data.

        Returns: standardized data, inferred dtype, inferred shape
        Raises: ValueError, TypeError

        .. Note:: `data` can be assumed ≠ 'None'; that case is already
           treated in `__init__`.
        """
        # DEVNOTE: Implementations in subclasses can assume that self.rank and
        #    self.dim are set to the value of arguments (but NOT self._dtype).
        #    They can also assume that `data` is not None.
        #    If both rank/dim and data are provided, subclasses SHOULD check
        #    that they are compatible, and raise ValueError otherwise.
        # DEVNOTE: This method should call `_validate_dataarray` one or more times
        raise NotImplementedError
        
    def _set_dtype(self, dtype: Optional[DType]):
        """
        Set the `self._dtype` attribute based on `dtype`, which may have been
        inferred from data or passed as argument.
        If no `dtype` is given, set it to the default 'float64'.
        
        This method is overridden e.g. by TorchSymmetricTensor to store a Torch
        dtype instead of a NumPy one.
        """
        if dtype is None:
            dtype = np.dtype('float64')
        else:
            dtype = np.dtype(dtype)
        self._dtype = dtype

    @abstractmethod
    def _init_data(self, data, symmetrize: bool):
        """
        Cast the data to ``self._dtype`` and assign to the tensor's ``_data``
        attribute, based on the initialization data `data`.
        Works along with `_validate_data`: `data` can expect to be in the
        standard format returned by `_validate_data`.
        """
        # DEVNOTE: Implementations in subclasses can assume that self.rank,
        #   self.dim and self._dtype are available, and that data is not None
        raise NotImplementedError

    #### Serialization ####
    @dataclass
    class Data:
        rank: int
        dim: int
        # NB: JSON keys must be str, int, float, bool or None, but not tuple => convert to str
        data: Dict[str, Array]
        
        @staticmethod
        def encode(symtensor: SymmetricTensor): 
            return (symtensor.rank, symtensor.dim, {str(k): v for k,v in symtensor.items()})
        @staticmethod
        def decode(data: "SymmetricTensor.Data"):
            # Invert the conversion tuple -> str that was done in `encode`
            data_dict = {tuple(int(key_str) for key_str in re.findall(r"\d+", s)): arr
                         for key_str, arr in data.data.items}
            # Instantiate the expected tensor
            return SymmetricTensor(rank, dim, data_dict)

    #### Subclassing magic ####

    # - Use parent's method docstring if the child's docstring is None
    # - Implement the '{{base_docstring}}' instruction, so derived
    #   classes don't need to repeat the whole docstring just to add a note.
    # - Perform programmatic correctness checks

    def __init_subclass__(cls, *args, **kwargs):
        # Update docstrings
        def get_base_doc(attr: str, cls: type) -> str:
            for C in cls.mro()[1:]:
                basedoc = getattr(getattr(C, nm, None), "__doc__", None)
                if basedoc is not None:
                    return basedoc
            return ""
        for nm, obj in cls.__dict__.items():
            if (not inspect.isfunction(obj)  # NB: On the base class, these are functions, not methods
                and not isinstance(obj, property)
                and not hasattr(obj, "__doc__")
                and not getattr(obj, nm, None) is obj):
                continue
            if obj.__doc__ is None:
                # Replace with the doc of the first parent which provides one.
                try:
                    obj.__doc__ = get_base_doc(nm, cls)
                except AttributeError:  # If a class has no docstring, it is read-only and can't be overwritten
                    pass
            elif "{{base_docstring}}" in obj.__doc__:
                obj.__doc__ = obj.__doc__.replace(
                    "{{base_docstring}}", get_base_doc(nm, cls))
        # Perform additional correctness checks
        if "_data" not in cls.__annotations__ and not inspect.isabstract(cls):
            raise RuntimeError(f"Class {cls} does not define '_data' in its class annotations.")
        # Add class-level attributes which require an already instantiated class
        cls.implements_ufunc = HandledUfuncsInterface(cls)
        cls.does_not_implement_ufunc = HandledUfuncsInterface(cls, implements=False)
        # Create child ChainMaps, so function registries can be updated without changing those of the parents
        cls._HANDLED_FUNCTIONS = cls._HANDLED_FUNCTIONS.new_child()
        cls._HANDLED_UFUNCS = {method: m.new_child()
                               for method, m in cls._HANDLED_UFUNCS.items()}
        # Pass control to parent __init_subclass__
        super().__init_subclass__()

    #### Dunder methods ####

    def __str__(self):
        s = f"{type(self).__qualname__}(rank: {self.rank}, dim: {self.dim})"
        return s

    def __repr__(self):
        s = f"{type(self).__qualname__}(rank: {self.rank}, dim: {self.dim}, data: "
        d = str(getattr(self, '_data', None))
        data_s = chain.from_iterable(line.split("\n") for line in textwrap.wrap(
            d, replace_whitespace=False, drop_whitespace=False))  # Keep newlines already in the formatted data, so arrays (which already contain well-placed line breaks) appear nicely
        # d1, d = str(self._data).split("\n", 1)
        # data_s = f"data: {d1}\n" + textwrap.indent(d, " "*6)
        data_s = "\n      " + "\n      ".join(data_s)
        return f"{s}{data_s})"

    @abstractmethod
    def __getitem__(self, key):
        """
        Three paths:
        - If `key` is a string (e.g. `'iij'`), return the array of
          *independent components* corresponding to that permutation class.
        - If `key` is a tuple of length equal to the rank, treat it as a tuple
          from a dense array and return the corresponding value.
        - If 'key' is an int or a tuple of length shorter than the rank,
          return a SymmetricTensor C of rank= self.rank - len(key) such that
          C.todense() = self.todense()[key,:,...,:]
        """
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, key, value):
        """
        2+1 paths:
        - If `key` is a string (e.g. `'iij'`), set the corresponding values
          in the permutation class.
          from a dense array and set the corresponding value.
        - If `key` is a tuple of length equal to the rank, treat it as a tuple
          from a dense array and set the corresponding value.
        - If 'key' is an int or a tuple of length shorter than the rank:
          Treat it as a tuple from a dense array and set the corresponding values
          This is both the trickiest code path and the least commonly used, so
          not all subclasses may implement it.
        """
        raise NotImplementedError

    def __iter__(self):
        """
        Return an iterator which yields `SymmetricTensor`s of rank r-1.
        """
        for i in range(self.dim):
            yield self[i]

    #### Translation functions ####
    # Mostly used internally, but part of the public API

    def copy(self) -> SymmetricTensor:
        """
        Return a copy of the current tensor
        """
        return self.__class__(dim = self.dim, rank = self.rank,
                              data = {k: arr.copy() for k, arr in self.items()})

    #### Public attributes & API ####

    @property
    def dtype(self) -> np.dtype:
        return self._dtype

    @property
    def shape(self) -> Tuple[int,...]:
        return (self.dim,)*self.rank

    @property
    def ndim(self) -> int:
        return self.rank
    
    @property
    def data_alignment(self) -> Tuple[str,int,int]:
        """
        Return a tuple which can be used to check whether two SymmetricTensors
        are memory-aligned.
        If the tuples compare equal, than element-wise operations can be
        performed directly on the tensor’s underlying memory.
        
        The default implementation returns
        
            (data_format, rank, dim)
        
        Subclasses may return a different format. For example,
        `PermClsSymmetricTensor` allows additional compression by storing
        an entire permutation class as a scalar. Since the underlying data are
        separate arrays (one per class), operations can still be performed, so
        the format above is used. However, the planned variation
        `PermClsFlatSymmetricTensor` would store everything as a single flat
        array, and would need to prevent operations between tensors with
        different compression.
        """
        return (self.data_format, self.rank, self.dim)

    @property
    @abstractmethod
    def size(self) -> int:
        """
        Return the maximum number of allocated memory elements, typically the
        sum of sizes of the underlying memory arrays. Analogous to
        `ndarray.size`. The actual amount of allocated memory may be less,
        if repeated component values are stored as a single scalar, or if
        no data was allocated at all.

        A good way to check whether it's safe to instantiate an array is to
        first create it with `data=None` and check its `size` attribute.
        Size values over 1 million should be considered risky, and values over
        10 million will exceed memory 16 GB machines.
        """
        # NB: Don't implement as simply sum(A.size for A in self.values()), because actually allocated data may be less
        raise NotImplementedError

    @property
    def dense_size(self) -> int:
        """
        Return the number of elements of the corresponding dense array.

        Equivalent to ``self.todense().size``, but without the potentially
        disastrous memory requirements.
        """
        return self.dim**self.rank

    @property
    def indep_size(self) -> int:
        """
        Return the number of independent components in the tensor.
        Subclasses may choose to override this if they implement additional
        symmetries.

        .. Note:: This may overestimate the memory footprint (if multiple
           independent components are stored as a shared scalar) or
           underestimate it (if there are redundant components).
        """
        return math.comb(self.dim + self.rank - 1, self.rank)
        # Timings: (rank, dim)
        # (8, 1000): 1 μs,  (8, 1 000 000): 1.24 μs

    @abstractmethod
    def todense(self) -> Array:
        raise NotImplementedError

    #### Iterators ####

    @abstractmethod
    def keys(self) -> KeysView:
        """
        Return a `KeysView` on the keys of the underlying data (each key
        corresponds to one flat array, as it is stored in data).
        If the data are stored as a single array, these consist of a single
        empty tuple ``()``.
        If no data are stored at all (i.e. if the tensor was created with
        ``data=None``), returns an empty set of keys.
        """
        # DEVNOTE: The returned object must behave as a `KeysView`, in
        #    particular, it must be comparable with other KeysViews.
        #    If data are not stored as a dictionary, the easiest way to do
        #    this is to construct a throwaway dict, and return its `.keys()`.
        raise NotImplementedError
    @abstractmethod
    def values(self) -> Iterator:
        """
        Return an iterator yielding the underlying data arrays.
        This can be used to write efficient generic operations; for example,
        elementwise scalar functions can always be applied directly to the
        underlying data.
        If no data are stored at all (i.e. if the tensor was created with
        ``data=None``), returns an empty iterator.
        """
        raise NotImplementedError
    def items(self) -> Iterator:
        """
        Returns keys and values, matching how they are stored internally.
        If no data are stored at all (i.e. if the tensor was created with
        ``data=None``), returns an empty iterator.
        """
        return zip(self.keys(), self.values())

    @property
    @abstractmethod
    def flat(self):
        """
        Return an iterator which returns each tensor component.
        This is meant to be equivalent to calling `flat` on a NumPy dense
        array, so components are repeated as many times as they have symmetries.
        The order of elements will in general different NumPy's `flat` (it is
        chosen for iteration efficiency), but the resulting generator can be
        zipped with `flat_index` to get associated indices.
        """
        raise NotImplementedError
    @property
    @abstractmethod
    def flat_index(self):
        """
        Return an iterator which yields the index of each tensor component
        exactly once. Each possible index permutation is returned separately.
        Can be zipped with `flat` to also get the component values.
        """
        raise NotImplementedError

    @abstractmethod
    def indep_iter(self) -> Iterator:
        """
        Return a generator which yields values for the all independent
        components, in the order in which they are stored, as a flat vector.
        One value per independent component is returned.
        """
        raise NotImplementedError

    def indep_iter_index(self) -> Generator:
        """
        Return a iterator which yields all indices, in the order in which they
        are stored as a flat vector. Equivalent permutations are returned
        together, as a single “advanced index” tuple, such that they can be used
        to simultaneously get or set values in a dense tensor.

        .. Note:: To construct the advanced index, permuted indices are
           collated. So for example, the index `(0,1,2)` has six permutations,
           which are collated as
           `([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [2, 1, 2, 0, 1, 0])`
           (For more information on the integer array indexing, see section in the
           `NumPy docs <https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing>`_.)
        """
        symmetrize_index = utils.symmetrize_index
        for index in self.indep_iter_repindex():
            yield symmetrize_index(index)
    @abstractmethod
    def indep_iter_repindex(self) -> Iterator:
        """
        Return a iterator which yields representative indices, one per
        independent component.
        Can be zipped with `indep_iter` or `indep_iter_index`.
        """
        return combinations_with_replacement(range(self.dim), self.rank)

    @abstractmethod
    def permcls_indep_iter(self, σcls: str=None) -> Generator:
        """
        Return a generator which yields values for the independent components
        in the class associated to `σcls`, in the order in which they
        are stored as a flat vector.

        Parameters
        ---------
        σcls: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: Can be combined with `indep_iter_index`, `permcls_indep_iter_repindex` and
           `permcls_multiplicity_iter`.
        """
        raise NotImplementedError

    def permcls_indep_iter_index(self, σcls: Union[str, Tuple[int]]
        ) -> Generator[Tuple[List[int],...]]:
        """
        Return a generator which yields all the indices in the class associated
        to `σcls`, in the order in which they are stored as a flat vector.
        Equivalent permutations are returned together, as a single “advanced
        index” tuple, such that they can be used to simultaneously get or set values
        in a dense tensor.

        Parameters
        ---------
        σcls: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: To construct the advanced index, permuted indices are
           collated. So for example, the index `(0,1,2)` has six permutations,
           which are collated as
           `([0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1], [2, 1, 2, 0, 1, 0])`
           (For more information on the integer array indexing, see section in the
           `NumPy docs <https://numpy.org/doc/stable/reference/arrays.indexing.html#advanced-indexing>`_.)
        """
        symmetrize_index = utils.symmetrize_index
        for index in self.permcls_indep_iter_repindex(σcls):
            yield symmetrize_index(index)

    @abstractmethod
    def permcls_indep_iter_repindex(self, σcls: Union[str, Tuple[int]]
        ) -> Generator[Tuple[int]]:
        """
        Return a generator which yields one representative for each index class
        *I* in the permutation class given by `σcls`, in the order in
        which they are stored as a flat vector.

        Parameters
        ---------
        σcls: (Optional)
           Permutation class over which to iterate. If no class is specified,
           iterate over all classes.

        .. Note:: In contrast to `indep_iter_index`, this does not return all
           equivalent permutations of a index. In general, `permcls_indep_iter_repindex`
           is more suited to operations involving only symmetric tensors,
           while `indep_iter_index` is more suited to operations involving also
           dense tensors.
        """
        raise NotImplementedError

    def permcls_multiplicity_iter(self):
        """
        Return the multiplicity of each class.
        In contrast to simply iterating over `self.perm_classes` and calling
        `get_permclass_multiplicity`, multiplicities are returned per *index*
        class (i.e., they line up with independent components).
        This makes this iterator appropriate for zipping with `indep_iter`
        and `indep_iter_index`.
        """
        for σcls in utils._perm_classes(self.rank):
            γ = utils._get_permclass_multiplicity(σcls, self.dim)
            s = utils._get_permclass_size(σcls, self.dim)
            yield from itertools.repeat(γ, s)

    ## Array creation, copy, etc. ##

    def __array__(self, dtype=None):  # C.f. ndarray.__array__'s docstring
        warn(f"Converting a SymmetricTensor to a dense NumPy array of shape {self.shape}.")
        return np.array(self.todense(), dtype=None)  # Returns a new reference if dtype is not modified

    def astype(self, dtype, order, casting, subok=True, copy=True):
        new_data = {k: v.astype(dtype, order, casting, subok, copy) if not np.isscalar(v)
                       else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                       else v.astype(dtype)
                    for k, v in self.items()}
        if all(orig_arr is new_arr for orig_arr, new_arr
               in zip(self.values(), new_data.values())):
            # We received copy=False & all sub arrays were successfully not copied
            return self
        else:
            return type(self)(self.rank, self.dim, data=new_data)

    ## __array_function__ protocol (NEP 18) ##

    def __array_function__(self, func, types, args, kwargs):
        if func not in self._HANDLED_FUNCTIONS:
            return NotImplemented
        # NB: In contrast to the example in NEP18, we don't require
        #     arguments to be SymmetricTensors – ndarray is also allowed.
        if not all(issubclass(t, (SymmetricTensor, np.ndarray)) for t in types):
            return NotImplemented
        return self._HANDLED_FUNCTIONS[func](*args, **kwargs)

    @classmethod
    def implements(cls, numpy_function):
        """Register an __array_function__ implementation for SymmetricTensor objects."""
        def decorator(func):
            cls._HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    ## __array_ufunc__ protocol (NEP 13) ##

    # DEVNOTE: The definition below should cover all cases where ufuncs are used
    #     *with all arguments SymmetricTensors of the same type* (or scalars).
    #     Therefore subclasses only need to override `__array_ufunc__` to add
    #     support for mixed arguments (e.g. adding a PermClsSymmetricTensor
    #     to a PermClsTorchSymmetricTensor).
    #     The generic implementation below reverts to operating on permutation
    #     classes if the types don't match exactly, which should always work,
    #     but is possibly suboptimal.
    # For a list of ufuncs, see https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs
    # NB: The function dispatch mechanism will look for the lowest subclass among
    #     both inputs and outputs which defines __array_ufunc__, and use that.
    #     Therefore, subclasses may define optimized operations for their
    #     memory layout, and revert to super() for the generic code.

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        """
        Execution essentially goes through one of three code branches:

        1. If 'out' is given and all SymmetricTensor arguments are exactly the
           same type.
           - Operations are performed in-place directly on the underlying arrays.
             This is the most efficient code path.
        2. If 'out' is not given and all SymmetricTensor arguments are exactly
           the same type
           - Operations are performed directly on the underlying arrays, but not
             in-place. (A new data structure is constructed as a dictionary,
             and then a new SymmetricTensor is instantiated.)
        3. If not all SymmetricTensor arguments (either inputs or outputs) are
           exactly the same type
           - Operations are performed by iterating over permutation classes.
             (Since permutation classes are defined for all subclasses.)
             If this does not match the underlying storage, it can be much slower.
             Even if storage is structured into permutation classes, the
             iteration over classes may still be slower than in cases 1 or 2.
        """
        f = self._HANDLED_UFUNCS[method].get(ufunc, "not found")  # Don’t use None as sentinel, because it can be added to registry to disable a ufunc
        if f == "not found":
            if ufunc.signature is None:
                # We are able to provide a default implementation for standard
                # ufuncs with "(),()->()" signature: just apply them element-wise
                if method == "__call__":
                    if ufunc.nin == 1:
                        f = partial(self.default_unary_ufunc, ufunc, method)
                    elif ufunc.nin == 2:
                        f = partial(self.default_binary_ufunc, ufunc, method)
                    else:
                        assert False, "Ufunc should have either 1 or 2 arguments."
                elif method == "at":
                    # Since 'at' is always an in-place op, supporting it with the
                    # non-contiguous indexing without triggering extra copies may
                    # require a different approach than in the default functions
                    # used above. OTOH, it is basically a restricted version of
                    # __call__, so it should be possible to do something.
                    logger.error(
                        "NOT YET IMPLEMENTED: We haven’t had a need for '.at' variants "
                        "of ufuncs, but it should be possible to add generic support for "
                        f"SymmetricTensors in {__file__}. If this message is displayed, "
                        "that’s a good indication that it would be useful.")
                    f = None
                else:
                    # method ∈ {reduce, reduceat, accumulate, outer}
                    f = None
            else:
                f = None

        if f is None:
            return NotImplemented  # EARLY EXIT
        else:
            # Initial 'ufunc' (& 'method') arg is bound either above, or in the `implements_ufunc` decorator
            return f(*inputs, **kwargs)

    @staticmethod
    def default_unary_ufunc(ufunc, method, *inputs, **kwargs):
        """
        A generic implementation for supporting unary ufuncs on
        `SymmetricTensor`s. Since unary ufuncs all have the same ``"()->()"``
        signature, this should work with all of them.

        .. Note:: The only valid methods with unary ops are ``__call__`` and
           ``at``. ``at`` is always an in-place operation and is at present
           NOT supported. (Although we believe it would be possible to do so.)
        """
        # All unary ufuncs should have the "()->()" signature and are
        # therefore easily supported: just apply them to the data.
        assert ufunc.signature is None, "Default unary operation only supports ufuncs with '()->()' signature."
        assert method == "__call__"  # Only other allowed unary ufunc is 'at', which we excluded above
        assert ufunc.nin == 1, "Unary ufunc should have 1 arguments."

        A, = inputs
        # Standardize ’outs’ to a tuple, like __array_function__
        outs = kwargs.pop("out", ())
        if not isinstance(outs, tuple):
            outs = (outs,)

        f = ufunc.__call__
        if outs:
            # Note that we can end up here if a symmetric tensor is given as an 'out'
            # argument, but the argument is not a symmetric tensor.
            # In that case, it is especially useful to cast the data to a
            # symmetric tensor first, since that may reduce the number of operations.
            if len(outs) > 1:
                raise ValueError(f"More than one value specified as 'out' argument of unary operation '{f}'.")
            out, = outs
            if isinstance(out, SymmetricTensor):
                if out is not A:
                    out[:] = A  # NB: Requires __setitem__ to support setting from whatever the type of `A` is.
                for arr in out.values():
                    f(arr, out=arr, **kwargs)
            else:
                # Assume that `out` is a normal NumPy array, or at least behaves like one
                assert isinstance(A, SymmetricTensor), "Unexpected: Called SymmetricTensor's ufunc handler, but none of the arguments are SymmetricTensors."
                for symcls, arr in A.items():
                    f(arr, out=out[A.datacls_dense_indices(symcls)], **kwargs)
        else:
            data = {key: f(arr, **kwargs) for key, arr in A.items()}
            dtype = next(iter(data.values())).dtype
            out = A.__class__(rank=A.rank, dim=A.dim, data=data, dtype=dtype)

        return out

    @staticmethod
    def default_binary_ufunc(ufunc, method, *inputs, **kwargs):
        """
        A generic implementation for supporting binary ufuncs on
        `SymmetricTensor`s. Only ufuncs with the signature ``(),()->()`` are
        supported (which is the majority).
        
        .. Note:: One the ``__call__`` method is supported by this default.
           Additional default methods for other methods like ``accumulate``
           could be implemented, if needed.
        """
        assert ufunc.signature is None, "Default binary operation only supports ufuncs with '()->()' signature."
        assert method == "__call__", "Default binary operation only supports ufuncs’ '__call__' method."
        assert ufunc.nin == 2, "Binary ufunc should have 2 arguments."

        if ufunc.nout > 1:
            # Ufuncs with > 1 outputs are rare, and we haven’t had a need for them yet.
            return NotImplemented  # EARLY EXIT

        A, B = inputs
        # Standardize ’outs’ to a tuple, like __array_function__
        outs = kwargs.pop("out", ())
        if not isinstance(outs, tuple):
            outs = (outs,)
        if outs:
            out, = outs  # Safe because we checked above that nout == 1
        else:
            out = None   # Setting to None here is easier than rewriting the logic below

        # Apply the promotion logic
        scalars = (np.ndim(A) == 0, np.ndim(B) == 0)
        symtensor_args = [o for o in inputs + outs
                          if isinstance(o, SymmetricTensor)]
        nonsymtensor_args = [o for o in inputs + outs
                             if not isinstance(o, SymmetricTensor)]
        assert len(symtensor_args) > 0, "How did we end up here if no argument is a SymmetricTensor ?"

        cls = utils.common_superclass(*symtensor_args)
        assert issubclass(cls, SymmetricTensor), "If there is a good reason not to output a SymmetricTensor here, we could support that"

        # Determine rank and dim of the output
        # At present we only support operations with signature "()->()"
        ranks = set(t.rank for t in symtensor_args)
        dims = set(t.dim for t in symtensor_args)
        if len(ranks) > 1 or len(dims) > 1:
            return NotImplemented
        rank = next(iter(ranks))
        dim = next(iter(dims))

        # Ensure 'out' is a SymmetricTensor of the correct shape
        # We could support non-symmetric outputs, it just doesn’t seem worth it for now
        outshape = np.broadcast_shapes(np.shape(A), np.shape(B))
        if out is not None and not isinstance(out, SymmetricTensor):  # We already tested above that the rank & dim match those of the inputs
            raise TypeError("At present 'out' argument only supports "
                            "outputting to another SymmetricTensor of the same shape.")
        elif out is not None and np.shape(out) != outshape:
            raise ValueError("'out' argument does not have the expected shape: input -> output shapes are "
                             f"{np.shape(A)}, {np.shape(B)} -> {outshape}, but 'out' has shape {np.shape(out)}")

        # Finally we can apply the ufunc
        f = ufunc.__call__
        if all(scalars):
            # NB: It is possible for args to be scalar SymmetricTensor
            if isinstance(A, SymmetricTensor):
                A = next(iter(A.values()))  # FIXME: Should be `next(iter(A))`, but that requires support for iterating SymTensors
            if isinstance(B, SymmetricTensor):
                B = next(iter(B.values()))  # FIXME: Same as above
            if ufunc.nout > 1:
                raise NotImplementedError("Ufuncs with > 1 outputs are rare, and we haven’t had a need for them yet.")
            if out is None:
                out = cls(rank=rank, dim=dim, data=f(A, B, **kwargs))
            else:
                val = f(A, B, **kwargs)
                for arr in out.values():
                    arr[:] = val

        elif any(scalars):
            if ufunc.nout > 1:
                raise NotImplementedError("Ufuncs with > 1 outputs are rare, and we haven’t had a need for them yet.")
            if scalars[0]:
                assert isinstance(B, SymmetricTensor), "Second argument is not a SymmetricTensor."
                if out is None:
                    # Pre-allocating a blank SymmetricTensor to `out` might not always work:
                    # e.g. PermClsSymmetricTensor would allocate scalars where we may need arrays
                    data = {key: f(A, dataB, **kwargs) for key, dataB in B.items()}
                    out = cls(rank=rank, dim=dim, data=data)
                # NB: There currently is a check above ensuring that out is a `SymmetricTensor`
                elif B.data_alignment == out.data_alignment:
                    # All SymmetricTensors are of the same type and have the same shape: their memory is aligned
                    for (key, dataB), (keyO, dataO) in zip(B.items(), out.items()):
                        assert key == keyO, "Tensors have different memory layouts, but we took a code branch where this should not be the case."
                        f(A, dataB, out=dataO, **kwargs)
                # elif not isinstance(out, SymmetricTensor):  # There currently is a check above ensuring that out is a `SymmetricTensor`
                #     for outidx, Bval in zip(B.indep_iter_index, B.indep_iter):
                #         out[outidx] = f(A, Bval, **kwargs)
                else:
                    # Possibly non-matching memory layouts: revert to σcls iteration
                    logger.info("For best performance, ensure all inputs and 'out' "
                                "arguments are of the same SymmetricTensor type.\n"
                                f"{f} received {[type(o) for o in symtensor_args]}.")
                    for σcls in B.perm_classes:
                        out[σcls] = f(A, B[σcls], **kwargs)
            else:
                assert scalars[1]
                assert isinstance(A, SymmetricTensor), "First argument is not a SymmetricTensor."
                if out is None:
                    # Pre-allocating a blank SymmetricTensor to `out` might not always work:
                    # e.g. PermClsSymmetricTensor would allocate scalars where we may need arrays
                    data = {key: f(dataA, B, **kwargs) for key, dataA in A.items()}
                    out = cls(rank=rank, dim=dim, data=data)
                # NB: There currently is a check above ensuring that out is a `SymmetricTensor`
                elif A.data_alignment == out.data_alignment:
                    # All SymmetricTensors are of the same type and have the same shape: their memory is aligned
                    for (key, dataA), (keyO, dataO) in zip(A.items(), out.items()):
                        assert key == keyO, "Tensors have different memory layouts, but we took a code branch where this should not be the case."
                        f(dataA, B, out=dataO, **kwargs)
                # elif not isinstance(out, SymmetricTensor):
                #     for outidx, Aval in zip(A.indep_iter_index, A.indep_iter):
                #         out[outidx] = f(Aval, B, **kwargs)
                else:
                    # Possibly non-matching memory layouts: revert to σcls iteration
                    logger.info("For best performance, ensure all inputs and 'out' "
                                "arguments are of the same SymmetricTensor type.\n"
                                f"{f} received {[type(o) for o in symtensor_args]}.")
                    for σcls in A.perm_classes:
                        out[σcls] = f(A[σcls], B, **kwargs)

        elif isinstance(A, SymmetricTensor) and isinstance(B, SymmetricTensor):
            if out is None:
                # Pre-allocating a blank SymmetricTensor to `out` might not always work:
                # e.g. PermClsSymmetricTensor would allocate scalars where we may need arrays
                if A.data_alignment == B.data_alignment:
                    # All SymmetricTensors are of the same type and have the same shape: their memory is aligned
                    assert A.keys() == B.keys(), "Unexpected mismatch in memory layouts of SymmetricTensor arguments."
                    data = {key: f(dataA, dataB, **kwargs)
                            for (key, dataA), dataB in zip(A.items(), B.values())}
                    out = cls(rank=rank, dim=dim, data=data)
                else:
                    # Possibly non-matching memory layouts: revert to σcls iteration
                    for σcls in B.perm_classes:
                        out[σcls] = f(A, B[σcls], **kwargs)

            elif A.data_alignment == B.data_alignment == out.data_alignment:
                # All SymmetricTensors are of the same type and have the same shape: their memory is aligned
                for (keyA, dataA), (keyB, dataB), (keyO, dataO) \
                      in zip(A.items(), B.items(), out.items()):
                    assert keyA == keyB == keyO, "Tensors have different memory layouts, but we took a code branch where this should not be the case."
                    f(dataA, dataB, out=dataO, **kwargs)

            else:
                # Possibly non-matching memory layouts: revert to σcls iteration
                logger.info("For best performance, ensure all inputs and 'out' "
                            "arguments are of the same SymmetricTensor type.\n"
                            f"{f} received {[type(o) for o in symtensor_args]}.")
                assert A.perm_classes == B.perm_classes == out.perm_classes
                for σcls in A.perm_classes:
                    out[σcls] = f(A[σcls], B[σcls], **kwargs)

        else:
            return NotImplemented

        assert isinstance(out, cls), f"Resulting output is not of the expected type. Expected {cls}, received {type(out)}."
        assert ufunc.nout == 1  # Redundant with above; repeated to document why just returning 'out' is valid
        return out


# Class-level reflective attributes
# (For subclasses, these are automatically attached by __init_subclass__)
SymmetricTensor.implements_ufunc = HandledUfuncsInterface(SymmetricTensor)

# Support calling `total_size` on SymmetricTensors
@total_size_handler(SymmetricTensor)
def _symtensor_total_size_handler(symtensor):
    # NB: __dict__ should only contain values that aren't already defined in the class
    for key, value in symtensor.__dict__.items():
        yield key
        yield value
# NB: I don't know why, but the following doesn't work (it only yields the first tuple in __dict__):
#        _total_size_handlers[SymmetricTensor] = lambda symtensor: iter(symtensor.__dict__.items())
#     Weirdly, if we step through with the debugger, then the line above yields all entries in __dict__

# # Support for calling `utils.result_type` on SymmetricTensors
# @utils.result_type_handler("statGLOW.stats.symtensor")
# def symtensor_result_type_handler(*args: Tuple[type]):
#     return tuple(a.dtype if isinstance(a, SymmetricTensor) else a for a in args)

# %% [markdown]
# ### Implementation of the `__array_ufunc__` dispatch protocol
#
# :::{hint} Function implementations can use the *data_alignment* attribute to check whether two symmetric tensors are memory-aligned. In many cases, operations can then be applied directly to the underlying data arrays, irrespective of the precise data format.
# :::
#
# Following [NEP 13](https://numpy.org/neps/nep-0013-ufunc-overrides.html), the base class `SymmetricTensor` already provides basic support for universal functions, as long as all arguments are SymmetricTensors. of exactly the same type.
#
# - If all arguments are SymmetricTensors of exactly the same type, ufuncs are applied directly to the underlying storage arrays. In most cases this should already be optimal.
# - Otherwise, operation falls back to applying ufuncs on permutation classes. This at least avoids redundant operations, but may not be optimal for those arrays storage format.
#
# Limitations of the generic base class implementation:
#
# - Arguments must be SymmetricTensor of exactly the same shape (rank and dim). No broadcasting is supported at the moment.
# - Only ufuncs taking one or two arguments, and returning exactly one output, are supported. This is the case for the majority of ufuncs.
#   (For a list of ufuncs, see https://numpy.org/doc/stable/reference/ufuncs.html#available-ufuncs)
# - Only the default (`__call__`) methods are supported. We are not confident that variants like `outer` and `reduce` can be implemented in a generic manner for all ufuncs, so a mechanism of handlers similar to the one for `__array_function__` is provided to add supported for ufunc variants.
#
# Note that on the NumPy side, the protocol is implemented such that the `__array_ufunc__` method which is called is that of lowest subclass is called among all inputs and outputs(independent of where it is in the functions arguments). This allows subclasses to override `__array_ufunc__` for specific cases, and pass control to `super()` for more generic ones.
#
# **Some implemented ufuncs**
#
# - `neg`
# - `add`, `sub`, `multiply`, `divide`, `power`
#   (Only with scalars and other symmetric tensors.)
# - `+`, `*`, `-`
# - `exp`, `sin`, `cos`, `tan`, `cosh`, `sinh`, `sign`, `abs`, `sqrt`, `log`
# - The `outer` method of `add`, `sub` and `multiply` (defined in [symalg.py](./symalg.py))
#
# The implementations in *symalg.py* work with all SymmetricTensors, but often by converting them to dense arrays. Specializing these functions for each subclass is highly recommended.
#
# **Handler mechanism**
#
# The decorator `implements_ufunc` is provided to allow adding support for specific ufuncs and specific variants. For example, the code below adds support for `np.add.outer`, `np.subtract.outer` and `np.multiply.outer` to all subclasses of `SymmetricTensor`:

# %% [markdown]
# ```python
# @SymmetricTensor.implements_ufunc.outer(np.add, np.subtract, np.multiply)
# def symmetric_outer(a, b, ufunc, **kwargs):
#     ...
# ```

# %% [markdown]
# To *remove* support for a function in a subclass (when the parent class supports it), use the similar `does_not_implement_ufunc` function. For example:
#
# ```python
# PermClsSymmetricTensor.does_not_implement_ufunc(np.modf)
# ```

# %% [markdown]
# ### Implementation of the `__array_function__` dispatch protocol
#
# Support for *non*-universal functions should be added here, following the pattern provided in [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html#example-for-a-project-implementing-the-numpy-api).
# (Support for universal functions [should](https://numpy.org/neps/nep-0018-array-function-protocol.html#specialized-protocols)  be added with the more specialized `__array_ufunc__` protocol described [above](#Implementation-of-the-__array_ufunc__-protocol).)
#
#
# Additional references/remarks:
# - There has been a lot of discussion regarding dispatching mechanisms for NumPy duck arrays – see [NEP 18](https://numpy.org/neps/nep-0018-array-function-protocol.html), [NEP 22](https://numpy.org/neps/nep-0022-ndarray-duck-typing-overview.html), [NEP 30](https://numpy.org/neps/nep-0030-duck-array-protocol.html), [NEP 31](https://numpy.org/neps/nep-0031-uarray.html), [NEP 35](https://numpy.org/neps/nep-0035-array-creation-dispatch-with-array-function.html), [NEP 47](https://numpy.org/neps/nep-0047-array-api-standard.html). Of these, only NEP 18 and NEP 35 have actually been adopted; NEP 47 seems to be where this will go in the future, but it's likely to be a few years still before this becomes implemented.
# - `tensordot` is often used as a motivating example in these cases, so whenever this matures, it likely will address the use cases we have here.
# - There is already an [open issue](https://github.com/numpy/numpy/issues/11506) for supporting `einsum_path` with non-NumPy arrays on NumPy's GitHub.
#
# Implementation allows the standard numpy functions to work as expected, e.g. `np.tensordot(A, B)` where `A`, `B` are `SymmetricTensors` will work.

# %% [markdown]
# #### `ndim()`, `shape()`

# %%
# Note: These are not strictly necessary, since of no function is defined, the default
# implementations of `np.ndim` and `np.shape` check for `ndim` and `shape` attributes.
@SymmetricTensor.implements(np.ndim)
def ndim(a: SymmetricTensor) -> int:
    return a.ndim

@SymmetricTensor.implements(np.shape)
def shape(a: SymmetricTensor) -> Tuple[int,...]:
    return a.shape


# %% [markdown]
# #### `asarray()`, `asanyarray()`

# %%
@SymmetricTensor.implements(np.asarray)
def asarray(a, dtype=None, order=None):
    new_data = {k: np.asarray(v, dtype, order) if not np.isscalar(v)
                   else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                   else v.astype(dtype)
                for k, v in a.items()}
    if all(orig_arr is new_arr for orig_arr, new_arr
           in zip(a.values(), new_data.values())):
        # None of the sub arrays were copied
        return a
    else:
        return type(a)(a.rank, a.dim, data=new_data)

@SymmetricTensor.implements(np.asanyarray)
def asanyarray(a, dtype=None, order=None):
    new_data = {k: v.asanyarray(dtype, order) if not np.isscalar(v)
                   else v if v.dtype == dtype  # Scalars are always copied with astype – even when copy=False
                   else v.astype(dtype)
                for k, v in a.items()}
    if all(orig_arr is new_arr for orig_arr, new_arr
           in zip(a.values(), new_data.values())):
        # None of the sub arrays were copied
        return a
    else:
        return type(a)(a.rank, a.dim, data=new_data)

# %% [markdown]
# #### `result_type()`

# %%
@SymmetricTensor.implements(np.result_type)
def result_type(*arrays_and_dtypes) -> DType:
    """
    Extends support for numpy.result_type. SymmetricTensors are treated as
    arrays of the same dtype.
    """
    # If the array function dispatch got here, one of the args is a SymmetricTensor
    return np.result_type(*(getattr(a, 'dtype', a) for a in arrays_and_dtypes))

# %% [markdown]
# #### `isclose()`

# %%
@SymmetricTensor.implements(np.isclose)
def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> Union[np.ndarray, SymmetricTensor]:
    """
    If `a` and `b` are compatible SymmetricTensors, returns a
    `SymmetricTensor` of the same shape with boolean entries.
    Otherwise returns an `ndarray`.
    """
    return _elementwise_compare(
        partial(np.isclose, rtol=rtol, atol=atol, equal_nan=equal_nan),
        a, b)

# %% [markdown]
# :::{margin} `_elementwise_compare`
# :::

# %%
def _elementwise_compare(comp, a, b):
    
    if not isinstance(a, SymmetricTensor):
        if not isinstance(b, SymmetricTensor):
            # Case: neither a nor b is SymmetricTensor – shouldn't happen
            assert False, "Triggered SymmetricTensor function with no SymmetricTensor arguments."
            return NotImplemented
        else:
            # Case: only b is SymmetricTensor => swap, so we can assume a is Symmetric
            a, b = b, a

    # From here can assume 'a' is a SymmetricTensor

    if np.ndim(b) == 0:
        data = {k: comp(v, b) for k,v in a.items()}
        return type(a)(rank=a.rank, dim=a.dim, data=data)

    elif not isinstance(b, SymmetricTensor):
        # Case: one SymmetricTensor vs NdArray | TorchTensor | etc.
        # OPTIMIZATION: If b.ndim < a.rank, we could save quite a number of comparisons by returning a partially symmetric tensor
        logger.warning("Comparisons between symmetric and dense tensors are "
                       "currently implemented by converting the symmetric "
                       "tensor to a dense array, which may be costly.")
        return comp(a.todense(), b)

    elif isinstance(b, SymmetricTensor):
        if a.data_alignment == b.data_alignment:
            # Case: two SymmetricTensors with the same memory layout: compare them directly
            cls = result_array(a, b)
            data = {k: comp(x, y) for (k, x), y in zip(a.items(), b.values())}
            return cls(rank=a.rank, dim=a.dim, data=data)

        elif a.rank == b.rank and a.dim == b.dim:
            # Case: two SymmetricTensors with different memory layouts but same shape: compare based on permutation classes, which at least avoids many redundant comparisons
            logger.warning("Comparing tensors with potentially different memory "
                           "layouts: comparison are done based on permutation "
                           "classes, which may involve inefficient indexing.")
            assert a.perm_classes == b.perm_classes, "How did we get two SymmetricTensors with same shape but different σ-classes ?"
            cls = result_array(a, b)
            data = {σcls: comp(a[σcls], b[σcls]) for σcls in a.perm_classes}
            return cls(rank=a.rank, dim=a.dim, data=data)

        else:
            # Case: two SymmetricTensors with different memory layouts: convert to dense and compare
            # OPTIMIZATION: Similar to the case with ndarray, a partially symmetric result could save a lot of comparisons
            logger.warning("Comparing tensors with different shapes: comparison "
                           "involves a possibly expensive conversion to dense arrays.")
            cls = result_array(a, b)
            data = comp(a.todense(), b.todense())
            return cls(a.rank, dim=a.dim, data=data)

    else:
        logger.error("SymmetricTensor array function triggered with an argument "
                     "this is neither an ndarray array nor a numer.")
        return NotImplemented

# %% [markdown]
# #### `all()`, `any()`
#
# **[TODO]** Current implementations do not support any additional arguments.
#
# %%

@SymmetricTensor.implements(np.all)
def all_(a) -> bool:
    return all(a.flat)

@SymmetricTensor.implements(np.any)
def any_(a) -> bool:
    return any(a.flat)

# %% [markdown]
# #### `array_equal()`, `allclose()`
#
# **[TODO]** For consistency with NumPy, `allclose` should apply broadcasting, and raise `ValueError` if the shapes aren’t broadcastable.

# %%

@SymmetricTensor.implements(np.array_equal)
def array_equal(a, b) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are equal. C.f. `numpy.array_equal`.
    """
    return _array_compare(np.array_equal, a , b)

@SymmetricTensor.implements(np.allclose)
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are close. C.f. `numpy.allclose`.
    """
    return _array_compare(partial(np.allclose, rtol=rtol, atol=atol, equal_nan=equal_nan),
                       a, b)


# %% [markdown]
# :::{margin} `_array_compare`
# :::

# %%
def _array_compare(comp, a, b) -> bool:
    "`comp` should be like np.array_equal or allclose: returns a single bool."
    if not isinstance(a, SymmetricTensor):
        if not isinstance(b, SymmetricTensor):
            # Case: neither a nor b is SymmetricTensor – shouldn't happen
            assert False, "Triggered SymmetricTensor function with no SymmetricTensor arguments."
            return NotImplemented
        else:
            # Case: only b is SymmetricTensor => swap, so we can assume a is Symmetric
            a, b = b, a

    # From here can assume 'a' is a SymmetricTensor

    if isinstance(b, np.ndarray):
        # Case: one SymmetricTensor vs NdArray
        return a.shape == b.shape and comp(a.todense(), b)

    elif isinstance(b, Number_):
        # The branch above would also work, but may unnecessarily create a dense array
        return a.ndim == 0 and comp(next(iter(a.flat)), b)

    elif isinstance(b, SymmetricTensor):
        if a.data_alignment == b.data_alignment:  # Includes a check that shape is the same
            # Case: two SymmetricTensors with the same memory layout: compare them directly
            return all(comp(x, y)
                       for x, y in zip(a.values(), b.values()))

        elif a.rank == b.rank and a.dim == b.dim:
            # Case: two SymmetricTensors with different memory layouts but same shape: compare based on permutation classes, which at least avoids many redundant comparisons
            assert a.perm_classes == b.perm_classes, "How did we get two SymmetricTensors with same shape but different σ-classes ?"
            return all(comp(a[σcls], b[σcls]) for σcls in a.perm_classes)

        else:
            # Case: two SymmetricTensors with different shape
            return False

    else:
        logger.error("SymmetricTensor array function triggered with an argument "
                     "this is neither an ndarray array nor a numer.")
        return NotImplemented


# %% [markdown]
# #### Definition of new array functions
#
# **Implementation explanation:**
#
# - The `@array_function_dispatch` decorator is used to define a new array function.
#   The function it decorates serves as the default, if the arguments do not match those of another function.
# - To each array function is paired a *dispatcher* function. This function must have the same signature as its associated function, and return a tuple of "important" arguments:[^1] those arguments the dispatcher should check to determine which function to redirect to.
# - Specialized functions can be associated to specific `SymmetricTensor` subclasses by using the `@implements` decorator.
#
#
# [^1]: Exception: if the function includes arguments with default values, the corresponding arguments of the dispatcher must use `None` as their default value. See also the docstring of numpy.core.overrides.array_function_dispatch; some examples can be found in numpy.core.numeric.py

# %% [markdown]
# ::: {margin} `array_function_dispatch`
# :::

# %%
def array_function_dispatch(dispatcher, module=None, verify=True,
                            docs_from_dispatcher=False):
    """
    Combine NumPy's `@array_function_dispatch` with `SymmetricTensor.implements`.

    When defining new symmetric array functions, we need to do two things:

    - Augment implementations with the dispatch logic. This is done with the
      `@array_function_dispatch` decorator and the ``*_dispatcher`` functions.
    - Add functions `SymmetricTensor`'s ``_HANDLED_FUNCTIONS`` registry, so the
      dispatcher can find it.
      This is done with the `@SymmetricTensor.implements` decorator.

    Note that NumPy's `@array_function_dispatch` function returns a new function;
    it is the original one which we need to add to the ``_HANDLED_FUNCTIONS``
    registry. Because of this, we can't achieve what we want by applying both
    decorators, so we create a new one below combines both.
    """
    dispatch_decorator = _array_function_dispatch(
        dispatcher, module=module, verify=verify, docs_from_dispatcher=docs_from_dispatcher)
    def decorator(implementation):
        array_function = dispatch_decorator(implementation)
        SymmetricTensor.implements(array_function)(implementation)
        return array_function
    return decorator

# %% [markdown]
# #### Type promotion: `result_array`
#
# Similar to how `result_type` is used to determine dtype promotion, `result_array` determines promotion of array classes.
#
# The default implementation does the following:
#
# - If no arguments are SymmetricTensors, return `ndarray`.
# - If one argument is a SymmetricTensor, return that type.
# - If multiple arguments are SymmetricTensors, return the most specific SymmetricTensor subclass which is common to all arguments.
#   This may be the abstract class `SymmetricTensor`, if that is the only shared subclass.
#
#
# A subclass may decide to add itself to the dispatch to give its own types precedence. For example, to ensure that any operation with a `TorchSymmetricTensor` returns a `TorchSymmetricTensor`, that class should define a new function and hook it into the array function dispatch protocol:
#
# ```python
# @TorchSymmetricTensor.implements(symalg.result_array)
# def result_array(*arrays_and_types):
#     ...
# ```

# %%
def _result_array_dispatch(*arrays_and_types):
    return arrays_and_types

@array_function_dispatch(_result_array_dispatch)
def result_array(*arrays_and_types) -> Type[np.ndarray]:
    """
    Analogue to `result_type`: apply type promotion on array classes themselves.
    Arguments may be types or instances.

    Default implementation. If this function is called, none of the arguments
    triggered one of the specialized functions, so we return the default
    `np.ndarray`.
    """
    types = (arr if isinstance(arr, type) else type(arr)
             for arr in arrays_and_types)
    if not all(issubclass(T, np.ndarray) for T in types):
        raise NotImplementedError(
            "Default implementation of `result_array` expects all arguments to "
            f"be ndarrays. Received arguments with the following types: {types}")
    return np.ndarray


# %%
@SymmetricTensor.implements(result_array)
def result_symtensor(*arrays_and_types) -> Type[SymmetricTensor]:
    """
    Analogue to `result_type`: apply type promotion on array classes themselves.
    Arguments may be types or instances.

    When multiple arguments are SymmetricTensors, the returned type is the
    most specific common superclass.

    result_array(ndarray, ndarray) -> ndarray
    result_array(ndarray, SymmetricTensor) -> SymmetricTensor
    result_array(SymmetricTensor, ndarray) -> SymmetricTensor
    result_array(SymmetricTensor, SymmetricTensor) -> SymmetricTensor
    """
    types = (arr if isinstance(arr, type) else type(arr)
             for arr in arrays_and_types)
    symtypes = tuple(T for T in types if issubclass(T, SymmetricTensor))
    return utils.common_superclass(*symtypes)



# %% [markdown]
# ### Symmetrized algebra
#
# The [*symalg*](./symalg.py) module provides default implementations for the
# following functions which work for all `SymmetricTensor`s :
#
# - `np.tensordot` (*symmetrized* version)
# - `contract_all_indices_with_matrix`
# - `contract_all_indices_with_vector`
# - `contract_tensor_list`
#
# [**TODO**] We would also like to support:
# - `einsum_path()`  (Already possible with per-subclass definitions.)
# - `einsum()`

# %%
