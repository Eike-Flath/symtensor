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

from typing import Union, ClassVar, Iterator, Generator, Dict, List, Tuple, Set
from warnings import warn
from ast import literal_eval
import itertools
import math  # For operations on plain Python objects, math can be 10x faster than NumPy
import numpy as np
import torch
from pydantic import BaseModel
from tqdm.auto import tqdm
import time
from mackelab_toolbox.utils import TimeThis
import statGLOW
from statGLOW.smttask_ml.scityping import Serializable, TorchTensor, DType
import pytest
from statGLOW.utils import does_not_warn
from collections import Counter
from statGLOW.stats.symmetric_tensor import *
from statGLOW.stats.symmetric_tensor.permcls_symmetric_tensor import _get_perm_class,_get_perm_class_size,_indexcounts, partition_list_into_two

# %%
if __name__ == "__main__":
    import holoviews as hv
    hv.extension('bokeh')

# %%
__all__ = ["TorchSymmetricTensor"]


# %% [markdown]
# ### `TorchSymmetricTensor`
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

# %% [markdown]
# ### CPU to GPU: 
# We want to move heavy calculations from CPU to GPU using `pytorch`. 
#
#
# To do this we must: 
#   - [x] Rewrite `__init__` to define the device
#   - [ ] Ensure that data is stored on GPU:
#     - [x] if `SymmetricTensor` is initialized with data dictionary, ensure that the data is stored as `torch.Tensor` on the right device
#     - [x] if `__setitem__()` is called, ensure that the data are stored on the right device **(?)**
#     - [x] Rewrite `__getitem__` for pytorch **(?)**
#     - [x] Rewrite `indep_iter` for pytorch
#   - [ ] Ensure data manipulations are done on GPU: 
#      - [ ] Rewrite `__array_ufunc_` for torch functions
#      - [x] Rewrite `__array_function_` for torch functions **if necessary?**
#      - [x] Rewrite `tensordot` for pytorch
#      - [x] Rewrite `outer_product` for pytorch
#      - [ ] Rewrite `contract_all_indices` for pytorch in Schatz paper fig 3 way
#      - [ ] Rewrite `contract_tensor_list` for pytorch
#      - [ ] Rewrite `poly_term` for pytorch
#      
#      
# Question collection: 
# -  For serialisation: Does `Array` include `torch.Tensor`?

# %%
class TorchSymmetricTensor(PermClsSymmetricTensor):
    """
    On creation, defaults to a zero tensor.
    """
    _HANDLED_FUNCTIONS = {}  # Registry used for __array_function__ protocol
    indices: ClassVar[str] = "ijklmnαβγδ"

    rank                 : int
    dim                  : int
    _dtype                : DType
    _data                : Dict[Tuple[int], Union[float, TorchTensor[float,1]]]
    _class_sizes         : Dict[Tuple[int], int]
    _class_multiplicities: Dict[Tuple[int], int]
        # NB: Internally we use the index counts instead of the equivalent
        # string to represent classes, since it is more useful for calculations

    def __init__(self, rank: int, dim: int,
                 data: Optional[Dict[Union[Tuple[int,...], str],
                                     Array[float,1]]]=None,
                 dtype: Union[None,str,DType]=None):
        if data:
            raise NotImplementedError("Initializing TorchSymmetricTensor with data" 
                                     "is not yet supported. Please use '__setitem__'"
                                     "to set tensor values.")
        #initialize as empty Tensor
        super(TorchSymmetricTensor, self).__init__(rank, dim, data = None, dtype = None)
        #set device  
        if torch.cuda.is_available():
            self._device = torch.device('cuda')
        else:
            self._device = torch.device('cpu')
        #data in pytorch
        self._data = {tuple(repeats): torch.tensor(0.0).to(device=self._device)
                      for repeats in _indexcounts(rank, rank, rank)}
        


    ## Pydantic serialization ##
    '''
    class Data(BaseModel):
        rank: int
        dim: int
        # NB: JSON keys must be str, int, float, bool or None, but not tuple => convert to str
        data: Dict[str, Union[float, TorchTensor[float,1]]] # comparison to SymmetricTensor: change Array -> TorchTensor here
        @classmethod
        def json_encoder(cls, symtensor: SymmetricTensor):
            return cls(rank=symtensor.rank, dim=symtensor.dim,
                       data={str(k): v for k,v in symtensor._data.items()})'''

    ## Translation functions ##
    # Mostly used internally, but part of the public API

    ## Dunder methods ##

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
                    C = SymmetricTensor(rank = new_rank, dim = self.dim)
                    for idx in C.index_class_iter():
                        C[idx] = self[idx +indices_fixed]
                    return C

            else:
                σcls = _get_perm_class(key)
                vals = self._data[σcls]
                if vals.ndim == 0:
                    return vals
                else:
                    σcls, pos = self._convert_dense_index(key)
                    return vals[pos]
        elif self.rank==1 and isinstance(key,int): #special rules for vectors
            vals = self._data[(1,)]
            return vals if vals.ndim == 0 else vals[key]
        elif self.rank >1 and isinstance(key, int):
            if self.dim ==1:
                σcls, pos = self._convert_dense_index(key)
                vals = self._data[σcls]
                return vals if vals.ndim == 0 else vals[pos]
            elif self.dim >1:
                B = SymmetricTensor(rank = self.rank-1, dim = self.dim)
                for idx in B.index_class_iter():
                    B[idx] = self[idx+(key,)]

                return B
        else:
            raise KeyError(f"{key}")

    def __setitem__(self, key, value):
        if type(value) == int or type(value) == float: 
            value = torch.tensor(value)
        assert type(value)==torch.Tensor, "Values must be torch tensors"
        if isinstance(key, str):
            repeats = _get_perm_class(tuple(key))
            
            if repeats not in self._data:
                raise KeyError(f"'{key}' does not match any permutation class.\n"
                               f"Permutation classes: {self.perm_classes}.")
            if value.ndim == 0:
                self._data[repeats] = value.to(device=self._device)
            else:
                if len(value) != _get_perm_class_size(repeats, self.dim):
                    raise ValueError(
                        "Value must either be a scalar, or match the index "
                        f"class size.\nValue size: {len(value)}\n"
                        f"Permutation class size: {_get_perm_class_size(repeats, self.dim)}")
                self._data[repeats] = value.to(device=self._device) # put data on gpu/cpu
        else:
            if self.rank==1 and isinstance(key,int): #special rules for vectors
                σcls = (1,)
                pos = key
            else:
                σcls, pos = self._convert_dense_index(key)
            v = self._data[σcls]
            if np.ndim(v) == 0:
                if pos == slice(None):  # Equivalent to setting the whole permutation class
                    self._data[σcls] = value.to(device=self._device)
                elif np.ndim(value) == 0 and v == value:
                    # Value has not changed; no need to expand
                    pass
                else:
                    # Value is no longer uniform for all positions => need to expand storage from scalar to vector
                    v = v * torch.ones(self._class_sizes[σcls], device = self._device)
                    v[pos] = value.to(device=self._device)
                    self._data[σcls] = v
            else:
                self._data[σcls][pos] = value.to(device=self._device)

    ## Numpy dispatch protocols ##

    # __array_function__ protocol (NEP 18)

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

    # __array_ufunc__ protocol (NEP 13)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        ufunc_dict = {np.add      : torch.add, 
                      np.subtract : torch.subtract, 
                      np.multiply : torch.multiply, 
                      np.divide   : torch.divide, 
                      np.power    : torch.float_power,
                      np.exp      : torch.exp, 
                      np.sin      : torch.sin, 
                      np.cos      : torch.cos, 
                      np.tan      : torch.tan, 
                      np.cosh     : torch.cosh, 
                      np.sinh     : torch.sinh, 
                      np.tanh     : torch.tanh, 
                      np.sign     : torch.sign, 
                      np.abs      : torch.abs, 
                      np.sqrt     : torch.sqrt, 
                      np.log      : torch.log
        
        }
        if method == "__call__":  # The "standard" ufunc, e.g. `multiply`, and not `multiply.outer`
            if ufunc in {np.add, np.subtract, np.multiply, np.divide, np.power}:  # Set of all ufuncs we want to support
                A, B = inputs   # FIXME: Check the shape of `inputs`. It might also be that we need `B = self` instead
                if not isinstance(A, SymmetricTensor):
                    if np.ndim(A) ==0: #check if is scalar
                        C = self.__class__(dim=A.dim, rank=A.rank)
                        for σcls in C.perm_classes:
                            C[σcls] = ufunc_dict[ufunc](A, B[σcls])
                        return C
                    else:
                        raise TypeError(f"{ufunc} is not supported for objects of type {type(A)} and {type(B)}")
                elif not isinstance(B, SymmetricTensor):
                    if np.ndim(B) ==0: #check if is scalar
                        C = self.__class__(dim=A.dim, rank=A.rank)
                        for σcls in C.perm_classes:
                            C[σcls] = ufunc_dict[ufunc](A[σcls], B)
                        return C
                    else:
                        raise TypeError(f"{ufunc} is not supported for objects of type {type(A)} and {type(B)}")
                elif A.dim != B.dim or A.rank != B.rank:
                    return NotImplemented
                else:
                    C = self.__class__(dim=A.dim, rank=A.rank)
                    for σcls in C.perm_classes:
                        C[σcls] = ufunc_dict[ufunc](A[σcls], B[σcls])  # This should always do the right whether, whether A[σcls] is a scalar or 1D array
                    return C
            elif ufunc in {np.exp, np.sin, np.cos, np.tan, np.cosh, np.sinh, np.tanh, np.sign, np.abs, np.sqrt, np.log}:  # Set of all ufuncs we want to support
                A, = inputs   # FIXME: Check the shape of `inputs`. It might also be that we need `B = self` instead
                C = self.__class__(dim=A.dim, rank=A.rank)
                for σcls in C.perm_classes:
                    C[σcls] = ufunc_dict[ufunc](A[σcls])  # This should always do the right whether, whether A[σcls] is a scalar or 1D tensor
                return C
        elif method == "outer":
            assert ufunc is np.multiply, f"{ufunc}.outer is not supported"
            A, B = inputs
            return self.outer_product(B, ufunc = ufunc, **kwargs)
        else:
            return NotImplemented  # NB: This is different from `raise NotImplementedError`

    def outer_product(self, other, ufunc=np.multiply):  # SYMMETRIC ALGEBRA
        """
        Implement the outer product. Note that the outer product of two symmetric tensors is not symmetric.
        The result generated here is the symmetrized version of the outer product.
        """
        if isinstance(other, TorchSymmetricTensor):
            if self.dim != other.dim:
                raise NotImplementedError("Currently only outer products between SymmetricTensors of the same dimension are supported.")
            else:
                C = TorchSymmetricTensor(dim=self.dim, rank=self.rank+other.rank)
                for I in C.index_class_iter():
                    list1, list2, L = partition_list_into_two(I, self.rank, other.rank)
                    C[I] = sum( torch.multiply(self[tuple(idx1)], other[tuple(idx2)]) for idx1, idx2 in zip(list1,list2) )/L
                return C
        elif isinstance(other, list):
            C = self.copy()
            for o in other:
                C = C.outer_product(o)
            return C
        elif not isinstance(other, (TorchSymmetricTensor,list)):
            raise TypeError( 'Argument must be SymmetricTensor or list of SymmetricTensors')

    def tensordot(self, other, axes=2):
        """
        like numpy.tensordot, but outputs are all symmetrized.
        """
        if not isinstance(other, TorchSymmetricTensor):
            raise NotImplementedError("Currently only tensor products between SymmetricTensors are supported.")
        if self.dim != other.dim:
            raise NotImplementedError("Currently only tensor products between SymmetricTensors of the same dimension are supported.")
        if isinstance(axes,int):
            if axes == 0:
                return self.outer_product(other)
            elif axes == 1:
                # note: \sum_i A_jkl..mi B_inop..z = \sum_i A_ijkl..m B_inop..z for A, B symmetric
                if other.rank == 1 and self.rank ==1:
                    return np.dot(self['i'],other['i'])
                elif other.rank ==1 and self.rank >1:
                    return sum((self[i]*other[i] for i in range(self.dim)),
                               start=TorchSymmetricTensor(self.rank -1, self.dim))
                elif other.rank >1 and self.rank ==1:
                    return sum((self[i]*other[i] for i in range(self.dim)),
                               start=TorchSymmetricTensor(other.rank -1, self.dim))
                else:
                    return sum((self[i].outer_product(other[i]) for i in range(self.dim)),
                           start=TorchSymmetricTensor(self.rank + other.rank - 2, self.dim))
            elif axes == 2:
                if self.rank < 2 or other.rank < 2:
                    raise ValueError("Both tensors must have rank >=2")
                get_slice_index = lambda i,j,rank: (i,j,) +(slice(None,None,None),)*(rank-2)
                if self.rank ==2 or other.rank==2:
                     C = sum((torch.multiply(
                            self[get_slice_index(i,j,self.rank)],
                            other[get_slice_index(i,j,other.rank)])
                         for i in range(self.dim) for j in range(other.dim)),
                        start=TorchSymmetricTensor(self.rank + other.rank - 4, self.dim))
                else:
                    C = sum((np.multiply.outer(
                                self[get_slice_index(i,j,self.rank)],
                                other[get_slice_index(i,j,other.rank)])
                             for i in range(self.dim) for j in range(other.dim)),
                            start=TorchSymmetricTensor(self.rank + other.rank - 4, self.dim))
                return C
            else:
                raise NotImplementedError("tensordot is currently implemented only for 'axes'= 0, 1, 2. "
                                          f"Received: {axes}")
        elif isinstance(axes, tuple):
            axes1 ,axes2 = axes
            if isinstance(axes1, tuple):
                if not isinstance(axes2, tuple):
                    raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
                                    f"Received: {axes}")
                if len(axes1) != len(axes2):
                    raise ValueError("# dimensions to sum over must match")
                rank_deduct = len(axes1)
                get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-rank_deduct)
                C = sum((np.multiply.outer(self[get_slice_index(idx,self.rank)],
                                           other[get_slice_index(idx,other.rank)])
                         for idx in itertools.product(range(self.dim),repeat = rank_deduct)),
                        start=TorchSymmetricTensor(self.rank + other.rank - 2*rank_deduct, self.dim))
                return C
            elif isinstance(axes1,int):
                if not isinstance(axes2,int):
                    raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
                                    f"Received: {axes}")
                return self.tensordot(other, axes = 1)
            else:
                raise TypeError("'axes' must be either int, tuple of length 2, or tuple of tuples. "
                                f"Received: {axes}")
        else:
            raise NotImplementedError("Tensordot with more axes than two is currently not implemented. "
                                      f"Received: axes={axes}")

    def contract_all_indices(self,W):
        """
        compute the contraction over all indices with a non-symmetric matrix, e.g.

        C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}

        if current tensor has rank 3.
        """

        C = TorchSymmetricTensor(rank = self.rank, dim = self.dim)
        if self.rank == 1: 
            return torch.tensordot(W,self['i'], dims =1, device = self._device)
        if self.rank == 2: 
            for i in range(0, self.dim): 
                y = W[:,i]
                t_1 = torch.tensordot(self.to_dense(),y, dims = 1, device= self._device) 
                for j in range(0, i+1):
                    y = W[:,j]
                    C[i,j] = torch.tensordot(t_1, y, dims =1, device = self._device)
        if self.rank == 3:
            for i in range(0, self.dim): 
                y = TorchSymmetricTensor(rank =1, dim = self.dim)
                y['i'] = W[:,i]
                t_1 = self.tensordot(y, axes = 1).todense() #t_1 is matrix 
                for j in range(0, i+1): 
                    y = W[:,j]
                    t_2 = torch.tensordot(t_1,y, dims = 1, device= self._device) 
                    for k in range(0, j+1):
                        y = W[:,k]
                        C[i,j,k] = torch.tensordot(t_2, y, dims =1, device = self._device)
        elif self.rank == 4: 
            for i in range(0, self.dim): 
                y = TorchSymmetricTensor(rank =1, dim = self.dim)
                y['i'] = W[:,i]
                t_1 = self.tensordot(y, axes = 1) 
                for j in range(0, i+1): 
                    y = TorchSymmetricTensor(rank =1, dim = self.dim)
                    y['i'] = W[:,j]
                    t_2 = t_1.tensordot(y, axes = 1).todense() #t_2 is matrix 
                    for k in range(0, j+1): 
                        y = W[:,k]
                        t_3 = torch.tensordot(t_2,y, dims = 1, device= self._device)
                        for l in range(0, k+1):
                            y = W[:,l]
                            C[i,j,k,l] = torch.tensordot(t_3,y, dims = 1, device= self._device)

        return C

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
            if not isinstance(list_entry, TorchSymmetricTensor):
                raise  TypeError("tensor_list entries must be SymmetricTensors")
        if self.rank ==1 and n_times ==1:
            return sum((tensor_list[i]*self[i] for i in range(self.dim)),
                        start=TorchSymmetricTensor(tensor_list[0].rank, tensor_list[0].dim))
        else:
            get_slice_index = lambda idx,rank: idx +(slice(None,None,None),)*(rank-n_times)
            if rule == 'second_half':
                first_half = int(np.ceil(self.dim/2.0))
                indices_for_contraction = range(first_half, self.dim)
                indices = itertools.product( indices_for_contraction, repeat = n_times)
            else:
                indices = itertools.product(range(self.dim), repeat = n_times)
            chi_rank = tensor_list[0].rank
            C = TorchSymmetricTensor(dim = self.dim, rank = self.rank +(chi_rank-1)*n_times) #one dimension used for contraction
            if n_times < self.rank:
                for idx in indices:
                    slice_idx = get_slice_index(idx, self.rank)
                    C += self[slice_idx].outer_product([tensor_list[i] for i in idx])
            else:
                for idx in indices:
                    slice_idx = get_slice_index(idx, self.rank)
                    C += tensor_list[idx[0]].outer_product([ tensor_list[i] for i in idx[1:]])*self[slice_idx]
            return C

    def poly_term(self, x):
        """
        for x an array, compute
        \sum_{i_1, ..., i_r} self_{i_1,..., i_r} x_{1_1} ... x_{i_r}
        """
        if not len(x) == self.dim:
            raise ValueError('dimension of vector must match dimension of tensor')
        if np.isclose(x,np.zeros(self.dim)).all():
            return 0
        else:
            vec = TorchSymmetricTensor(rank =1, dim = self.dim)
            vec['i'] = x
            C = self.copy()
            for r in range(self.rank):
                C = C.tensordot(vec, axes =1)
            return C

    def todense(self) -> TorchTensor:
        A = torch.empty(self.shape)
        for idx, value in zip(self.index_iter(), self.indep_iter()):
            A[idx] = value
        return A

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
            if v.ndim == 0:
                yield from itertools.repeat(v, size*mult)
            else:
                for vi in v:
                    yield from itertools.repeat(vi, mult)


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
                if v.ndim == 0:
                    yield from itertools.repeat(v, size)
                else:
                    yield from v
        else:
            self._check_class_label(class_label)
            repeats = self.get_class_tuple(class_label)
            v = self._data[repeats]
            if v.ndim==0:
                size = self._class_sizes[repeats]
                yield from itertools.repeat(v, size)
            else:
                yield from v


# %% [markdown]
# ## Tests

# %%
if __name__ == "__main__":
    import pytest
    from statGLOW.utils import does_not_warn
    from collections import Counter
    def test_tensors() -> Generator:
        for d, r in itertools.product([2, 3, 4, 6, 8], [2, 3, 4, 5, 6]):
            yield TorchSymmetricTensor(rank=r, dim=d)
    assert TorchSymmetricTensor(rank=4, dim=3).perm_classes == \
        ['iiii', 'iiij', 'iijj', 'iijk', 'ijkl']

# %% [markdown]
# ### Assignement
#
# Test assignement: Assigning one value modifies all associated symmetric components.

# %%
if __name__ == "__main__":
    #test 1-d tensor setting and getting
    A = TorchSymmetricTensor(1, 3)
    A['i'] = torch.Tensor([1,2,3])
    assert A[0] == 1
    assert A[2] == 3
    
    A[0] = -5.1
    assert A[0] == -5.1
    assert A[2] == 3
    assert A['i'].device == A._device
    assert A[0].device == A._device

    # %%
    B = TorchSymmetricTensor(2, 3)
    B['ii'] = torch.Tensor([1,2,3])
    assert B[0,0] == 1
    assert B[2,2] == 3
    assert B['ij'].device == B._device

# %% [markdown]
# ### `indep_iter`
# Test iteration over all values.

    # %%
    sum_entries = 1+2+3
    sum_entries_check = 0
    for v in B.indep_iter():
        sum_entries_check += v
        assert v.device == B._device
    assert sum_entries_check == sum_entries

# %% [markdown]
# ### Serialization 
# Still to do:

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
#     with does_not_warn(UserWarning):
#         np.einsum_path("ij,ik", A, B)
#         np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))
#
#     with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#         with does_not_warn(UserWarning):
#             np.einsum_path("ij,ik", A, B)
#             np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))

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
    test_tensor_1 = TorchSymmetricTensor(rank=rank, dim=dim)
    test_tensor_1['iiii'] = torch.randn(2)
    test_tensor_2 = np.add(test_tensor_1,1.0)
    test_tensor_3 = TorchSymmetricTensor(rank=rank, dim=dim)
    for σcls in test_tensor_3.perm_classes:
                test_tensor_3[σcls] = 1.0
    test_tensor_4 =  test_tensor_2 - test_tensor_3
    assert test_tensor_4.is_equal(test_tensor_1, prec =1e-5)
    test_tensor_5 = np.multiply(test_tensor_2, -1)
    test_tensor_6 = np.multiply(test_tensor_5, -1)
    #test multiplication
    assert test_tensor_6.is_equal(test_tensor_2, prec =1e-5)
    test_tensor_7 = np.exp(test_tensor_2)
    test_tensor_8 = np.log(test_tensor_7)
    #test log, exp
    assert test_tensor_8.is_equal(test_tensor_2, prec =1e-5)

# %% [markdown]
# ### Tensordot

# %%
if __name__ == "__main__":
    #outer product
    TimeThis.on = False

    test_tensor_1d = test_tensor_1.todense()
    test_tensor_2d = test_tensor_2.todense()
    test_tensor_3d = test_tensor_3.todense()
    prec =1e-5
    test_tensor_8 = np.multiply.outer(test_tensor_2,test_tensor_3)
    print(test_tensor_3.todense(),abs(test_tensor_8.todense()- symmetrize(np.multiply.outer(test_tensor_2d,test_tensor_3d))))
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
        if isinstance(test_tensor_15, TorchSymmetricTensor):
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

    A = TorchSymmetricTensor(rank = 3, dim=3)
    A[0,0,0] =1
    A[0,0,1] =-12
    A[0,1,2] = 0.5
    A[2,2,2] = 1.0
    A[0,2,2] = -30
    A[1,2,2] = 0.1
    A[1,1,1] =-0.3
    A[0,1,1] = 13
    A[2,1,1] = -6
    W = torch.randn(3,3)
    W1 = torch.randn(3,3)
    W2 = torch.randn(3,3)
    assert np.isclose(A.contract_all_indices(W).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W))).all()
    assert np.isclose(A.contract_all_indices(W1).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1))).all()
    assert np.isclose(A.contract_all_indices(W2).todense(), symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W2,W2,W2))).all()

    B = TorchSymmetricTensor(rank = 4, dim =4)
    B['iiii'] = torch.randn(4)
    B['ijkl'] =12
    B['iijj'] = torch.randn(6)
    B['ijkk'] =-0.5
    W = torch.randn(4,4)
    C = B.contract_all_indices(W)
    W1 = np.random.rand(4,4)
    W2 = np.random.rand(4,4)
    assert np.isclose(C.contract_all_indices(W).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W,W,W,W))).all()
    assert np.isclose(C.contract_all_indices(W1).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W1,W1,W1,W1))).all()
    assert np.isclose(C.contract_all_indices(W2).todense(), symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W2,W2,W2,W2))).all()


# %% [markdown]
# ## Contraction with list of SymmetricTensors

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
# ## Contraction with vector

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
    assert np.isclose(A.poly_term(x), np.einsum('abc, a,b,c -> ', A.todense(), x,x,x))
    assert np.isclose(A.poly_term(x1), np.einsum('abc, a,b,c -> ', A.todense(), x1,x1,x1))
    #assert np.isclose(A.poly_term(x2), 0)


# %%
if __name__ == "__main__":
    print(A.poly_term(x2))

# %% [markdown]
# ## Copying and Equality

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
# ## Slowness of slicing
# Some tests to see where slowness could come from:

# %%
if __name__=="__main__":
    TimeThis.on= True
    with TimeThis("check slicing speed"):
        D = A[0]


# %% [markdown]
# ### slowness of outer product:
#

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
