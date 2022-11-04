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
# # Symmetric PyTorch tensors from decomposed vectors
#
# Symmetric tensors can be written in decomposed form 
#
# $$
# T = \sum_{m} \lambda^m t^m \otimes t^m \otimes \dots t^m
# $$
# with $t^m$ vectors. 
#
# Sometimes we also obtain tensors of the form 
# $$
# T = \sum_{m} \lambda^{m,n} \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots v^n}_{l \text{ times}}
# $$
# with $u^m, v^m $ vectors. 
# This Tensor is not symmetric, but we may obtain its symmetrized form via permutations of the outer products
# $$
# T = \sum_{m} \lambda^{m,n} \left(\frac{(k+l)!}{l! k!}\right)^{-1}\left[ \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots v^n}_{l \text{ times}} + v^n  \otimes \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots v^n}_{l-1 \text{ times}} +\dots \right]
# $$
#
# We perform the symmetrization only when we retrieve specific entries of the tensor. 
#
# As there are $(k+1)!$ ways of arranging the $k+l$ other products, but permutations of the $k$ identical vectors $u^m$ or the $l$ identical vectors $v^n$ do not yield new terms. 
# This representation is especially useful for many operations, for example the contraction with a matrix along all indices. 
#
# We call $\lambda^m$ the weights and $t^m$ the vectors (strictly speaking, these could be again symmetric tensors) and $k,l$ in the example above the multiplicities.

# %% tags=["remove-input"]
from __future__ import annotations
import statGLOW.typing

# %% tags=["remove-input"]
from functools import reduce
import numpy as np   # To avoid MKL bugs, always import NumPy before Torch
import torch
from pydantic import BaseModel
from collections_extended import bijection
from scipy.special import binom
import itertools

from typing import Optional, ClassVar, Union
from scityping import Number
from scityping.numpy import Array, DType
from scityping.torch import TorchTensor


# %% tags=["active-ipynb", "remove-input"]
# # Module only imports
# from symtensor.torch_symtensor import TorchSymmetricTensor
# from symtensor.permcls_symtensor import PermClsSymmetricTensor
# import symtensor.utils as utils 

# %% tags=["active-py", "remove-cell"]
Script only imports
from .torch_symtensor import TorchSymmetricTensor
from .permcls_symtensor import PermClsSymmetricTensor
from . import utils


# %%
class DecompSymmetricTensor(TorchSymmetricTensor, PermClsSymmetricTensor):
    """
    Abstract `DecompTensor` using outer products to express 
    symmetric tensors.
    """
    
    # Overridden class attributes
    array_type  : ClassVar[type]=torch.tensor
    # New attributes
    _device_name : str
    # for some reason, needs this
    _data : Dict[Tuple[int], Union[float, Array[float,1]]]
    
    def __init__(self, rank: Optional[int]=None, dim: Optional[int]=None,
                 data: Union[Array, Number]=np.float64(0),
                 dtype: Union[str,DType]=None,
                 device: Union[str, 'torch.device']="cpu"):
        """
        {{base_docstring}}
        
        Torch-specific parameter
        ------------------------
        device: A string identifying a PyTorch device, either 'cpu' or 'gpu'.
           The default value 'None' will prevent the `torch` library from being
           loaded at all; thus it will work even when PyTorch is not installed.
           PyTorch device objects are also supported.

        """
        self._weights = None
        self._factors = None
        self._multiplicities = None
        
        super().__init__(rank = rank, dim = dim , data = data, dtype =dtype)
        # Sets rank, dim, device, _σclass_sizes, _σclass_multiplicities
        # Calls _validate_data
        # Sets _dtype
        # Calls _init_data

    @classmethod
    def _validate_data(cls,
                       data: Optional[Dict[Union[Tuple[int,...], str]]], 
                       symmetrize: bool = False
                       ) -> Dict[Tuple[int,...], Array[Any,1]]:
        """
        {{base_docstring}}

        For the case of DecompSymmetricTensor, this specifically means
        - Standardizing the `data` argument to a dict of 
            {weights: Array, factors: Tuple[Array,SymmetricTensors], multiplicities :Tuple}
        - Asserting that all array dtypes are numeric
        - Infer the dtype by applying type promotion on data dtypes
        - symmetrize is ignored.
        """
        if isinstance(data, np.ndarray):
            raise NotImplementedError("Casting plain arrays to DecompSymmetricTensor "
                                      "is not possible.")
        elif isinstance(data, dict):
            # ensure data has right keys and format
            for key in list(data):
                if key == "weights": 
                    data[key] = self._validate_dataarray(data[key])
                    self._weights = data[key]
                    datadtype = np.result_type(data.values())
                elif key == "multiplicities": 
                    if not isinstance(data[key],tuple): 
                        raise TypeError("multiplicities must be of type tuple[int]")
                    if not (isinstance(k,int) for k in data[key]).all(): 
                        raise TypeError("multiplicities must be of type tuple[int]")
                    if not (k>0 for k in data[key]).all():
                        raise ValueError("multiplicities must be > 0.")
                    self._multiplicities = data[key]
                elif key == "factors": 
                    if not isinstance(data[key],tuple): 
                        raise TypeError("factors must be of type tuple[array,SymmetricTensor]")
                    if not ((isinstance(v,torch.Tensor) or isinstance(v,SymmetricTensor)) for v in data[key]).all(): 
                        raise TypeError("factors must be of type tuple[array,SymmetricTensor]")
                    self._factors = data[key]
                else: 
                    raise ValueError(f"`data` contains the key '{key}'."
                                     "Permitted data keys are `mulitplicities`,"
                                     " `weights` and `factors`.")
                if isinstance(key, str):
                    newkey = literal_eval(key)  # NB: That this is Tuple[int] is verified below
                    if newkey in data:
                        raise ValueError(f"`data` contains the key '{key}' "
                                         "twice: in both its original and "
                                         "serialized (str) form.")
                    data[newkey] = data[key]
                    del data[key]

            # Infer the data dtype
            datadtype = np.result_type(value for key, value in data.items() if not key=="multiplicities")
            if not np.issubdtype(datadtype, np.number):
                    raise TypeError("Data should have numeric dtypes; received "
                                    f"{[np.result_type(v) for v in data.values()]}.")
            
            # Infer the data shape, ensure that multiplicities match the weights
            assert len(self._multiplicities) == len(self._weights.shape), "multiplicities do not match weights."
            data_shape = sum(self._multiplicities)*(self._factors.shape[1],)
                                     
        elif isinstance(data,float): 
            data_shape = None #have no information on data
            datadtype = None
        elif data is None: 
            data_shape = None #have no information on data
            datadtype = None
        else:
            raise TypeError("If provided, `data` must be a dictionary with "
                            "the format {weights: Array, "
                            "factors: Tuple[Array,SymmetricTensors],"
                            "multiplicities :Tuple}")

        return data, datadtype, data_shape
    
    def _init_data(self, data:  Dict[Tuple[int,...]], symmetrize: bool = False):
        self._data = data
        
    @property
    def num_factors(self,): 
        if self._factors is not None: 
            return len(self._factors[:,0])
        else: 
            return 0
        
    ## data getting and setting ##
    
    @property
    def weights(self):
        """Weight of each component"""
        return self._weights

    @weights.setter
    def weights(self, value):
        """Weight of each component"""
        self._weights = value
        
    @property
    def factors(self):
        """Weight of each component"""
        return self._factors

    @factors.setter
    def factors(self, value):
        """Weight of each component"""
        self._factors = value
    
    @property
    def multiplicities(self):
        """Weight of each component"""
        return self._multiplicities

    @multiplicities.setter
    def multiplicities(self, value):
        """Weight of each component"""
        self._multiplicities = value
    
    @property
    def num_arrangements(self): 
        """ number of ways to arrange factors in outer product: 
        E.G if multiplicities = (2,2), have
        AABB ABAB BBAA ABBA BAAB BABA = 6 = binom(4,2)
        possibilities. """
        if self.num_decomp_factors==1: 
            return 1
        else: 
            for i,m_ in enumerate(self.multiplicities): 
                if i ==0: 
                    continue
                if i == 1:
                    num_arrangements_ = binom(self.rank, self.multiplicities[i])
                elif i>1:
                    num_arrangements_ *= binom(self.rank - sum(self.multiplicities[1:i]), self.multiplicities[i])
            return num_arrangements_
    
    @property
    def num_decomp_factors(self): 
        """
        Quantifies how many different vectors appear in the outer products.
        E.g. 
        T = v\otimes w
        T' = v\otimes w \otimes w
        both have num_decomp_factors = 2
        But
        Q = = v\otimes w \otimes u
        has num_decomp_factors = 3. 
        """
        if self.multiplicities is None: 
            raise ValueError("multiplicities unspecified")
        else: 
            return len(self.multiplicities)
        
                
    
    def copy(self):
        """Copy"""
        other = DecompSymmetricTensor(rank = self.rank, dim=self.dim)
        other.multiplicities = self.multiplicities
        other.factors = self.factors
        other.weights = self.weights
        return other
    ## Dunder methods ##

    def __repr__(self):
        s = f"{type(self).__qualname__}(rank: {self.rank}, dim: {self.dim})"
        lines = [f"  {key}: {value}"
                 for key, value in self._data.items()]
        return "\n".join((s, *lines)) + "\n"  # Lists of SymmetricTensors look better if each tensor starts on its own line
    
    def __getitem__(self, key):
        """
        {{base_docstring}}

        .. Note:: slices are not yet supported.
        """
        if isinstance(key, str):
            return [ self.__getitem__(index) for index in \
                    self.permcls_indep_iter_repindex(σcls = key) ]

        elif isinstance(key, tuple):
            assert len(key) == self.rank, "number of indices must much rank"
            if any([isinstance(i,slice) for i in key]) or isinstance(key, slice):
                raise NotImplementedError
            if self._factors is None: 
                return 0
            elif len(self._multiplicities) == 1: 
                # do symmetrization step
                #todo: test this against numpy.prod()
                return sum( self._weights[i]*torch.prod(torch.tensor([self._factors[i,j] for j in key]))
                            for i in range(self.num_factors)) 
            elif len(self._multiplicities) == 2:
                # do symmetrization step
                return sum( self._weights[i,j] \
                           *sum(torch.prod(torch.tensor([self._factors[i,j_1] for j_1 in index_1]))\
                                *torch.prod(torch.tensor([self._factors[j,j_2] for j_2 in index_2]))\
                                for index_1, index_2 in \
                                utils.twoway_partitions_pairwise_iterator(key, self._multiplicities[0],
                                                        self._multiplicities[1], num_partitions = False))
                            for i,j in itertools.product(range(self.num_factors),repeat =2))/self.num_arrangements
            elif len(self._multiplicities) == 3:
                # do symmetrization step
                return sum( self._weights[i,j,k] \
                           *sum(torch.prod(torch.tensor([self._factors[i,j_1] for j_1 in index_1]))\
                                *torch.prod(torch.tensor([self._factors[j,j_2] for j_2 in index_2]))\
                                *torch.prod(torch.tensor([self._factors[k,j_3] for j_3 in index_3]))\
                                for index_1, index_2, index_3 in \
                                utils.nway_partitions_iterator(key, self._multiplicities, num_partitions = False))
                            for i,j,k in itertools.product(range(self.num_factors),repeat =3))/self.num_arrangements
            elif len(self._multiplicities) == 4:
                # do symmetrization step
                return sum( self._weights[i,j,k,l] \
                           *sum(torch.prod(torch.tensor([self._factors[i,j_1] for j_1 in index_1]))\
                                *torch.prod(torch.tensor([self._factors[j,j_2] for j_2 in index_2]))\
                                *torch.prod(torch.tensor([self._factors[k,j_3] for j_3 in index_3]))\
                                *torch.prod(torch.tensor([self._factors[l,j_3] for j_4 in index_4]))\
                                for index_1, index_2, index_3, index_4 in \
                                utils.nway_partitions_iterator(key, self._multiplicities, num_partitions = False))
                            for i,j,k,l in itertools.product(range(self.num_factors),repeat =4))/self.num_arrangements
            else: 
                raise NotImplementedError
        else:
            if isinstance(key, int) and self.rank == 1: 
                return sum( self._weights[i]*self._factors[i,key] \
                            for i in range(self.num_factors)) 
            else:
                raise KeyError(f"{key}")
            
    def __setitem__(self, key):
        """
        {{base_docstring}}

        .. Note:: Cannot set individual entries of a 
        DecompSymmetricTensor.
        """
        raise NotImplementedError("It is not possible to set individual"+ \
                                 "entries of DecompSymmtetricTensor.")
        
    ## data properties ##
    @property
    def dtype(self) -> np.dtype: 
        return self._dtype

    @property
    def shape(self) -> Tuple[int,...]: 
        return self.rank*(self.dim,)

    @property
    def size(self) -> int: 
        return self.num_factors*(self.dim+1)
    
    def todense(self) -> Array: 
        dense_tensor = torch.zeros((self.dim,)*self.rank)
        if self.rank > 26: 
            raise NotImplementedError
        #construct outer products with einsum
        if self.num_decomp_factors == 1: 
            if self.rank == 1: 
                return sum(self.weights[i]*self.factors[i,:] for i in range(self.num_factors))
            else: 
                for i in range(self.num_factors): 
                    outer_prod = self.factors[i,:]
                    for j in range(self.rank-1):
                        outer_prod = torch.tensordot(outer_prod,self.factors[i,:], dims =0)
                    dense_tensor += self.weights[i]* outer_prod
            return dense_tensor
        elif self.num_decomp_factors == 2:
            for i,j in itertools.product(range(self.num_factors),repeat =2):
                #symmetrize outer products:
                #loop over arrangements of factors in outer product
                for pos_1, pos_2 in utils.nway_partitions_iterator(list(range(self.rank)), self.multiplicities, num_partitions = False):
                    indices = np.array([i,]*self.rank)
                    #put jth component in pos_2th position etc.
                    indices[pos_2] = j
                    outer_prod = self.factors[indices[0],:]
                    for m in range(self.rank-1):
                        outer_prod = torch.tensordot(outer_prod,self.factors[indices[m+1],:], dims =0)
                    dense_tensor += self.weights[i,j]*outer_prod/self.num_arrangements
            return dense_tensor
        elif self.num_decomp_factors == 3:
            for i,j,k in itertools.product(range(self.num_factors),repeat =3):
                #symmetrize outer products:
                #loop over arrangements of factors in outer product
                for pos_1, pos_2, pos_3 in utils.nway_partitions_iterator(list(range(self.rank)), self.multiplicities, num_partitions = False):
                    indices = np.array([i,]*self.rank)
                    #put jth component in pos_2th position etc.
                    indices[pos_2] = j
                    indices[pos_3] = k
                    outer_prod = self.factors[indices[0],:]
                    for m in range(1,self.rank):
                        outer_prod = torch.tensordot(outer_prod,self.factors[indices[m],:], dims =0)
                    dense_tensor += self.weights[i,j,k]*outer_prod/self.num_arrangements
            return dense_tensor
        elif self.num_decomp_factors == 4:
            for i,j,k,l in itertools.product(range(self.num_factors),repeat =4):
                #symmetrize outer products:
                #loop over arrangements of factors in outer product
                for pos_1, pos_2, pos_3, pos_4 in utils.nway_partitions_iterator(list(range(self.rank)), self.multiplicities, num_partitions = False):
                    indices = np.array([i,]*self.rank)
                    #put jth component in pos_2th position etc.
                    indices[pos_2] = j
                    indices[pos_3] = k
                    indices[pos_4] = l
                    outer_prod = self.factors[indices[0],:]
                    for m in range(self.rank-1):
                        outer_prod = torch.tensordot(outer_prod,self.factors[indices[m+1],:], dims =0)
                    dense_tensor += self.weights[i,j,k,l]*outer_prod/self.num_arrangements
            return dense_tensor
        else:
            raise NotImplementedError
            
            
    def contract_all_indices_with_matrix(self, W): 
        """
        Contract all indices of the tensor with a matrix W:
        out_ijkl = \sum self_abdc W_ai W_bj W_ck W_dl 
        """
        out = self.copy()
        out.factors = torch.einsum("mj, jk -> mk", self.factors, W)
        return out

    
    def multinomial(self, x): 
        """
        Contract all indices of the tensor with a vector x:
        out = \sum self_abdc x_a x_b x_c x_d 
        """
        if not len(x) == self.dim: 
            raiseValueError("dimension must match Tensor dimension")
        num_decomp_factors = self.num_decomp_factors
        factors_times_x = torch.einsum('ij,j-> i', self.factors, x)
        out = sum( self.weights[index]*torch.prod(torch.Tensor([factors_times_x[index[m]]**k for m,k in enumerate(self.multiplicities)])) 
                  for index in itertools.product(range(self.num_factors), repeat =  self.num_decomp_factors))
        return out
                        

# %% [markdown]
# # Tensor addition
#
# ### Fully decomposed tensors
# We want to do 
# $$
# T + U = \sum_{m}^M \lambda^m t^m \otimes t^m \otimes \dots t^m + \sum_{m}^N \kappa^m u^m \otimes u^m \otimes \dots u^m
# $$
# with $t^m, u^m$ vectors. Let $T$ have $M$ factors. Let $U$ have $N$ factors. 
# For this to work, it is necessary, that the multiplicities of both classes are the same which is equivalent to the rank being the same. 
# We just define
# $$
# V = T+U = \sum_{m=1}^{N+M} \nu^m v^m \otimes v^m \otimes \dots v^m
# $$
# with 
# $$
# v^m = \begin{cases}
# t^m & m \leq M \\
# u^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
# and 
# $$
# \nu^m = \begin{cases}
# \lambda^m & m \leq M \\
# \kappa^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
# And we just initialize a new tensor with weights $\nu$ and factors $v$ and the same mutliplicity as before.
#

# %% [markdown]
#
# ### Partially decomposed tensors
# #### Bipartite Partially decomposed tensors
# For tensors of shape 
# $$
# T = \sum_{m=1}^M \lambda^{m,n} \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} 
# $$
#
# $$
# U = \sum_{m=1}^N \kappa^{m,n} \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{u^n \otimes \dots \otimes u^n}_{l \text{ times}} 
# $$
# with $t^m, u^m $ vectors. 
# For this to work, it is necessary, that the multiplicities $(k,l)$ of both tensors are the same which is a stronger condition than the rank being the same. 
#
# We just define
# $$
# P = T+U = \sum_{m=1}^{N+M} \nu^{m,n} \underbrace{p^m \otimes \dots \otimes p^m}_{k \text{ times}} \otimes \underbrace{p^n \otimes \dots \otimes p^n}_{l \text{ times}}
# $$
# with 
# $$
# p^m = \begin{cases}
# t^m & m \leq M \\
# u^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
#
# and 
# $$
# \nu^{m,n} = \begin{cases}
# \lambda^{m,n} & m,n \leq M \\
# \kappa^{m,n} & M+1 \leq m,n \leq M+N \\
# 0 & \text{else}
# \end{cases}
# $$
# And we just initialize a new tensor with weights $\nu$ and factors $v$ and the same mutliplicities as before.
#

# %% [markdown]
# #### Tripartite decomposed tensors
# For tensors of shape 
# $$
# T = \sum_{m=1}^M \lambda^{m,n,o} \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} \otimes \underbrace{t^o \otimes \dots \otimes t^o}_{j \text{ times}}
# $$
#
# $$
# U = \sum_{m=1}^N \kappa^{m,n,o} \underbrace{u^m \otimes \dots \otimes u^m}_{k \text{ times}} \otimes \underbrace{u^n \otimes \dots \otimes u^n}_{l \text{ times}} \otimes \underbrace{u^o \otimes \dots \otimes u^o}_{j \text{ times}}
# $$
# with $t^m, u^m $ vectors. 
# For this to work, it is necessary, that the multiplicities $(k,l,j)$ of both tensors are the same which is a stronger condition than the rank being the same. 
#
# We just define
# $$
# P = T+U = \sum_{m=1}^{N+M} \nu^{m,n,o} \underbrace{p^m \otimes \dots \otimes p^m}_{k \text{ times}} \otimes \underbrace{p^n \otimes \dots \otimes p^n}_{l \text{ times}} \otimes \underbrace{p^o \otimes \dots \otimes p^o}_{j \text{ times}}
# $$
# with 
# $$
# p^m = \begin{cases}
# t^m & m \leq M \\
# u^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
# and 
# $$
# \nu^{m,n,o} = \begin{cases}
# \lambda^{m,n,o} & m,n,o \leq M \\
# \kappa^{m,n,o} & M+1 \leq m,n,o \leq M+N \\
# 0 & \text{else}
# \end{cases}
# $$
# And we just initialize a new tensor with weights $\nu$ and factors $v$ and the same mutliplicities as before. 

# %% [markdown]
# #### Fourpartite decomposed tensors
# The generalization of the scheme outlined above is straightforward.

# %%
### Algebra ###
@DecompSymmetricTensor.implements_ufunc(np.add)
def symmetric_add(self, other: DecompSymmetricTensor) -> DecompSymmetricTensor: 
    #check if compatible
    if not isinstance(other, DecompSymmetricTensor): 
        raise TypeError("can only add DecompSymmetricTensor to DecompSymmetricTensor")
    if not self.rank == other.rank: 
        raise ValueError("Tensor rank must match.")
    if not self.dim == other.dim: 
        raise ValueError("Tensor dimension must match.")
    if not self.multiplicities == other.multiplicities:
        raise ValueError("Component multiplicities must match.")
    
    out = DecompSymmetricTensor(rank = self.rank, dim = self.dim)
    out.factors = torch.cat((self.factors , other.factors ), 0) 
    out.multiplicities = self.multiplicities
    #fully decomposed tensor
    if self.num_decomp_factors==1:
        out.weights =  torch.cat((self.weights, other.weights), 0) 
        return out
    #partially decomposed tensor
    if self.num_decomp_factors==2:
        out.weights = torch.zeros(out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:] = other.weights
        return out
    if self.num_decomp_factors==3:
        out.weights = torch.zeros(out.num_factors,out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:,self.num_factors:] = other.weights
        return out
    if self.num_decomp_factors==4:
        out.weights = torch.zeros(out.num_factors,out.num_factors,out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:,self.num_factors:,self.num_factors:] = other.weights
        return out
    else: 
        raise NotImplementedError



# %% [markdown]
# ## Outer product
#
# ### Outer product for fully decomposed Tensors
# Suppose we have 
#
# $$
# T = \sum_{m} \lambda^m t^m \otimes \dots \otimes t^m
# $$
# with $t^m$ vectors and $T$ has rank $\tau$.
# and
# $$
# U = \sum_{m} \kappa^m u^m \otimes \dots \otimes  u^m
# $$
# with $u^m$ vectors and $T$ has rank $\nu$.
#
# Now we want to compute
# $$
# V = T \otimes U \\
# = \sum_{m,n} \lambda^m \kappa^n t^m \otimes t^m \otimes \dots t^m \otimes u^n \otimes \dots \otimes  u^n
# $$
# with $\nu^{m,n} = \lambda^m \kappa^n $ a matrix.
# We denote by $M,N$ the number of factors in $T,V$.
# To place everything on a common index, we set
# $t^{M+1},\dots, t^{M+N} = u^1,\dots u^N$ and furthermore set the weights to be
# $$
# \Lambda^{1:M, M+1:M+N} = \nu
# $$
# With all other entries of $\Lambda =0$. 
#

# %% [markdown] tags=[]
#
# ### Outer product for partially decomposed tensors
#
# #### Bipartite and fully decomposed tensor
# Suppose we have 
#
# $$
# T = \sum_{m=1}^M \lambda^{m,n} \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} 
# $$
# with $t^m$ vectors and $T$ has rank $\tau$.
# and
# $$
# U = \sum_{m=1}^N \kappa^m \underbrace{u^m \otimes \dots \otimes u^m}_{j \text{ times}}
# $$
# with $u^m$ vectors and $T$ has rank $\nu$.
#
# The result will be a Tensor of multiplicity $(k,l,j)$.
# $$
# V = T \otimes U \\
# = \sum_{m=1,n=1}^M \sum_{o=1}^N \lambda^{m,n} \kappa^o \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} \otimes \underbrace{u^o \otimes \dots \otimes u^o}_{j \text{ times}} \\
# = \sum_{m=1,n=1,o=1}^{M+N} \nu^{m,n,o} \underbrace{v^m \otimes \dots \otimes v^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots \otimes v^n}_{l \text{ times}} \otimes \underbrace{v^o \otimes \dots \otimes v^o}_{j \text{ times}}
# $$
# with
# $$
# v^m := \begin{cases}
# t^m & m \leq M \\
# u^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
# and 
# $$
# \nu^{m,n,o} := \begin{cases}
# \lambda^{m,n} \kappa^o & m,n \leq M \text{ and } M+1 \leq o \leq M+N \\
# 0 & \text{else}
# \end{cases}
# $$
# #### Bipartite and bipartite tensor
# Suppose we have 
#
# $$
# T = \sum_{m=1}^M \lambda^{m,n} \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} 
# $$
# with $t^m$ vectors and $T$ has rank $\tau$.
# and
# $$
# U = \sum_{m=1,n=1}^N \kappa^{m,n} \underbrace{u^m \otimes \dots \otimes u^m}_{j \text{ times}}\otimes \underbrace{u^n \otimes \dots \otimes u^n}_{i \text{ times}}
# $$
# with $u^m$ vectors and $T$ has rank $\nu$.
#
# The result will be a Tensor of multiplicity $(k,l,j,i)$.
# $$
# V = T \otimes U \\
# = \sum_{m,n=1}^M \sum_{o,p=1}^N \lambda^{m,n} \kappa^{o,p} \underbrace{t^m \otimes \dots \otimes t^m}_{k \text{ times}} \otimes \underbrace{t^n \otimes \dots \otimes t^n}_{l \text{ times}} \otimes \underbrace{u^o \otimes \dots \otimes u^o}_{j \text{ times}}  \otimes \underbrace{u^p \otimes \dots \otimes u^p}_{i \text{ times}}\\
# = \sum_{m,n,o,p=1}^{M+N} \nu^{m,n,o,p} \kappa^{o,p} \underbrace{v^m \otimes \dots \otimes v^m}_{k \text{ times}} \otimes \underbrace{v^n \otimes \dots \otimes v^n}_{l \text{ times}} \otimes \underbrace{v^o \otimes \dots \otimes v^o}_{j \text{ times}} \otimes \underbrace{v^p \otimes \dots \otimes v^p}_{i \text{ times}}
# $$
# with
# $$
# v^m := \begin{cases}
# t^m & m \leq M \\
# u^m & M+1 \leq m \leq M+N
# \end{cases}
# $$
# and 
# $$
# \nu^{m,n,o,p} := \begin{cases}
# \lambda^{m,n} \kappa^{o,p} & m,n \leq M \text{ and } M+1 \leq o,p \leq M+N \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ### Tripartite and fully decomposed Tensor
#
# The generalization of the scheme above is straightforward. 

# %%
@DecompSymmetricTensor.implements(np.outer)
def symmetric_outer(self,other): 
    #check if compatible
    if not isinstance(other, DecompSymmetricTensor): 
        raise TypeError("can only tensordot DecompSymmetricTensor to DecompSymmetricTensor")
    if not self.dim == other.dim: 
        raise ValueError("Tensor dimension must match.")
    if not self.num_decomp_factors<=3:
        raise NotImplementedError
    if not other.num_decomp_factors+self.num_decomp_factors<=4:
        raise NotImplementedError
    if other.num_decomp_factors> self.num_decomp_factors: 
        return np.outer(other, self)
    
    out = DecompSymmetricTensor(rank = self.rank+other.rank, dim = self.dim)
    out.factors = torch.cat((self.factors , other.factors ), 0) 
    #fully decomposed 
    if self.num_decomp_factors ==1 and other.num_decomp_factors==1:
        #higher multiplicities come first
        if other.multiplicities[0] > self.multiplicities[0]: 
            return np.outer(other,self)
        out.multiplicities = (self.multiplicities[0], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*2) 
        out.weights[:self.num_factors,self.num_factors:] = torch.einsum('m,n->mn', self.weights, other.weights)
    #bipartite and fully decomposed
    elif self.num_decomp_factors == 2 and other.num_decomp_factors==1:
        out.multiplicities = (self.multiplicities[0], self.multiplicities[1], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*3) 
        out.weights[:self.num_factors,:self.num_factors,self.num_factors:] = torch.einsum('mn,o->mno', self.weights, other.weights)
    #tripartite and fully decomposed
    elif self.num_decomp_factors == 3 and other.num_decomp_factors==1:
        out.multiplicities = (self.multiplicities[0], self.multiplicities[1], self.multiplicities[2], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*4) 
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors,self.num_factors:] = torch.einsum('mno,p->mnop', self.weights, other.weights)
    #bipartite and bipartite decomposed
    elif self.num_decomp_factors == 2 and other.num_decomp_factors==2:
        out.multiplicities = (self.multiplicities[0], self.multiplicities[1], other.multiplicities[0], other.multiplicities[1])
        out.weights = torch.zeros((out.num_factors,)*4) 
        out.weights[:self.num_factors,:self.num_factors,self.num_factors:,self.num_factors:] = torch.einsum('mn,op->mnop', self.weights, other.weights)
    return out


@DecompSymmetricTensor.implements_ufunc.outer(np.multiply)
def symmetric_multiply_outer(self,other): 
    return np.outer(self,other)



# %% [markdown]
#
# ## Tensordot
#
# ### Single contraction 
# #### for fully decomposed tensors
# Suppose we have 
#
# $$
# T = \sum_{m} \lambda^m t^m \otimes t^m \otimes \dots t^m
# $$
# with $t^m$ vectors and $T$ has rank $\tau$.
# and
# $$
# U = \sum_{m} \kappa^m u^m \otimes u^m \otimes \dots u^m
# $$
# with $u^m$ vectors and $T$ has rank $\nu$.
#
# Now we want to contract to 
# $$
# V_{i_1,...i_{\tau+\nu-2}} = \sum_j T_{i_1,...i_{\tau-1},j} U_{j,i_{tau},...i_{\tau+\nu-2}} \\
# = \sum_m \sum_n \lambda^m \kappa^n \sum_j t^m_j u^n_j \left(\underbrace{t^m \otimes \dots \otimes t^m}_{\tau-1 \text{ times}}  \underbrace{u^n\otimes \dots \otimes u^n}_{\mu-1 \text{ times}} \right)_{i_1,...i_{\tau+\nu-2}} \\
# = \sum_{m,n} \sum_n \nu^{m,n}\left(\underbrace{t^m \otimes \dots \otimes t^m}_{\tau-1 \text{ times}}  \underbrace{u^n\otimes \dots \otimes u^n}_{\mu-1 \text{ times}} \right)_{i_1,...i_{\tau+\nu-2}}
# $$
# with $\nu^{m,n} = \lambda^m \kappa^n \sum_j t^m_j u^n_j$ a matrix.
# We denote by $M,N$ the number of factors in $T,V$.
# To place everything on a common index, we set
# $t^{M+1},\dots, t^{M+N} = u^1,\dots u^N$ and furthermore set the weights to be
# $$
# \Lambda^{1:M, M+1:M+N} = \nu
# $$
# With all other entries of $\Lambda =0$.

# %% [markdown]
# ### Double contraction for fully decomposed tensors
#
# As above, but we evaluate
# $$
# V_{i_1,...i_{\tau+\nu-2}} = \sum_j T_{i_1,...i_{\tau-2},j,k} V_{j,k,i_{tau},...i_{\tau+\nu-4}} \\
# = \sum_m \sum_n \lambda^m \kappa^n \sum_j t^m_j u^n_j \sum_k t^m_k u^n_k \left(\underbrace{t^m \otimes \dots \otimes t^m}_{\tau-1 \text{ times}}  \underbrace{u^n\otimes \dots \otimes u^n}_{\mu-1 \text{ times}} \right)_{i_1,...i_{\tau+\nu-2}} \\
# = \sum_{m,n} \sum_n \tilde{\nu}^{m,n}\left(\underbrace{t^m \otimes \dots \otimes t^m}_{\tau-1 \text{ times}}  \underbrace{u^n\otimes \dots \otimes u^n}_{\mu-1 \text{ times}} \right)_{i_1,...i_{\tau+\nu-2}}
# $$
# with $\tilde{\nu}^{m,n} = \lambda^m \kappa^n (\sum_j t^m_j u^n_j)^2$ a matrix.
# We denote by $M,N$ the number of factors in $T,V$.
# To place everything on a common index, we set
# $t^{M+1},\dots, t^{M+N} = u^1,\dots u^N$ and furthermore set the weights to be
# $$
# \Lambda^{1:M, M+1:M+N} = \tilde{\nu}
# $$
# With all other entries of $\Lambda =0$. 
# In other words, $\Lambda$ has block structure: 
# $$
# \Lambda = \begin{pmatrix}
# 0 & \tilde{\nu}\\
# 0 & 0
# \end{pmatrix}
# $$

# %%
@DecompSymmetricTensor.implements(np.tensordot)
def symmetric_tensordot(self, other: DecompSymmetricTensor, axes: int = 2) -> Union[DecompSymmetricTensor, float]: 
    #check if compatible
    if axes == 0: 
        return np.outer(self,other)
    if not isinstance(other, DecompSymmetricTensor): 
        raise TypeError("can only tensordot DecompSymmetricTensor to DecompSymmetricTensor")
    if not self.dim == other.dim: 
        raise ValueError("Tensor dimension must match.")
    if not self.num_decomp_factors==1:
        raise NotImplementedError("Tensordot is currently only available for fully decomposed tensors")
    if not other.num_decomp_factors==1:
        raise NotImplementedError("Tensordot is currently only available for fully decomposed tensors")
    if other.multiplicities[0] > self.multiplicities[0]: 
        return np.tensordot(other,self, axes = axes)
    
    if axes ==1:
        if self.multiplicities[0]>1:
            out = DecompSymmetricTensor(rank = self.rank+other.rank-2, dim = self.dim)
            if other.multiplicities[0]>1:
                out.multiplicities = (self.multiplicities[0]-1,other.multiplicities[0]-1)
                out.factors = torch.cat((self.factors , other.factors ), 0) 
                out.weights = torch.zeros((self.num_factors + other.num_factors,
                                               self.num_factors + other.num_factors)) 
                #equivalent to \nu in desc. above
                out.weights[:self.num_factors,self.num_factors:] = torch.einsum('m,n,mj,nj->mn', \
                                            self.weights, other.weights, self.factors, other.factors)
                return out
            else: 
                #second tensor gets completely consumed
                out.multiplicities = (self.multiplicities[0],)
                out.factors = self.factors 
                #equivalent to \nu in desc. above
                out.weights = torch.einsum('m,n,mj,nj->m', \
                                           self.weights, other.weights, 
                                           self.factors, other.factors)
                return out
        elif self.multiplicities[0]==1:
            assert other.multiplicities[0] <=1
            out = torch.einsum('m,n,mj,nj->',self.weights, other.weights, 
                              self.factors, other.factors)
            return out
    #contraction over two or more indices
    elif axes >=2: 
        assert self.rank >=2, "Can only do double contraction with rank >= 2 tensor."
        assert other.rank >=2, "Can only do double contraction with rank >= 2 tensor."
        
        if self.multiplicities[0] >axes and other.multiplicities[0]>axes:
            # structure of contraction \sum_{jk} A_...jk B_...jk
            out = DecompSymmetricTensor(rank = self.rank+other.rank-4, dim = self.dim)
            out.multiplicities = (self.multiplicities[0]-axes,other.multiplicities[0]-axes)
            out.factors = torch.cat((self.factors , other.factors ), 0) 
            out.weights = torch.zeros((self.num_factors + other.num_factors,
                                       self.num_factors + other.num_factors))
            lambda_without_weighfactors = torch.einsum('mj,nj->mn', self.factors, other.factors)**axes
            out.weights[:self.num_factors,self.num_factors:] = \
                torch.einsum('m,n,mn->mn', self.weights, other.weights, lambda_without_weighfactors)
            
        elif other.multiplicities[0] == axes: 
            if self.multiplicities[0] > axes:
                # structure of contraction \sum_{jk} A_...jk B_jk
                # other.factors consumed by sum, only outer products of first component remain. 
                out = DecompSymmetricTensor(rank = self.rank+other.rank-4, dim = self.dim)
                out.multiplicities = (self.multiplicities[0]-axes,)
                out.factors = self.factors
                lambda_without_weighfactors = torch.einsum('mj,nj->mn', \
                                                           self.factors, other.factors)**axes
                out.weights = torch.einsum('m,n,mn->m', self.weights, other.weights, lambda_without_weighfactors)
            if self.multiplicities[0] == axes:
                # structure of contraction \sum_{jk} A_jk B_jk
                # result is therefore a float.
                lambda_without_weighfactors = torch.einsum('mj,nj->mn', \
                                                           self.factors, other.factors)**axes
                out = torch.einsum('m,n,mn->', self.weights, other.weights, lambda_without_weighfactors)
            
    return out


# %% [markdown]
# ## Tensor comparison

# %%
@DecompSymmetricTensor.implements(np.allclose)
def symmetric_tensorcompare(self, other: DecompSymmetricTensor, axes: int = 2) -> Union[DecompSymmetricTensor, float]: 
    if not isinstance(other, DecompSymmetricTensor): 
        raise TypeError("can only compare DecompSymmetricTensor to DecompSymmetricTensor")
    if not self.dim == other.dim: 
        raise ValueError("Tensor dimension must match.")
    if not self.rank == other.rank: 
        raise ValueError("Tensor dimension must match.")
    self_flat = torch.Tensor([self[index] for index in self.indep_iter_repindex()])
    other_flat = torch.Tensor([other[index] for index in other.indep_iter_repindex()])
    return torch.allclose(self_flat,other_flat)



# %% [markdown]
# # Tests

# %% [markdown]
# ## Instantiation, getting and setting of weights, factors and multiplicities

# %%
if __name__ == "__main__":
    # instantiation of vector
    A = DecompSymmetricTensor(rank = 1, dim =10) 
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
    import pytest
    from collections import Counter
    from statGLOW.utils import does_not_warn
    from symtensor.utils import symmetrize
    import itertools

    def test_tensors() -> Generator:
        for d, r in itertools.product([2, 3, 4, 6, 8], [2, 3, 4, 5, 6]):
            yield DecompSymmetricTensor(rank=r, dim=d)
            
    def two_comp_test_tensor(d,r):
        A = DecompSymmetricTensor(rank=r, dim=d)
        A.weights = torch.randn(size =(2,))
        A.factors =  torch.randn(size =(2,d))
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
    #assert len(A['iii']) == 5
    #assert A['iii'] == [0,]*5
    
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

    assert torch.allclose(C_2.todense(), A_2.todense()+B_2.todense())


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
    assert np.allclose(C.todense().numpy(), sym_C_dense)

    # rank 3 tripartite & rank 2 fully decomposed
    A = three_factor_test_tensor(2,3, q = 1)
    B = two_comp_test_tensor(2,2)
    C = np.outer(A,B)
    C_dense = np.tensordot(A.todense(), B.todense(), axes=0)
    sym_C_dense = utils.symmetrize(C_dense)
    assert np.allclose(C.todense().numpy(), sym_C_dense)

    # rank 2 bipartite & rank 2 bipartite
    A1 = two_factor_test_tensor(2,2, q = 1)
    B1 = two_factor_test_tensor(2,2, q = 1)
    C1 = np.outer(A,B)
    C1_dense = np.tensordot(A.todense(), B.todense(), axes=0)
    sym_C1_dense = utils.symmetrize(C1_dense)
    assert np.allclose(C1.todense().numpy(), sym_C1_dense)
    

# %% [markdown]
# ## Tensordot
#

    # %%
    d = 2
    for r in range(2,4): 
        tensor_1 = two_comp_test_tensor(d,r)
        for r_1 in range(2,4):
            print(r,r_1)
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



# %% [markdown]
# ### Contraction with matrix along all indices

# %%
if __name__ == "__main__":
    d = 3
    r = 2 
    A = two_comp_test_tensor(d,r)
    W = torch.randn(size=(3,3))
    B = A.contract_all_indices_with_matrix(W).todense().numpy()
    assert np.isclose(B, symmetrize(torch.einsum('ab, ai,bj -> ij', A.todense(), W,W).numpy())).all()
    r = 3
    A = two_comp_test_tensor(d,r)
    W = torch.randn(size=(3,3))
    assert np.isclose(A.contract_all_indices_with_matrix(W).todense(), symmetrize(torch.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W).numpy())).all()
    
    r = 2
    A = two_factor_test_tensor(d,r, q = 1)
    W = torch.randn(size=(3,3))
    B = A.contract_all_indices_with_matrix(W).todense().numpy()
    assert np.isclose(B, symmetrize(torch.einsum('ab, ai,bj -> ij', A.todense(), W,W).numpy())).all()
    r = 3
    A = two_factor_test_tensor(d,r, q = 1)
    W = torch.randn(size=(3,3))
    assert np.isclose(A.contract_all_indices_with_matrix(W).todense(), symmetrize(torch.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W).numpy())).all()


# %% [markdown]
# ### Multinomial 

# %%
    
    d = 3
    r =3
    A = two_comp_test_tensor(d,r)
    for i in range(d): 
        x = torch.zeros(d)
        x[i] =1
        assert torch.isclose(A[(i,)*r], A.multinomial(x))
    
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
    
