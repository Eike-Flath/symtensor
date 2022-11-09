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
# Script only imports
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
        """Factors in outer product"""
        return self._factors

    @factors.setter
    def factors(self, value):
        """Factors in outer product"""
        self._factors = value
    
    @property
    def multiplicities(self):
        """Number of repeats of factors 
        in outer product"""
        return self._multiplicities

    @multiplicities.setter
    def multiplicities(self, value):
        """Number of repeats of factors 
        in outer product"""
        self._multiplicities = value
    
    @property
    def num_arrangements(self): 
        """ number of ways to arrange factors in outer product: 
        E.G if multiplicities = (2,2), have
        AABB ABAB BBAA ABBA BAAB BABA = 6 = binom(4,2)
        possibilities. """
        if self.num_indep_factors==1: 
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
    def num_indep_factors(self): 
        """
        Quantifies how many different vectors appear in the outer products.
        E.g. 
        T = v\otimes w
        T' = v\otimes w \otimes w
        both have num_indep_factors = 2
        But
        Q = = v\otimes w \otimes u
        has num_indep_factors = 3. 
        """
        if self.multiplicities is None: 
            raise ValueError("multiplicities unspecified")
        else: 
            return len(self.multiplicities)
        
    
    def split_factors(self, pos): 
        """
        Create equivalent tensor with different
        multiplicities, where the `pos`th multiplicity 
        is split of. 
        
        Inputs: 
        =======
        pos: int
            which multiplicity to split
        
        E.g. 
        mult_old = (2,1,1)
        pos = 0
        -> new_multiplicity = (1,1,1,1)
        """
        assert self.multiplicities[pos] > 1, "cannot split factor with multiplicity one"
        
        letters = 'abdcefghijklmnopqrstuvwxy' # keep z seperate
        if self.num_indep_factors > len(letters): 
            #absurdly many factors
            raise NotImplemenetedError
        
        #update weights to incorporate multiplicity split
        indices_before_pos = letters[:pos+1]
        if self.num_indep_factors > pos+1:
            indices_after_pos = letters[pos+1:self.num_indep_factors]
        else: 
            indices_after_pos = ''
        indices_result = indices_before_pos + 'z' + indices_after_pos
        indices_in = indices_before_pos + indices_after_pos
        indices_delta = indices_before_pos[-1] + 'z'
        delta_iz = torch.eye(self.num_factors)
    
        self.weights = torch.einsum(indices_in +', '+ indices_delta + '-> ' + indices_result, 
                                   self.weights, delta_iz )
        
        
        new_multiplicities = self.multiplicities[:pos] \
                            + (self.multiplicities[pos]-1,1,) \
                            + self.multiplicities[pos+1:]
        self.multiplicities = new_multiplicities
        
        #TODO: make splitting off of factor with multiplicity >1
                    #possible 
    
    def match_multiplicities(self, mult): 
        """
        Create equivalent tensor with
        multiplicities equal to `mult`.
        
        Inputs: 
        =======
        m: tuple[int]
            new multiplicity
        
        """
        assert sum(mult)== self.rank, "new multiplicity does not match rank"
        assert len(mult) >= self.num_indep_factors, "can only increase number of independent factors"
        
        max_num_splits = 10 #safeguard against too many splits
        num_splits = 0
        while self.multiplicities != mult and num_splits < max_num_splits:
            for i,m in enumerate(mult):
                if self.multiplicities[i] > m: 
                    self.split_factors(i)
                    num_splits += 1
                    break
                elif self.multiplicities[i] == m:
                    continue
                elif self.multiplicities[i] < m:
                    raise ValueError("Can only reduce individual multiplicity factors, not increase them")
        if num_splits == max_num_splits: 
            raise ValueError("maximum number of splits reached. Reduce number of independent factors")
    
    def find_common_multiplicities(self,other):
        """
        Find min. number of multiplicities 
        such that multiplicities are equal.
        
        Inputs: 
        =======
        other: DecompSymmetricTensor
            Tensor to which multiplicities
            should be matched.
        
        Returns: 
        ========
        m: tuple[int]
            new multiplicity
        
        """
        assert isinstance(other, DecompSymmetricTensor), "can only match multiplicities between decomp tensors"
        if not self.rank == other.rank: 
            raise ValueError("Tensor ranks must be equal.")
            
        if self.multiplicities == other.multiplicities:
            return self.multiplicities
        elif other.num_indep_factors < self.num_indep_factors:
            return other.find_common_multiplicities(self)
        else:
            max_num_splits = 10 #safeguard against too many splits
            num_splits = 0
            new_mult = other.multiplicities
            while len(new_mult) < self.num_indep_factors and num_splits < max_num_splits:
                print(new_mult)
                for i,m in enumerate(self.multiplicities):
                    if m == new_mult[i]: 
                        continue
                    if self.multiplicities[i] > new_mult[i]: 
                        new_mult = new_mult[:i] \
                            + (new_mult[i]-1,1,) \
                            + new_mult[i+1:]
                        num_splits += 1
                        break
            if num_splits == max_num_splits: 
                raise ValueError("maximum number of splits reached. Reduce number of independent factors")
            else: 
                return new_mult
                        
                
                
    
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
        if self.num_indep_factors == 1: 
            if self.rank == 1: 
                return sum(self.weights[i]*self.factors[i,:] for i in range(self.num_factors))
            else: 
                for i in range(self.num_factors): 
                    outer_prod = self.factors[i,:]
                    for j in range(self.rank-1):
                        outer_prod = torch.tensordot(outer_prod,self.factors[i,:], dims =0)
                    dense_tensor += self.weights[i]* outer_prod
            return dense_tensor
        elif self.num_indep_factors == 2:
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
        elif self.num_indep_factors == 3:
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
        elif self.num_indep_factors == 4:
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
        num_indep_factors = self.num_indep_factors
        factors_times_x = torch.einsum('ij,j-> i', self.factors, x)
        out = sum( self.weights[index]*torch.prod(torch.Tensor([factors_times_x[index[m]]**k for m,k in enumerate(self.multiplicities)])) 
                  for index in itertools.product(range(self.num_factors), repeat =  self.num_indep_factors))
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
        #split factors such that multiplicities match
        new_multiplicities = self.find_common_multiplicities(other)
        self_1 = self.copy()
        self_1.match_multiplicities(new_multiplicities)
        other_1 = other.copy()
        other_1.match_multiplicities(new_multiplicities)
        return symmetric_add(self_1, other_1)
    
    out = DecompSymmetricTensor(rank = self.rank, dim = self.dim)
    out.factors = torch.cat((self.factors , other.factors ), 0) 
    out.multiplicities = self.multiplicities
    #fully decomposed tensor
    if self.num_indep_factors==1:
        out.weights =  torch.cat((self.weights, other.weights), 0) 
        return out
    #partially decomposed tensor
    if self.num_indep_factors==2:
        out.weights = torch.zeros(out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:] = other.weights
        return out
    if self.num_indep_factors==3:
        out.weights = torch.zeros(out.num_factors,out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:,self.num_factors:] = other.weights
        return out
    if self.num_indep_factors==4:
        out.weights = torch.zeros(out.num_factors,out.num_factors,out.num_factors,out.num_factors)
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors,:self.num_factors] = self.weights
        out.weights[self.num_factors:,self.num_factors:,self.num_factors:,self.num_factors:] = other.weights
        return out
    else: 
        raise NotImplementedError



# %%
@DecompSymmetricTensor.implements_ufunc(np.multiply)
def symmetric_multiply(self, other:Number) -> DecompSymmetricTensor: 
    #check if compatible
    if not isinstance(other, float) or isinstance(other, int):
        if isinstance(self, float) or isinstance(self, int):
            return symmetric_multiply(other, self)
        else:
            raise TypeError("Can only multiply DecompSymmetricTensors by int or float")
    
    out = self.copy()
    out.weights *= other
    return out


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
    if not self.num_indep_factors<=3:
        raise NotImplementedError
    if not other.num_indep_factors+self.num_indep_factors<=4:
        raise NotImplementedError
    if other.num_indep_factors> self.num_indep_factors: 
        return np.outer(other, self)
    
    out = DecompSymmetricTensor(rank = self.rank+other.rank, dim = self.dim)
    out.factors = torch.cat((self.factors , other.factors ), 0) 
    #fully decomposed 
    if self.num_indep_factors ==1 and other.num_indep_factors==1:
        #higher multiplicities come first
        if other.multiplicities[0] > self.multiplicities[0]: 
            return np.outer(other,self)
        out.multiplicities = (self.multiplicities[0], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*2) 
        out.weights[:self.num_factors,self.num_factors:] = torch.einsum('m,n->mn', self.weights, other.weights)
    #bipartite and fully decomposed
    elif self.num_indep_factors == 2 and other.num_indep_factors==1:
        out.multiplicities = (self.multiplicities[0], self.multiplicities[1], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*3) 
        out.weights[:self.num_factors,:self.num_factors,self.num_factors:] = torch.einsum('mn,o->mno', self.weights, other.weights)
    #tripartite and fully decomposed
    elif self.num_indep_factors == 3 and other.num_indep_factors==1:
        out.multiplicities = (self.multiplicities[0], self.multiplicities[1], self.multiplicities[2], other.multiplicities[0])
        out.weights = torch.zeros((out.num_factors,)*4) 
        out.weights[:self.num_factors,:self.num_factors,:self.num_factors,self.num_factors:] = torch.einsum('mno,p->mnop', self.weights, other.weights)
    #bipartite and bipartite decomposed
    elif self.num_indep_factors == 2 and other.num_indep_factors==2:
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
    if not self.num_indep_factors==1 and  axes >1:
        raise NotImplementedError("Double contraction is currently only available for fully decomposed tensors")
    if not other.num_indep_factors==1:
        raise NotImplementedError("Tensordot is currently only available for fully decomposed tensors")
    
    if axes ==1:
        #fully decomp tensors 
        if self.num_indep_factors==1:
            if other.multiplicities[0] > self.multiplicities[0]: 
                return np.tensordot(other,self, axes = axes)
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
                    out.multiplicities = (self.multiplicities[0]-1,)
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
        #partially decomp tensor and fully decomp tensor
        elif self.num_indep_factors ==1 and other.num_indep_factors > 1:
            return symmetric_tensordot(other,self)
        elif self.num_indep_factors >1 and other.num_indep_factors ==1: 
            # in symmetrized version, must sum over all possible arrangements of factors. 
            # contraction therefore happens with each factor multiple times, according to which factor 
            # is in last position.
            # Therefore: do contraction with each factor and weight with multiplicity of factor /sum multiplicites.
            out_tensors = []
            for i,m in enumerate(self.multiplicities):
                out = DecompSymmetricTensor(rank = self.rank+other.rank-2, dim = self.dim)
                out.factors = torch.cat((self.factors , other.factors ), 0) 
                letters = 'abdcefghjklmnopqrstuvwx' # keep i,y,z seperate
                if self.num_indep_factors > len(letters): 
                    raise NotImplemenetedError('absurdly many factors')

                indices_before_i = letters[:i]
                if self.num_indep_factors > i+1:
                    indices_after_i = letters[i+1:self.num_indep_factors]
                else: 
                    indices_after_i = ''
                
                #contraction leaves all indices intact
                if self.multiplicities[i]>1 and other.multiplicities[0] >1:
                    out.multiplicities = self.multiplicities[:i] \
                                        + (self.multiplicities[i]-1,) \
                                        + self.multiplicities[i+1:] \
                                        + (other.multiplicities[0]-1,)
                    
                    indices_result = indices_before_i+'i'+ indices_after_i +'z'
                    indices_in = indices_before_i + 'i'  + indices_after_i 
                    out.weights = torch.zeros((out.num_factors,)*out.num_indep_factors)
                    out.weights[(slice(self.num_factors),)*(self.num_indep_factors)+(slice(self.num_factors,None),)] =  \
                                torch.einsum(indices_in +', '+ 'iy' + ', '+ 'zy' + ', '+'z' '-> ' + indices_result, 
                                            self.weights, self.factors, other.factors, other.weights)
                #contraction consumes factor
                if self.multiplicities[i]==1 and other.multiplicities[0] >1:
                    out.multiplicities = self.multiplicities[:i] \
                                        + self.multiplicities[i+1:] \
                                        + (other.multiplicities[0]-1,)
                    
                    indices_result = indices_before_i+ indices_after_i +'z'
                    indices_in = indices_before_i + 'i'  + indices_after_i 
                    out.weights = torch.zeros((out.num_factors,)*out.num_indep_factors)
                    out.weights[(slice(self.num_factors),)*(self.num_indep_factors-1)+(slice(self.num_factors,None),)] =  \
                                torch.einsum(indices_in +', '+ 'iy' + ', '+ 'zy' + ', '+'z' '-> ' + indices_result, 
                                            self.weights, self.factors, other.factors, other.weights)
                #contraction consumes factor of other
                if self.multiplicities[i]>1 and other.multiplicities[0] ==1:
                    out.multiplicities = self.multiplicities[:i] \
                                        + (self.multiplicities[i]-1,) \
                                        + self.multiplicities[i+1:] 
                    out.factors = self.factors
                    
                    indices_result = indices_before_i+'i'+ indices_after_i 
                    indices_in = indices_before_i + 'i'  + indices_after_i 
                    out.weights = torch.zeros((out.num_factors,)*out.num_indep_factors)
                    out.weights[(slice(self.num_factors),)*(self.num_indep_factors)] =  \
                                torch.einsum(indices_in +', '+ 'iy' + ', '+ 'zy' + ', '+'z' '-> ' + indices_result, 
                                            self.weights, self.factors, other.factors, other.weights)
                #contraction consumes factor and factor of other
                if self.multiplicities[i]==1 and other.multiplicities[0] ==1:
                    out.multiplicities = self.multiplicities[:i] \
                                        + self.multiplicities[i+1:] 
                    out.factors = self.factors 
                    
                    indices_result = indices_before_i+ indices_after_i 
                    indices_in = indices_before_i + 'i'  + indices_after_i 
                    out.weights = torch.zeros((out.num_factors,)*out.num_indep_factors)
                    out.weights[(slice(self.num_factors),)*(self.num_indep_factors-1)] =  \
                                torch.einsum(indices_in +', '+ 'iy' + ', '+ 'zy' + ', '+'z' '-> ' + indices_result, 
                                            self.weights, self.factors, other.factors, other.weights)
                 #weight with relative number of occurences in symmetrization sum
                out_tensors += [np.multiply(out,m*1.0/self.rank)]
            
            #sum up to symmetrized result
            result = out_tensors[0]
            for tensor in out_tensors[1:]: 
                result = result + tensor
            return result 
        else:
            raise NotImplementedError("Contraction of two partially decomposed tensors not yet possible")
            
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
            return out
            
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
                return out
            elif self.multiplicities[0] == axes:
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

