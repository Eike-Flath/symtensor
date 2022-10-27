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

from typing import Optional, ClassVar, Union
from scityping import Number
from scityping.numpy import Array, DType
from scityping.torch import TorchTensor

# %% tags=["active-ipynb", "remove-input"]
# # Module only imports
# from symtensor.symtensor.torch_symtensor import TorchSymmetricTensor
# from symtensor.symtensor.permcls_symtensor import PermClsSymmetricTensor

# %% tags=["active-py", "remove-cell"]
# Script only imports
from .base import SymmetricTensor, array_function_dispatch
from .torch_symtensor import TorchSymmetricTensor
from .permcls_symtensor import PermClsSymmetricTensor
from . import base
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
        super().__init__(rank = rank, dim = dim , data = data, dtype =dtype)
        # Sets rank, dim, device, _σclass_sizes, _σclass_multiplicities
        # Calls _validate_data
        # Sets _dtype
        # Calls _init_data

    @classmethod
    def _validate_data(cls,
                       data: Optional[Dict[Union[Tuple[int,...], str]]]
                       ) -> Dict[Tuple[int,...], Array[Any,1]]:
        """
        {{base_docstring}}

        For the case of DecompSymmetricTensor, this specifically means
        - Standardizing the `data` argument to a dict of 
            {weights: Array, components: Tuple[Array,SymmetricTensors], multiplicities :Tuple}
        - Asserting that all array dtypes are numeric
        - Infer the dtype by applying type promotion on data dtypes
        """
        if isinstance(data, np.ndarray):
            raise NotImplementedError("Casting plain arrays to DecompSymmetricTensor "
                                      "is not possible.")
        elif isinstance(data, dict):
            # ensure data has right keys and format
            for key in list(data):
                if key == "weights": 
                    data[key] = self._validate_dataarray(data[key])
                    self.weights = data[key]
                    datadtype = np.result_type(data.values())
                elif key == "multiplicities": 
                    if not isinstance(data[key],tuple): 
                        raise TypeError("multiplicities must be of type tuple[int]")
                    if not (isinstance(k,int) for k in data[key]).all(): 
                        raise TypeError("multiplicities must be of type tuple[int]")
                    if not (k>0 for k in data[key]).all():
                        raise ValueError("multiplicities must be > 0.")
                    self.multiplicities = data[key]
                elif key == "components": 
                    if not isinstance(data[key],tuple): 
                        raise TypeError("components must be of type tuple[array,SymmetricTensor]")
                    if not ((isinstance(v,torch.Tensor) or isinstance(v,SymmetricTensor)) for v in data[key]).all(): 
                        raise TypeError("components must be of type tuple[array,SymmetricTensor]")
                    self.components = data[key]
                else: 
                    raise ValueError(f"`data` contains the key '{key}'."
                                     "Permitted data keys are `mulitplicities`,"
                                     " `weights` and `components`.")
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
            assert len(self.multiplicities) == len(self.weights.shape), "multiplicities do not match weights."
            data_shape = sum(self.multiplicities)*(self.components.shape[1],)
                                     
        elif isinstance(data,float): 
            data_shape = None #have no information on data
            datadtype = None
        elif data is None: 
            data_shape = None #have no information on data
            datadtype = None
        else:
            raise TypeError("If provided, `data` must be a dictionary with "
                            "the format {weights: Array, "
                            "components: Tuple[Array,SymmetricTensors],"
                            "multiplicities :Tuple}")

        return data, datadtype, data_shape
    
    def _init_data(self, data:  Dict[Tuple[int,...]], symmetrize: bool = False):
        self._data = data
    
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
            raise NotImplementedError

        elif isinstance(key, tuple):
            if any([isinstance(i,slice) for i in key]) or isinstance(key, slice):
                raise NotImplementedError
            else:
                raise NotImplementedError
        else:
            raise KeyError(f"{key}")
            
    def __setitem__(self, key):
        """
        {{base_docstring}}

        .. Note:: Cannot set individual entries of a 
        DecompSymmetricTensor.
        """
        raise NotImplementedError
        



# %%
if __name__ == "__main__":
    # instantiation
    A = DecompSymmetricTensor(rank = 1, dim =10) 
    assert A.rank == 1
    assert A.dim ==10
