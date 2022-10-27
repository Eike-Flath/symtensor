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
# # Symmetric PyTorch tensors

# %% tags=["remove-input"]
from __future__ import annotations

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
# #from symtensor import SymmetricTensor
# from symtensor.symtensor.base import SymmetricTensor,_elementwise_compare, _array_compare
# from symtensor.symtensor import base
# from symtensor.symtensor import utils

# %% tags=["active-py", "remove-cell"]
Script only imports
from .base import SymmetricTensor, array_function_dispatch
from . import base
from . import utils

# %% [markdown]
# ## Considerations

# %% [markdown]
# ### Array functions
#
# Torch tensors support the [*universal function*](./base.py#implementation-of-the-array-ufunc-dispatch-protocol) dispatch protocol, but not the [*array function*](./base.py#implementation-of-the-array-function-dispatch-protocol) dispatch protocol. Therefore, while this returns a Torch tensor,

# %%
A = torch.tensor([3, 6, 9])
np.exp(A)

# %% [markdown]
# the following does not

# %%
np.tensordot(A, A, axes=1)

# %% [markdown]
# Instead we need to use the torch version,
# :::{margin}
# Why PyTorch uses `dims` here instead of `axes` one can only guess…
# :::

# %%
torch.tensordot(A, A, dims=1)


# %% [markdown]
# :::{admonition} Takeaway
#
# Functions which use only ufuncs should therefore not require a specialized implementation for PyTorch, and can simply be inherited from the base class.
#
# All the functions defined using `SymmetricTensor.implements`, however, probably need a PyTorch version.  
# :::
#
# Note that many array functions – like `np.ndim`, `np.isclose`, … – still work, because the default implementation happens to work with Torch tensors. (For example, `np.ndim` checks for an `ndim` attribute.) This is probably a combination of accident and design. In no cases however will these array functions dispatch to an equivalent torch functions such as `torch.isclose`.
#
# We also note that PyTorch defines an analogous mechanism dispatch mechanism for classes which define a `__torch_function__` method. We could use this mechanism to support calling torch functions (like `torch.tensordot`) with `SymmetricTensor` arguments. Whether we need this is still to be determined.

# %% [markdown]
# NumPy dtype <-> PyTorch dtype conversions. Based on a list from PyTorch's [test utilities](https://github.com/pytorch/pytorch/blob/e180ca652f8a38c479a3eff1080efe69cbc11621/torch/testing/_internal/common_utils.py#L349). except that we use the actual NumPy dtypes.

# %%

_numpy_to_torch_dtypes = bijection({
    np.dtype(np.bool)       : torch.bool,
    np.dtype(np.uint8)      : torch.uint8,
    np.dtype(np.int8)       : torch.int8,
    np.dtype(np.int16)      : torch.int16,
    np.dtype(np.int32)      : torch.int32,
    np.dtype(np.int64)      : torch.int64,
    np.dtype(np.float16)    : torch.float16,
    np.dtype(np.float32)    : torch.float32,
    np.dtype(np.float64)    : torch.float64,
    np.dtype(np.complex64)  : torch.complex64,
    np.dtype(np.complex128) : torch.complex128
})

# %% [markdown] tags=[]
# ## Abstract `TorchSymmetricTensor`
#
# Compared to `SymmetricTensor`:
# + *Adds* the following attributes :
#     - `device`: Set to either `"cpu"` or `"gpu"`.
#
# + *Removes* the following methods:
#     - `astype`: Torch tensors don't implement it
#
# + *Modifies* the following private methods:
#     - `_validate_dataarray`: Validates to Torch types. This is anyway normally overridden by subclasses.
#     - `_set_dtype`: Uses Torch dtypes

# %%
class TorchSymmetricTensor(SymmetricTensor):
    """
    Abstract `SymmetricTensor` using Torch tensors instead of NumPy arrays
    for the underlying storage.
    """
    
    # Overridden class attributes
    array_type  : ClassVar[type]=torch.tensor
    # New attributes
    _device_name: str
    
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
        # Set device
        self._device_name = str(device).lower()
        if not self._device_name in {"cpu", "gpu"}:
            raise ValueError("`device` should be either 'cpu' or 'gpu'; "
                             f"received '{self._device_name}'.")
                             
        # Convert possibly NumPy dtype to PyTorch
                             
        # Let super class do the initialization
        super().__init__(rank = rank, dim = dim , data = data, dtype =dtype)

    def _validate_dataarray(self, array: "array-like") -> Array:
        # Cast to array if necessary
        if not isinstance(array, torch.Tensor):
            # Reproduce the same range of standardizations NumPy has: Python bools & ints, NumPy types, tuples, lists, etc.
            array = torch.tensor(array)

        # Validate dtype
        # At present, PyTorch only has numeric & bool dtypes, so there isn't really anything to check
        assert isinstance(array.dtype, torch.dtype), "Not a Torch dtype."
        
        return array

    def _set_dtype(self, dtype: Optional[DType]):
        if dtype is None:
            dtype = torch.float64
        elif not isinstance(dtype, torch.dtype):
            dtype = _numpy_to_torch_dtypes[np.dtype(dtype)]
        self._dtype = dtype

    @property
    def device(self) -> "torch.device":
        return torch.device(self._device_name)

    #### Pydantic serialization ####
    
    # TODO: Can we get rid of this ? The only difference with the base class is the type of `data`; maybe that can be inferred from a `cls.array_type` ?
    class Data(BaseModel):
        rank: int
        dim: int
        # NB: JSON keys must be str, int, float, bool or None, but not tuple => convert to str
        data: Dict[str, TorchTensor]
        @classmethod
        def json_encoder(cls, symtensor: SymmetricTensor):
            return cls(rank=symtensor.rank, dim=symtensor.dim,
                       data={str(k): v for k,v in symtensor.items()})
        
        #Todo: Write encode funtion
        def encode(self,): 
            #dirty hack
            pass

    ## Array creation, copy, etc. ##

    def astype(self, dtype, order, casting, subok=True, copy=True):
        "Torch tensors don't implement this."
        raise NotImplementedError

# %% [markdown]
# ### Implementation of the `__array_function__` dispatch protocol

# %% [markdown]
# #### `result_type()`

# %%
@TorchSymmetricTensor.implements(np.result_type)
def result_type(*tensors_or_numbers) -> DType:
    """
    Extends support for numpy.result_type. SymmetricTensors are treated as
    arrays of the same dtype.
    
    .. Caution:: In contrast to NumPy's `result_type`, this does not accept
       dtypes as arguments.
    """
    # If the array function dispatch got here, one of the args is a TorchSymmetricTensor
    # torch.result_type only works with pairs, so we use functools.reduce to apply to all args
    return reduce(torch.result_type, tensors_or_numbers)

# %% [markdown]
# #### `isclose()`

# %%

@TorchSymmetricTensor.implements(np.isclose)
def isclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> Union[np.ndarray, SymmetricTensor]:
    """
    If `a` and `b` are compatible SymmetricTensors, returns a
    `SymmetricTensor` of the same shape with boolean entries.
    Otherwise returns an `ndarray`.
    """
    return base._elementwise_compare(
        partial(torch.isclose, rtol=rtol, atol=atol, equal_nan=equal_nan),
        a, b)

# %% [markdown]
# #### `array_equal()`, `allclose()`
#
# **[TODO]** For consistency with NumPy, `allclose` should apply broadcasting, and raise `ValueError` if the shapes aren’t broadcastable.

# %%

@TorchSymmetricTensor.implements(np.array_equal)
def array_equal(a, b) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are equal. Emulates `numpy.array_equal` for torch tensors.
    """
    # NB: torch.array_equal is not defined, but np.array_equal seems to work.
    return base._array_compare(np.array_equal, a , b)

@TorchSymmetricTensor.implements(np.allclose)
def allclose(a, b, rtol=1e-5, atol=1e-8, equal_nan=False) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are close. C.f. `torch.allclose`.
    """
    return base._array_compare(
        partial(torch.allclose, rtol=rtol, atol=atol, equal_nan=equal_nan),
         a, b)

# %% [markdown]
# #### Definition of new array functions

# %% [markdown]
# #### Type promotion: `result_array`

# %%
@TorchSymmetricTensor.implements(base.result_array)
def result_array(*arrays_and_types) -> None:
    """
    Analogue to `result_type`: apply type promotion on array classes themselves.
    Arguments may be types or instances.

    When multiple arguments are TorchSymmetricTensors, the returned type is the
    most specific common superclass.

    result_array(ndarray, ndarray) -> ndarray
    result_array(ndarray, SymmetricTensor) -> SymmetricTensor
    result_array(SymmetricTensor, TorchSymmetricTensor) -> TorchSymmetricTensor
    result_array(TorchSymmetricTensor, TorchSymmetricTensor) -> TorchSymmetricTensor
    """
    types = (arr if isinstance(arr, type) else type(arr)
             for arr in arrays_and_types)
    symtypes = tuple(T for T in types if issubclass(T, TorchSymmetricTensor))
    return utils.common_superclass(*symtypes)


# %% [markdown]
# (End of generic definitions)
# --------------------------------

# %% [markdown]
# ## `DenseTorchSymmetricTensor`

# %%
from symtensor.symtensor.dense_symtensor import DenseSymmetricTensor

class DenseTorchSymmetricTensor(DenseSymmetricTensor, TorchSymmetricTensor):
    _data                : Union[TorchTensor]

    def _init_data(self, data, symmetrize: bool):
        # NB: Can assume that `data` is a Torch tensor and self._dtype a Torch dtype
        if data.dtype != self._dtype:  # Only cast if necessary
            data = torch.tensor(data.numpy(), dtype=self._dtype)
        if data.shape == (self.dim,)*self.rank:
            self._data = data
        else:
            self._data = torch.empty((self.dim,)*self.rank, dtype=self._dtype)
            self._data[:] = data
        if symmetrize:
            self._data = utils.symmetrize(self._data)
        elif not utils.is_symmetric(self._data):
            raise ValueError("Data are not symmetric.")
