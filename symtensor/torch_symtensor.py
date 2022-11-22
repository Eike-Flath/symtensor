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
from functools import reduce, partial
import math
import numpy as np   # To avoid MKL bugs, always import NumPy before Torch
import torch
from pydantic import BaseModel
from collections import namedtuple
from collections_extended import bijection
import dataclasses

from typing import Optional, ClassVar, Union, Callable
from scityping import Number
from scityping.numpy import Array, DType
from scityping.torch import TorchTensor

# %% tags=["active-ipynb"]
# Module only imports

# %% tags=["active-ipynb"]
# #from symtensor import SymmetricTensor
# from symtensor.base import SymmetricTensor,_elementwise_compare, _array_compare
# from symtensor import base, symalg, utils

# %% [markdown] tags=["remove-cell"]
# Script only imports

# %% tags=["active-py", "remove-cell"]
from .base import SymmetricTensor, array_function_dispatch
from . import base, symalg, utils

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
    np.dtype(bool)       : torch.bool,
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
# + *Adds* the following attributes and methods :
#     - `device`: Set to either `"cpu"` or `"gpu"`.
#     - `clone`: Call `.clone()` on all underlying tensors
#     - `detach`: Call `.detach()` on all underlying tensors
#
# + *Removes* the following methods:
#     - `astype`: Torch tensors don't implement it
#
# + *Modifies* the following private methods:
#     - `_validate_dataarray`: Validates to Torch types. This is anyway normally overridden by subclasses.
#     - `_set_dtype`: Uses Torch dtypes
#     - `copy`: Uses `.detach().clone()` instead of `.copy()`

# %%
@dataclasses.dataclass
class TorchUfunc:
    npufunc  : Callable # Real ufunc: the NumPy ufunc we want to replace
    __call__ : Callable   # Faked ufunc: the equivalent Torch function
    signature: Optional[str]=None
    nin      : int=None
    nout     : int=None
    def __post_init__(self):
        self.__call__ = getattr(torch, self.npufunc.__name__)
        self.signature = self.npufunc.signature
        self.nin = self.npufunc.nin
        self.nout = self.npufunc.nout

# %%
class TorchSymmetricTensor(SymmetricTensor):
    """
    Abstract `SymmetricTensor` using Torch tensors instead of NumPy arrays
    for the underlying storage.
    """

    # Overridden class attributes
    array_type  : ClassVar[type]=torch.tensor
    data_format : ClassVar[str]="None: This class is abstract"  # Special value converted to 'None'. Used to skip validation check.
    # New attributes
    _device_name: str

    def __init__(self, rank: Optional[int]=None, dim: Optional[int]=None,
                 data: Union[Array, Number]=np.float64(0),
                 dtype: Union[str,DType]=None,
                 symmetrize: bool=False,
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
        super().__init__(rank=rank, dim=dim , data=data, dtype=dtype,
                         symmetrize=symmetrize)

    def _validate_dataarray(self, array: "array-like") -> Array:
        # Cast to array if necessary
        if not isinstance(array, torch.Tensor):
            # Reproduce the same range of standardizations NumPy has: Python bools & ints, NumPy types, tuples, lists, etc.
            # NB: During initialization, `self._dtype` is None.
            array = torch.tensor(array, dtype=self._dtype)
        elif self._dtype is not None and array.dtype != self._dtype:
            array = array.type(self._dtype)

        # Validate dtype
        # At present, PyTorch only has numeric & bool dtypes, so there isn't really anything to check
        assert isinstance(array.dtype, torch.dtype), "Not a Torch dtype."

        return array

    def _set_dtype(self, dtype: Optional[DType]):
        if dtype is None:
            dtype = torch.float64
        # TODO: if isinstance(dtype, torch.Tensor)
        elif not isinstance(dtype, torch.dtype):
            dtype = _numpy_to_torch_dtypes[np.dtype(dtype)]
        self._dtype = dtype

    def copy(self) -> TorchSymmetricTensor:
        """
        .. Caution:: This is equivalent to `.clone()`, meaning that the data are
           copied but still port of the computational graph. to get a  “pure data”
           object, as might be expected for a copy, use `.detach().clone()`
        """
        return self.clone()

    def clone(self) -> TorchSymmetricTensor:
        """
        Call `.clone()` on all underlying Torch tensors and return a new
        TorchSymmetricTensor with the result.
        """
        return self.__class__(dim = self.dim, rank = self.rank,
                              data = {k: arr.clone() for k, arr in self.items()})

    def detach(self) -> TorchSymmetricTensor:
        """
        Call `.detach()` on all underlying Torch tensors and return a new
        TorchSymmetricTensor with the result.
        """
        return self.__class__(dim = self.dim, rank = self.rank,
                              data = {k: arr.detach() for k, arr in self.items()})

    @property
    def device(self) -> "torch.device":
        return torch.device(self._device_name)

    #### Pydantic serialization ####

    # TODO: Can we get rid of this ? The only difference with the base class is the type of `data`; maybe that can be inferred from a `cls.array_type` ?
    # class Data(SymmetricTensor.Data):
    #     pass
        # _symtensor_type: ClassVar[Optional[type]]="DenseTorchSymmetricTensor"  # NB: DenseTorchSymmetricTensor is not yet defined
        # rank: int
        # dim: int
        # # NB: JSON keys must be str, int, float, bool or None, but not tuple => convert to str
        # data: Dict[str, TorchTensor]
        # @classmethod
        # def json_encoder(cls, symtensor: SymmetricTensor):
        #     return cls(rank=symtensor.rank, dim=symtensor.dim,
        #                data={str(k): v for k,v in symtensor.items()})

        # #Todo: Write encode funtion
        # def encode(self,):
        #     #dirty hack
        #     pass

    ## Array creation, copy, etc. ##

    def astype(self, dtype, order, casting, subok=True, copy=True):
        "Torch tensors don't implement this."
        raise NotImplementedError

    ## Override default ufuncs to use torch ops ##

    # NB: Making a @staticmethod would prevent using `super()`
    def default_unary_ufunc(self, ufunc, method, *inputs, **kwargs):
        # TODO: Use type(self).with_backend("numpy").default_unary_ufunc(self, ufunc, ...)
        return super().default_unary_ufunc(TorchUfunc(ufunc), method,
                                           *inputs, **kwargs)

    def default_binary_ufunc(self, ufunc, method, *inputs, **kwargs):
        # TODO: Use type(self).with_backend("numpy").default_unary_ufunc(self, ufunc, ...)
        return super().default_binary_ufunc(TorchUfunc(ufunc), method,
                                            *inputs, **kwargs)

# %% [markdown]
# ### Implementations for the `__array_ufunc__` dispatch protocol

# %% [markdown]
# Replace NumPy ufuncs by Torch equivalents in implementations.
# (E.g. `np.multiply.outer` becomes `torch.outer`)
# Note that `symalg` ops expect a `UfuncWrapper`, which for the purposes of the specialized implementation, is just a container with an attribute `ufunc` pointing to the ufunc to specialize.
#
# So internally, `symalg.outer` only uses its `ufunc` argument to retrieve `ufunc.ufunc.outer`. In order to reuse the function, we create a proxy object where the torch-compatible `outer` can be retrieved at the same location.
#
# Note that Torch doesn’t provide `outer` variants of its functions, so we emulate it with broadcasting.

# %%
class UfuncWithOuter:  # TODO: Combine with TorchUfunc defined above
    def __init__(self, torchop):
        self.torchop = torchop
    def outer(self, a, b, **kwargs):
        slc = (slice(None),)*np.ndim(a) + (np.newaxis,)*np.ndim(b)
        return self.torchop(torch.as_tensor(a)[slc], torch.as_tensor(b), **kwargs)

ObjWithUfunc = namedtuple("ObjWithUfunc", ["ufunc"])

# %%
for symalgop, torchop in [(symalg.add, torch.add),
                          (symalg.subtract, torch.subtract),
                          (symalg.multiply, torch.multiply)]:

    @TorchSymmetricTensor.implements_ufunc.outer(symalgop)
    def outer(ufunc: UfuncWrapper, a, b, **kwargs):
        return symalg.outer(ObjWithUfunc(UfuncWithOuter(torchop)),
                            a, b, **kwargs)


# %% [markdown]
# ### Implementations for the `__array_function__` dispatch protocol

# %% [markdown]
# #### `result_type()`

# %%
@TorchSymmetricTensor.implements(np.result_type)
def result_type(*arrays_and_dtypes) -> torch.dtype:
    """
    Extends support for numpy.result_type. SymmetricTensors are treated as
    arrays of the same dtype.

    .. Caution:: In contrast to NumPy's `result_type`, this does not accept
       dtypes as arguments.
    """
    # If the array function dispatch got here, one of the args is a TorchSymmetricTensor
    # torch.result_type only works with pairs, so we use functools.reduce to apply to all args
    # torch.result_type ALSO only works with values (not dtypes), so for dtypes we create throwaway vars
    #   We also do this inside a function, since the return value of result_type is itself a dtype
    def result_type(a, b):
        x = a
        a = (torch.scalar_tensor(1, dtype=x) if isinstance(x, torch.dtype)
             else x.type(1) if isinstance(x, np.dtype)
             else x)
        x = b
        b = (torch.scalar_tensor(1, dtype=x) if isinstance(x, torch.dtype)
             else x.type(1) if isinstance(x, np.dtype)
             else x)
        return torch.result_type(a, b)
    res = reduce(result_type,
                 (x for x in (y.dtype if isinstance(y, SymmetricTensor) else y
                              for y in arrays_and_dtypes)))
    # If `arrays_and_dtypes` has only one element, `reduce` simply returns it.
    # I.e. it will likely be a torch Tensor rather than dtype
    if isinstance(res, torch.Tensor):
        res = res.dtype
    return res

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
        lambda _a, _b: torch.isclose(torch.as_tensor(_a), torch.as_tensor(_b),
                                     rtol=rtol, atol=atol, equal_nan=equal_nan),
        a, b
        )

# %% [markdown]
# #### `array_equal()`, `allclose()`
#
# **[TODO]** For consistency with NumPy, `allclose` should apply broadcasting, and raise `ValueError` if the shapes aren’t broadcastable.

# %%

# NB: torch.array_equal is not defined, but the default implementation with np.array_equal seems to work.
# @TorchSymmetricTensor.implements(np.array_equal)
# def array_equal(a, b) -> bool:
#     """
#     Return True if `a` and `b` are both `SymmetricTensors` and all their
#     elements are equal. Emulates `numpy.array_equal` for torch tensors.
#     """
#     return base._array_compare(np.array_equal, a , b)

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
# #### `utils.astype`, `utils.empty`
#
# Adaptations of array creation and coercion functions to support Torch types.

# %%
@utils.astype.register
def _(a: torch.Tensor, dtype):
    return a.type(dtype)

# %%
@utils.empty_array_like.register(torch.Tensor)
@utils.empty_array_like.register(TorchSymmetricTensor)
def _(like, shape: Tuple[int,...], dtype=None):
    return torch.empty(shape, dtype=dtype)

# %% [markdown]
# #### `utils.symmetrize`
#
# This is a port of the default implementation in `symtensor.utils`, with the
# only difference that a Torch array is created (instead of a NumPy array).
#
# **TODO?**: We put this in this module because it requires the `torch` import,
#   but it would be better placed in something like a `torch.utils` module.

# %%
import itertools

@utils.symmetrize.register
def _(tensor: torch.Tensor, out: Optional[TorchTensor]=None) -> TorchTensor:
    # OPTIMIZATION:
    # - If possible (i.e. if tensor ≠ out), use `out` to avoid intermediate copies in sum
    # - In this regard, perhaps a version using `np.add.accumulate` might be faster ?
    # - Does not seem to use threading: only one CPU is active, even with large matrices

    # Inspect args for correctness and whether symmetrization can be skipped.
    if len(set(tensor.shape)) > 1:
        raise ValueError(f"Cannot symmetrize tensor of shape {denser_tensor.shape}: "
                         "Dimensions do not all have the same length.")
    D = tensor.ndim
    if D <= 1:
        return tensor # Nothing to symmetrize: EARLY EXIT

    # Perform symmetrization
    n = math.prod(range(1,D+1))  # Factorial – number of permutations
    if out is None:
        out = torch.empty_like(tensor)
    out[:] = sum(tensor.permute(*σaxes) for σaxes in itertools.permutations(range(D))) / n
    return out

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
#
# --------------------------------

# %% [markdown]
# ## `DenseTorchSymmetricTensor`

# %%
from symtensor.dense_symtensor import DenseSymmetricTensor

class DenseTorchSymmetricTensor(TorchSymmetricTensor, DenseSymmetricTensor):
    _data                : Union[TorchTensor]

    # NB: Torch doesn’t define `flat`, but instead `flatten` returns a view when possible
    @property
    def flat(self):
        return self._data.flatten()

    class Data(DenseSymmetricTensor.Data):
        _symtensor_type: ClassVar[Optional[type]]="DenseTorchSymmetricTensor"  # NB: Data.decode expects a string, in order resolve the forward ref

# %% [markdown]
# ## `DenseTorchSymmetricTensor`

# %%
from symtensor import permcls_symtensor as σst

class PermClsTorchSymmetricTensor(TorchSymmetricTensor, σst.PermClsSymmetricTensor):
    _data       : Dict[Tuple[int,...], Union[TorchTensor]]

    class Data(σst.PermClsSymmetricTensor.Data):
        _symtensor_type: ClassVar[Optional[type]]="PermClsTorchSymmetricTensor"  # NB: Data.decode expects a string, in order resolve the forward ref

    def _validate_data(self, data, symmetrize: bool=False):
        # Override the case where we initialize with a dense array / tensor or a dict
        # For scalar case, the implementation in PermClsSymmetricTensor still works

        if isinstance(data, np.ndarray):
            data = torch.tensor(data)

        if isinstance(data, torch.Tensor):
            rank = np.ndim(data) if self.rank is None else self.rank
            dim = (self.dim if self.dim is not None
                   else max(*data.shape, 1) if data.shape  # NB: For 1D arrays, torch.shape returns a non-iterable scalar
                   else 1)  # Last line for scalars, which have an empty shape tuple
            datashape = (dim,)*rank
            datadtype = data.dtype
            try:
                broadcasted_data = torch.broadcast_to(data, datashape)
            except RuntimeError as e:
                # Translate into a ValueError for consistency with NumPy.
                # Because of course PyTorch chose to raise a different exception…
                raise ValueError(str(e)) from e
            if rank == 0:
                data = {(): data}
            else:
                if symmetrize:
                    broadcasted_data = utils.symmetrize(broadcasted_data)
                elif not utils.is_symmetric(broadcasted_data):
                    raise ValueError("Data array is not symmetric.")
                data = {σcls: self._validate_dataarray(
                            broadcasted_data[tuple(np.array(idcs)
                                for idcs in zip(*σst.σindex_iter(σcls, dim)))])
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
            datadtype = result_type(*data.values())  # ONLY CHANGE WRT PARENT CLASS: If we could define `result_type` for torch tensor (not just TorchSymmetricTensors), the parent class def would work

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
            # Currently only the scalar case remains in this branch
            data, datadtype, datashape = super()._validate_data(data, symmetrize)
        if isinstance(datadtype, torch.Tensor):
            import pdb; pdb.set_trace()
        return data, datadtype, datashape

    def todense(self) -> TorchTensor:
        A = torch.empty(self.shape, dtype=self.dtype)
        for idx, value in zip(self.indep_iter_index(), self.indep_iter()):
            A[idx] = value
        return A

# %% [markdown]
# ### Implementations for the `__array_function__` dispatch protocol

# %% [markdown]
# #### `array_equal()`
#
# Overriden to allow for scalars in the underlying arrays: underlying arrays not having the same shape is fine

# %%
@PermClsTorchSymmetricTensor.implements(np.array_equal)
def array_equal(a, b) -> bool:
    """
    Return True if `a` and `b` are both `SymmetricTensors` and all their
    elements are equal. Emulates `numpy.array_equal` for torch tensors.
    """
    return np.shape(a) == np.shape(b) and base._array_compare(
        lambda x, y: torch.all(x == y), a , b)
