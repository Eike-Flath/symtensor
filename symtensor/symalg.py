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
# # Additional linear algebra functions for symmetric tensors
#
# Normally these functions would be accessed from the top level:
# ```python
# import symtensor
# symtensor.contract_all_indices_with_matrix(A, B)
# ```
# If they are used frequently, the short form `st` is recommended:
# ```python
# import symtensor as st
# ```

# %%
from __future__ import annotations

# %%
from collections.abc import Iterable as Iterable_, Sequence as Sequence_
from functools import partial, reduce
from itertools import product
import numpy as np

# %%
from typing import Union, Type
from scityping import Real

# %%
import logging
logger = logging.getLogger(__name__)

# %% tags=["active-ipynb"]
# Notebook only imports

# %% tags=["active-ipynb"]
# from symtensor.base import SymmetricTensor, result_array, array_function_dispatch
# from symtensor.symtensor.dense_symtensor import DenseSymmetricTensor
# from symtensor import utils

# %% [markdown] tags=[]
# Script only imports

# %% tags=["active-py"]
from .base import SymmetricTensor, result_array, array_function_dispatch
from .dense_symtensor import DenseSymmetricTensor
from . import utils

# %%
__all__ = [
    "symmetric_outer", "symmetric_tensordot", "contract_all_indices_with_matrix",
    "contract_all_indices_with_vector", "contract_tensor_list"
    ]


# %% [markdown]
# ## Symmetrized operations
#
# Some of the functions in this module redefine operations such that they always return symmetric tensors.
# The formal definition is simply to apply the op and then symmetrize
# the result, although implementations may use a more efficient approach.
#
# Since the space of symmetric tensor tensors is closed under these operations,
# we could say that they define a *symmetric algebra*.
#
# The reference implementations provided here simply convert SymmetricTensors
# to dense arrays, then apply the function. This is obviously to be avoided
# whenever possible, so subclasses should provide specialized implementations.

# %%
def symmetrized_op(op, a, b, **kwargs):
    """
    Apply a symmetrized version of a binary `op` on the arguments `a`, `b`.
    Effectively, this simply wraps the statement

        utils.symmetrize(op(a, b))

    with argument validation and casting of the result.
    Additional arguments to `op` can be specified as keywords.
    If given, the 'out' argument will be passed to both `op` and `utils.symmetrize`.

    SymmetricTensor arguments are converted to dense arrays before applying `op`.
    Therefore providing specialized versions for SymmetricTensor subclasses
    is highly encouraged.

    .. Note:: Passing a SymmetricTensor as `out` is supported for compatibility:
       in specialized functions, it should allow to assign directly to the
       underlying memory of a SymmetricTensor. For this generic implementation
       however it provides little to no computational benefit.

    .. TODO:: For symmetric tensor arguments, it should still be possible to
       write a version which only operates on independent components, and is
       nonetheless generic.
       Especially, we should return a `SymmetricTensor` whenever possible.
    """
    # OPTIMIZE: See todo above. Just constructing DenseSymmetricTensor after
    #   operations should already save a lot of computations, and we should
    #   return a SymmetricTensor when possible.

    # Standardize `out`
    out = kwargs.pop("out", None)
    if isinstance(out, tuple):
        if len(out) > 1:
            raise TypeError("Only one 'out' argument is supported.")
        out, = out
    # Validate args
    if out is not None and not isinstance(out, (SymmetricTensor, np.ndarray)):
        raise NotImplementedError("Unsure how to perform an in-place operation from into "
                                  f"data of type {type(out)}.")

    # Replace symmetric tensors by dense arrays
    if isinstance(a, SymmetricTensor):
        a = a.todense()
    if isinstance(b, SymmetricTensor):
        b = b.todense()

    # Define `outdata`: The variable we actually use as out target
    # (`out` may be a SymmetricTensor, but `outdata` is always a NumPy array)
    if out is not None:
        if isinstance(out, DenseSymmetricTensor):
            outdata = out._data
        elif isinstance(out, SymmetricTensor):
            outdata = np.empty(out.shape)
        else:
            outdata = out
    else:
        outdata = None

    # Apply the function
    if outdata is None:
        # 'out' keyword argument was not specified
        out = utils.symmetrize(op(a, b, **kwargs))
    else:
        # 'out' keyword was specified; we presume 'op' accepts it
        op(a, b, out=outdata, **kwargs)
        utils.symmetrize(outdata, out=outdata)
        out[:] = outdata

    # Check that we actually got a symmetric matrix
    if len(set(out.shape)) > 1:
        raise RuntimeError("symmetrized op '{op}' resulted in a non-square "
                           f"result of shape {out.shape}.")

    return out

# %% [markdown]
# ## Array ufunc dispatch

# %% [markdown]
# ### `outer`

# %%
@SymmetricTensor.implements_ufunc.outer(np.add, np.subtract, np.multiply)
def symmetric_outer(ufunc: np.ufunc, a, b, **kwargs):
    ranka = np.ndim(a)
    rankb = np.ndim(b)
    dima = a.dim if isinstance(a, SymmetricTensor) else (*np.shape(a), 1)[0]  # Could be a scalar
    dimb = b.dim if isinstance(b, SymmetricTensor) else (*np.shape(b), 1)[0]  # Could be a scalar
        # TODO?: Check that arguments are square ?
    # `a` and `b` may have different rank, but they must either have matching
    # dimensions, or one must be a scalar
    if ranka != 0 and rankb != 0 and dima != dimb:
        logger.debug("Default `symmetric_outer` not implemented for args with different dimensions. "
                     f"Arg shapes: {np.shape(a)}, {np.shape(b)}")
        return NotImplemented
    dim = dima
    out = kwargs.pop('out', None)
    if out is None:
        symargs = tuple(x for x in (a, b) if isinstance(x, SymmetricTensor))
        assert symargs, "None of the arguments is a SymmetricTensor."
        cls = utils.common_superclass(*symargs)
        dtype = np.result_type(a, b)  # NB: Non-symmetric args also determine dtype
        out = cls(rank=ranka+rankb, dim=dim, dtype=dtype)
    return symmetrized_op(ufunc.outer, a, b, out=out)


# %% [markdown]
# ## Array function dispatch – existing ops

# %% [markdown]
# ### `tensordot`

# %%
@SymmetricTensor.implements(np.tensordot)
def symmetric_tensordot(a, b, axes=2):
    """
    .. Warning:: Although this defines `tensordot` for SymmetricTensors, it
       defines the *symmetrized* form. So results will in general differ
       between

           np.tensordot(A, B)

       and

           np.tensordot(A.todense(), B.todense())
    """
    cls = result_array(a, b)
    if isinstance(a, SymmetricTensor):
        a = a.todense()
    if isinstance(b, SymmetricTensor):
        b = b.todense()

    out = utils.symmetrize(np.tensordot(a, b, axes))

    if len(set(out.shape)) > 1:
        raise RuntimeError("`_symmetric_outer` resulted in a non-square "
                           f"result of shape {out.shape}.")

    if issubclass(cls, SymmetricTensor):
        dim = (*np.shape(out), 1)[0]  # NB: It is possible for both a and b to be scalars
        out = cls(rank=np.ndim(out), dim=dim, data=out)

    return out

# %% [markdown]
# ### `einsum_path`
#
# To support `einsum_path`, subclasses may define the following:
# ```python
# @MySymmetricTensorClass.implements(np.einsum_path)
# def einsum_path(*operands, optimize='greedy', einsum_call=False):
#     with utils.make_array_like(MySymmetricTensorClass(0,0), np.core.einsumfunc):
#         return np.core.einsumfunc.einsum_path.__wrapped__(
#             *operands, optimize=optimize, einsum_call=einsum_call)
# ```
# At present, this definition requires that the class be hard-coded.

# %% [markdown]
# #### Test
#
# Test that the use of `make_array_like` avoids coercion.
# (Obviously this test only makes sense if `einsum_path` is implemented)

# %% [markdown]
# ```python
# from symtensor.testing.utils import does_not_warn
#
# A = SymmetricTensor(rank=2, dim=3)
# B = SymmetricTensor(rank=2, dim=3)
#
# with does_not_warn(UserWarning):
#     np.einsum_path("ij,ik", A, B)
#     np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))
#
# with make_array_like(SymmetricTensor(0,0), np.core.einsumfunc):
#     with does_not_warn(UserWarning):
#         np.einsum_path("ij,ik", A, B)
#         np.einsum_path("ij,ik", np.ones((2,2)), np.ones((2,2)))
# ```

# %% [markdown]
# ### `einsum`
#
# **TODO** This should probably make use of the *opt_einsum* package.
#
# ```python
# @SymmetricTensor.implements(np.einsum)
# def einsum(*operands, dtype=None, order='K', casting='safe', optimize=False):
#     # NB: Can't use the implementation in np.core.einsumfunc, because that calls
#     #     C code which requires true arrays
#     ...
# ```

# %% [markdown]
# ## Array function dispatch – new ops

# %% [markdown]
# **Implementation explanation:**
#
# - The `@array_function_dispatch` decorator is used to define a new array function.
#   The function it decorates serves as the default, if the arguments do not match those of another function.
# - To each array function is paired a *dispatcher* function. This function must have the same signature as its associated function, and return a tuple of "important" arguments:[^1] those arguments the dispatcher should check to determine which function to redirect to.
# - Specialized functions can be associated to specific `SymmetricTensor` subclasses by using the `@implements` decorator.
#
# [^1]: Exception: if the function includes arguments with default values, the corresponding arguments of the dispatcher must use `None` as their default value. See also the docstring of numpy.core.overrides.array_function_dispatch; some examples can be found in numpy.core.numeric.py

# %% [markdown]
# ### Contraction functions

# %% [markdown]
# #### `contract_all_indices_with_matrix`
#
# \begin{equation*}
# C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}
# \end{equation*}

# %%
def _contract_all_indices_with_matrix_dispatcher(symtensor, W):
    return (symtensor, W)

@array_function_dispatch(_contract_all_indices_with_matrix_dispatcher)
def contract_all_indices_with_matrix(symtensor: SymmetricTensor, W: "array_like"
                                    ) -> SymmetricTensor:
    """
    Compute the contraction over all indices with a non-symmetric matrix, e.g.

    .. math::
       C_{ijk} = \sum_{abc} A_{abc} W_{ai} W_{bj} W_{ck}

    if current tensor has rank 3.
    
    Always returns a `SymmetricTensor` of the same type as `symtensor`
    """
    if not isinstance(symtensor, SymmetricTensor):
        return NotImplemented
    cls = type(symtensor)
    A = symtensor.todense()
    Aindices = utils.indices[:A.ndim]
    sum_indices = f"{Aindices}," + ",".join(
        "".join(i) for i in zip(Aindices, utils.indices[A.ndim:]))
    data = np.einsum(sum_indices, A, *(W,)*A.ndim)  # Operands must be passed as separate arguments
    return cls(rank=symtensor.rank, dim=symtensor.dim, data=data)

# %% [markdown]
# #### `contract_all_indices_with_vector`

# %%
def _contract_all_indices_with_vector_dispatcher(symtensor, x):
    return (symtensor, x)

@array_function_dispatch(_contract_all_indices_with_vector_dispatcher)
def contract_all_indices_with_vector(symtensor: SymmetricTensor, x: "array_like"
    ) -> Real:
    """
    For A a symmetric tensor and x a 1D array, compute
    \sum_{i_1, ..., i_r} A_{i_1,..., i_r} x_{1_1} ... x_{i_r}
    """
    if not isinstance(symtensor, SymmetricTensor):
        return NotImplemented
    if len(x) != symtensor.dim:
        raise ValueError("Dimensions of tensor and vector must match; received "
                         f"{symtensor.dim} (tensor) and {len(x)} (vector).")
    if np.isclose(x, 0).all():
        return 0
    else:
        # vec = DenseSymmetricTensor(rank=1, dim=self.dim)
        # vec['i'] = x
        x = np.asanyarray(x)
        tensordot = partial(np.tensordot, axes=1)
        return reduce(tensordot, (x,)*symtensor.rank, symtensor)

# %% [markdown]
# #### `contract_tensor_list`

# %% [markdown]
# **Receive**
# \begin{align*}
# n &\leftarrow \texttt{n_times} \\
# B &\leftarrow \texttt{out} \\
# A &\leftarrow \texttt{symtensor} \\
# [χ_1, χ_2, \dotsc,] &\leftarrow \texttt{tensorlist} \\
# \cdot \wedge \cdot &\leftarrow \texttt{outer}(\cdot, \cdot)
# \end{align*}
# **Do**
# \begin{multline*}
# %\begin{split}
# B_{\textstyle i_1,i_2,\dotsc , i_{r-n}, \underbrace{j_1, j_2, \dotsc, j_m, k_1, k_2, \dotsc, k_m, \dotsc}_{n \,\times\, m}} \\
#     = \mathop{\mathrm{Symmetrize}}\biggl( \sum_{\textstyle i_{r-n+1}, \dotsc , i_r} A_{\textstyle i_1,i_2,\dotsc, i_r} \wedge \underbrace{χ_{\textstyle i_{r-n+1}; j_1,j_2,\dotsc, j_m} \wedge χ_{\textstyle i_{r-n+2}; k_1,k_2,\dotsc, k_m} \wedge \dotsb}_{n \text{ terms}} \biggr),
# %\end{split}
# \end{multline*}
#
# If $A$ has rank $r$ and the $χ_i$ have rank $m$, then the resulting tensor $B$ has rank $r + n(m-1)$.

# %%
def _contract_tensor_list_dispatch(symtensor, tensor_list, n_times=None, rule=None):
    return (symtensor, *tensor_list)

@array_function_dispatch(_contract_tensor_list_dispatch)
def contract_tensor_list(
      symtensor: SymmetricTensor,
      tensor_list: Sequence[SymmetricTensor],
      n_times: int=1,
      rule: str="second_half"):
    """
    Do the following contraction:

    out_{i_1,i_2,..., i_(r-n_times), j_1, j_2, ...j_m, k_1, k_2, ... k_m, ...}
    = Symmetrize[ \sum_{i_{r-n_times+1}, ..., i_r} outer( symtensor_{i_1,i_2,...,i_r}, tensor_list[i_{r-n_times+1}]_{j_1,j_2,...,j_m} ) ]

    Important: The tensors in tensor_list must be symmetric.
    This is essentially a way to do a contraction between a symmetric and quasi_symmetric tensor χ. Let

    χ_{i,j_1,j_2,...,j_m} = tensor_list[i]_{j_1,j_2,...j_m}

    Then even if χ is not symmetric under exchange of the first indices with the rest, but the subtensors χ_i,...
    for fixed i are, we can do a contraction along the first index.

    TODO: Document 'rule' argument
    """
    if isinstance(tensor_list, Iterable_) and not isinstance(tensor_list, Sequence_):
        # Ensure we don’t consume an iterator
        tensor_list = list(tensor_list)

    if (not isinstance(symtensor, SymmetricTensor)
        or not all(isinstance(χ, SymmetricTensor) for χ in tensor_list)):
        return NotImplemented

    cls = result_array(symtensor, *tensor_list)

    # Assert n ⩽ r
    Ar = symtensor
    if n_times > Ar.rank:
        raise ValueError(f"n_times is {n_times}, but cannot do more contractions "
                         f"than {Ar.rank} with tensor of rank {Ar.rank}")
    # Assert first dimension of `tensorlist` matches dimension of A
    if len(tensor_list) != Ar.dim:
        raise ValueError("`tensor_list` emulates the first dimension of a tensor, and "
                         "therefore its length must match the dimenion of `symtensor`.\n"
                         f"Symtensor dim     : {Ar.dim}\nLength tensor list: {len(tensor_list)}")
    # Assert χi are symmetric
    if not all(isinstance(χ, SymmetricTensor) for χ in tensor_list):
        raise  TypeError("tensor_list entries must be SymmetricTensors")
    # Assert χi all have the same shape
    χ_ranks = {a.rank for a in tensor_list}  # m’s in the docs above
    χ_dims = {a.dim for a in tensor_list}
    if len(χ_ranks) > 1 or len(χ_dims) > 1:
        raise ValueError("Tensors in `tensor_list` do not all have the same "
                         f"shape:\n{[np.shape(χ) for χ in tensor_list]}")
    # Set χ_rank and χ_dim
    χ_rank = next(iter(χ_ranks))
    χ_dim = next(iter(χ_dims))
    # Assert χi have the same dimension as A
    if χ_dim != Ar.dim:
        raise ValueError("Tensors in `tensor_list` do not have the same "
                         f"dimension as `symtensor`.")
    

    if Ar.rank == 1 and n_times == 1:
        return sum((tensor_list[i]*Ar[i] for i in range(Ar.dim)),
                    start=cls(tensor_list[0].rank, tensor_list[0].dim))

    else:
        if rule == 'second_half':
            first_half = math.ceil(Ar.dim/2)
            indices_for_contraction = range(first_half, Ar.dim)
            indices = product(indices_for_contraction, repeat=n_times)
        else:
            indices = product(range(Ar.dim), repeat=n_times)

        C = cls(dim = Ar.dim, rank = Ar.rank + n_times*(χ_rank-1))
            # One dimension used for contraction
            
        for idx in indices:
            # outer(... (outer(outer(Ar[idx], χ[i0]), χ[i1]) ..., χ[in])
            C += reduce(np.multiply.outer, [tensor_list[i] for i in idx], Ar[idx])

        return C
