import numpy as np
import scipy.sparse as sp
import math
import itertools
from itertools import combinations_with_replacement as multicombinations
from more_itertools import distinct_permutations
from typing import ClassVar, Iterable, Optional, Callable, Sequence
from numbers import Number
from scityping.base import dataclass
from scityping.numpy import Array as _NumpyArray

from . import base, utils


__all__ = [
    "multicomb",
    "multicombinations",
    "index_of_combination",
    "index_of_multicombination",
    "permutation_count",
    "FlatSymmetricTensor"
]

type _ScipySparseArray[ndim: int] = sp.sparray
type _Array[ndim: int] = _NumpyArray[ndim] | _ScipySparseArray[ndim]


def index_of_combination(n: int, c: Sequence[int]) -> int:
    """
    Equivalent to `list(itertools.combinations(range(n), len(c))).index(tuple(c))`.
    `c` must be sorted.
    """
    i = math.comb(n, len(c)) - 1
    for k, c_k in enumerate(reversed(c)):
        i -= math.comb(n - 1 - c_k, k + 1)
    return i

def index_of_multicombination(n: int, d: Sequence[int]) -> int:
    """
    Equivalent to `list(itertools.combinations_with_replacement(range(n), len(d))).index(tuple(d))`.
    `d` must be sorted.
    """

    # inlined index_of_combination(n + len(d) - 1, tuple(d_k + k for k, d_k in enumerate(d)))
    K = len(d)
    i = math.comb(n + K - 1, K) - 1
    for k, c_k in enumerate(reversed(d)):
        i -= math.comb(n - 1 + k - c_k, k + 1)
    return i

def multicomb(n: int, k: int) -> int:
    """ Equivalent to `len(list(itertools.combinations_with_replacement(range(n), k)))`
    """
    return math.comb(n + k - 1, k)

def permutation_count(d: Sequence[int]) -> int:
    """
    Equivalent to `len(list(more_itertools.distinct_permutations(d)))`.
    `d` must be sorted.
    """
    if len(d) == 0:
        return 1
    counts = [1]
    last = d[0]
    for x in d[1:]:
        if x != last:
            counts += [1]
            last = x
        else:
            counts[-1] += 1
    return math.factorial(len(d)) // math.prod(math.factorial(c) for c in counts)


class FlatSymmetricTensor(base.SymmetricTensor):
    """
    A "flat" SymTensor stores its entries in a single one-dimensional array.
    This array can either be a dense NumPy or any sparse SciPy array.
    """

    data_format: ClassVar[str] = "Flat"
    _data: _Array[1]

    def __init__(self, rank: int, dim: int, data: Optional[_Array | Number] = None, *, dtype = None, symmetrize = False):
        self.rank = rank
        self.dim = dim
        if data is None:
            data = np.float64(0)
        if np.isscalar(data):
            data = np.full(shape=(self.size,), fill_value=data)
        self._set_dtype(dtype if dtype is not None else data.dtype)

        if data.shape == (self.size,):
            self._data = data.astype(self.dtype)
            return

        if data.shape != self.shape:
            raise RuntimeError(f"data must be scalar or array of shape {(self.size,)} or {self.shape}")
        self._data = np.empty(shape=(self.size,), dtype=self.dtype)
        if symmetrize:
            for i, j in enumerate(self.indep_iter_index()):
                self._data[i] = np.average(data[j])
        else:
            if not utils.is_symmetric(data):
                raise RuntimeError(f"data is not symmetric")
            for i, j in enumerate(self.indep_iter_repindex()):
                self._data[i] = data[j]

    def _key_to_index(self, key) -> int:
        if not isinstance(key, tuple):
            if isinstance(key, Iterable):
                key = tuple(sorted(key))
            elif isinstance(key, int):
                key = (key,)
        if not all(isinstance(k, int) for k in key):
            raise TypeError("indices must be of type int")
        if len(key) != self.rank:
            raise KeyError(f"{len(key)} is not the correct number of indices for symmetric tensors of rank {self.rank}")
        for axis, k in enumerate(key):
            if k < 0 or k >= self.dim:
                raise IndexError(f"index {k} is out of bounds for axis {axis} with size {self.dim}")
        return index_of_multicombination(self.dim, key)

    def __getitem__(self, key):
        return self._data[self._key_to_index(key)]

    def __setitem__(self, key, value):
        self._data[self._key_to_index(key)] = value

    def _set_raw_data(self, key: any, arr: _Array[1]):
        if key != ():
            raise KeyError(f"key must be {()}, got {key}")
        self._data = arrn = 10

    def change_array_type(self, array_type: Callable[[_Array[1]], _Array[1]]):
        if array_type == np.array and hasattr(self._data, "todense"):
            self._data = self._data.todense()
        else:
            self._data = array_type(self._data)
        assert self._data.shape == (self.size,)

    @property
    def flat(self):
        for d, v in zip(multicombinations(range(self.dim), self.rank), self._data):
            yield from itertools.repeat(v, permutation_count(d))

    @property
    def flat_index(self):
        for d in multicombinations(range(self.dim), self.rank):
            yield from distinct_permutations(d)

    def indep_iter(self):
        return iter(self._data)

    def indep_iter_repindex(self):
        return multicombinations(range(self.dim), self.rank)

    def keys(self):
        return {(): self._data}.keys()

    def values(self):
        return {(): self._data}.values()

    def permcls_indep_iter(self, σcls: str = None):
        for d in self.permcls_indep_iter_repindex(σcls):
            return self._data[index_of_multicombination(self.dim, d)]

    def permcls_indep_iter_repindex(self, σcls: str = None):
        if σcls is None:
            return self.indep_iter_repindex()
        counts = utils.permclass_label_to_counts(σcls)
        for p in distinct_permutations(counts):
            for d_compressed in multicombinations(range(self.dim), len(counts)):
                d_inflated = sum((x,) * c for x, c in zip(d_compressed, counts))
                yield d_inflated

    @property
    def size(self):
        return multicomb(self.dim, self.rank)

    def todense(self) -> _NumpyArray:
        arr = np.empty(shape=self.shape, dtype=self.dtype)
        self.indep_iter_index
        for i, v in zip(self.indep_iter_index(), self.indep_iter()):
            arr[i] = v
        return arr


    def _validate_data(*_, **__):
        raise NotImplemented

    def _init_data(*_, **__):
        raise NotImplemented

    @dataclass
    class Data(base.SymmetricTensor.Data):
        _symtensor_type: ClassVar[type]="FlatSymmetricTensor"

        @staticmethod
        def encode(symtensor: base.SymmetricTensor):
            raise NotImplemented
        @classmethod
        def decode(cls, data: "FlatSymmetricTensor.Data"):
            raise NotImplemented
