import numpy as np
import scipy.sparse as sp
import math
import itertools
from itertools import combinations_with_replacement as multicombinations
from more_itertools import distinct_permutations
from typing import ClassVar, Iterable, Optional, Callable, Sequence
from numbers import Number, Integral
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

type _ScipySparseArray[_: type] = sp.sparray
type _Array[T: type] = _NumpyArray[T] | _ScipySparseArray[T]


def index_of_combination(n: int, c: Sequence[Integral]) -> int:
    """
    Equivalent to `list(itertools.combinations(range(n), len(c))).index(tuple(c))`.
    `c` must be sorted.
    """
    i = math.comb(n, len(c)) - 1
    for k, c_k in enumerate(reversed(c)):
        i -= math.comb(n - 1 - c_k, k + 1)
    return int(i)


def index_of_multicombination(n: int, d: Sequence[Integral]) -> int:
    """
    Equivalent to `list(itertools.combinations_with_replacement(range(n), len(d))).index(tuple(d))`.
    `d` must be sorted.
    """

    # inlined index_of_combination(n + len(d) - 1, tuple(d_k + k for k, d_k in enumerate(d)))
    K = len(d)
    i = math.comb(n + K - 1, K) - 1
    for k, c_k in enumerate(reversed(d)):
        i -= math.comb(n - 1 + k - c_k, k + 1)
    return int(i)


def multicomb(n: int, k: int) -> int:
    """ Equivalent to `len(list(itertools.combinations_with_replacement(range(n), k)))`
    """
    return math.comb(n + k - 1, k)


def permutation_count(d: Sequence[any]) -> int:
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
    _data: _Array


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


    def _normalize_key(self, key: any) -> tuple[list[Integral | slice], bool]:
        def _normalize_slice(s: slice) -> slice:
            start, stop, step = s.start, s.stop, s.step
            if start is None:
                start = 0
            if stop is None:
                stop = self.dim
            if step is None:
                step = 1
            s = slice(start, stop, step)
            if s != slice(0, self.dim, 1):
                raise KeyError(f"0:{self.dim}:1 is currently the only supported slice for tensor of dimension {self.dim}, got {s.start}:{s.stop}:{s.step}")
            return s

        contains_slice = False
        if isinstance(key, Iterable):
            key = list(key)
            for i, k in enumerate(key):
                if isinstance(k, slice):
                    key[i] = _normalize_slice(k)
                    contains_slice = True
                elif not isinstance(k, Integral):
                    raise TypeError(f"index for axis {i} must be int or slice, got {type(k)}")
        elif isinstance(key, Integral):
            key = [key]
        elif isinstance(key, slice):
            key = [_normalize_slice(key)]
            contains_slice = True
        else:
            raise TypeError(f"index must be Iterable, int or slice, got {type(key)}")

        if len(key) > self.rank:
            raise KeyError(f"too many indices for tensor of rank {self.rank}, got {len(key)}")
        if len(key) < self.rank:
            key += [slice(0, self.dim, 1)] * (self.rank - len(key))
            contains_slice = True

        return key, contains_slice


    def __getitem__(self, key):
        key, contains_slice = self._normalize_key(key)
        if not contains_slice:
            return self._data[index_of_multicombination(self.dim, sorted(key))]
        idx = np.array([k if isinstance(k, Integral) else -1 for k in key], dtype=object)
        subidx_map = [i for i, k in enumerate(key) if isinstance(k, slice)]
        return FlatSymmetricTensorSlice(self, idx, subidx_map)


    def __setitem__(self, key, value):
        key, contains_slice = self._normalize_key(key)
        if not contains_slice:
            if not isinstance(value, Number):
                raise TypeError(f"value must be a number, got {type(value)}")
            self._data[index_of_multicombination(self.dim, sorted(key))] = value
            return

        idx = np.array([k if isinstance(k, int) else -1 for k in key])
        subidx_map = [i for i, k in enumerate(key) if isinstance(k, slice)]
        sub_rank = len(subidx_map)
        if isinstance(value, np.ndarray) or isinstance(value, sp.sparray) or isinstance(value, base.SymmetricTensor):
            if value.shape != (self.dim,) * sub_rank:
                raise RuntimeError(f"value of wrong shape: expected {(self.dim,) * sub_rank}, got {value.shape}")
        elif not isinstance(value, Number):
            raise TypeError(f"value must be an array, a SymTensor, or a number; got {type(value)}.")

        if isinstance(value, Number):
            for subidx in multicombinations(range(self.dim), sub_rank):
                idx[subidx_map] = subidx
                self._data[index_of_multicombination(self.dim, sorted(idx))] = value
        else:
            for subidx in multicombinations(range(self.dim), sub_rank):
                idx[subidx_map] = subidx
                self._data[index_of_multicombination(self.dim, sorted(idx))] = value[subidx]


    def _set_raw_data(self, key: any, arr: _Array):
        if key != ():
            raise KeyError(f"key must be {()}, got {key}")
        self._data = arr


    def change_array_type(self, array_type: Callable[[_Array], _Array]):
        if array_type == np.array and hasattr(self._data, "todense"):
            self._data = self._data.todense()
        else:
            self._data = array_type(self._data)
        assert self._data.shape == (self.size,)


    @property
    def flat(self):
        for d, v in zip(self.indep_iter_repindex(), self.indep_iter()):
            yield from itertools.repeat(v, permutation_count(d))


    @property
    def flat_index(self):
        for d in self.indep_iter_repindex():
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
                d_inflated = sum(((x,) * c for x, c in zip(d_compressed, counts)), start = ())
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


class FlatSymmetricTensorSlice(FlatSymmetricTensor):
    data_format: ClassVar[str] = "FlatSlice"
    _data: FlatSymmetricTensor
    _idx: _NumpyArray[object]
    _subidx_map: list[int]


    def __init__(self, data: FlatSymmetricTensor, idx: _NumpyArray[object], subidx_map: list[int]):
        self._data = data
        self._idx = idx
        self._subidx_map = subidx_map


    @property
    def dim(self):
        return self._data.dim


    @property
    def rank(self):
        return len(self._subidx_map)


    def _expand_key(self, key: any) -> tuple[int | slice]:
        if isinstance(key, Iterable):
            key = tuple(key)
            for i, k in enumerate(key):
                if not isinstance(k, (int, slice)):
                    raise TypeError(f"index for axis {i} must be int or slice, got {type(k)}")
        elif isinstance(key, Integral):
            key = (key,)
        elif isinstance(key, slice):
            key = (key,)
        else:
            raise TypeError(f"index must be tuple, int or slice, got {type(key)}")

        if len(key) > self.rank:
            raise KeyError(f"too many indices for tensor of rank {self.rank}, got {len(key)}")
        if len(key) < self.rank:
            key += (slice(0, self.dim, 1),) * (self.rank - len(key))

        self._idx[self._subidx_map] = key
        return tuple(self._idx)


    def __getitem__(self, key):
        return self._data[self._expand_key(key)]


    def __setitem__(self, key, value):
        self._data[self._expand_key(key)] = value


    def indep_iter(self):
        for d in self.indep_iter_repindex():
            yield self[d]


    def permcls_indep_iter(self, σcls: str = None):
        for d in self.permcls_indep_iter_repindex(σcls):
            return self[d]


    def _set_raw_data(self, _: any, __: _Array):
        raise NotImplemented

    def change_array_type(self, _: Callable[[_Array], _Array]):
        raise NotImplemented

    def keys(self):
        raise NotImplemented

    def values(self):
        raise NotImplemented

    @dataclass
    class Data(FlatSymmetricTensor.Data):
        _symtensor_type: ClassVar[type]="FlatSymmetricTensorSlice"
