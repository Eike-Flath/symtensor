# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     notebook_metadata_filter: -jupytext.text_representation.jupytext_version,-kernelspec
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
# ---

# %% [markdown]
# # Unit tests for `SymmetricTensor`
#
# Collection of unit tests for validating subclasses of `SymmetricTensor`.
#
# This defines the generic API which subclasses of `SymmetricTensor` provide.

# %% [markdown] tags=["remove-input"]
# **Explanation of this document structure**
# This file is meant for two different types of consumption:
# - As the file defining `pytest` tests for the SymmetricTensor API.
# - For Jupyter Book to produce browsable documentation for those tests.
#
# To ensure a consistent API across formats and backends, the test suite for each subclass of `SymmetricTensor` subclasses the `SymTensorAPI` defined here, adjusting it for that particular subclass. In particular, each test suite needs to define the `SymTensor` fixture defining the subclass to test.
#
# The test suites also reprint the source code for each test (substituting their class for `SymTensor`, thus serving as a kind of auto-generated documentation for each subclass. For example, the docs page generated from the test suite for `PermClsTorchSymmetricTensor` can serve as a self-contained, exhaustive list of examples for the usage examples supported and tested with that particular subclass.
#
# **Important:** In order to support the aforementioned dual usage, **all code cells must be indented**. This ensures that when consumed as a script, all the tests are part of the `SymTensorAPI` class.

# %%
import abc
from collections import Counter
import itertools
import math
from typing import Generator, Type

import numpy as np
from tqdm.auto import tqdm

import pytest
# %%

import symtensor as st
from symtensor import SymmetricTensor
from symtensor import utils, symalg
from symtensor.testing.utils import does_not_warn
# For serialization test
from scityping import Serializable
from scityping.numpy import Array, DType
from scityping.pydantic import BaseModel

# %%
class SymTensorAPI:

    # TODO? Convert to a parameterized fixture ?
    def get_test_tensors(self, SymTensor, max_dim=8, max_rank=6) -> Generator:
        for d, r in itertools.product([2, 3, 4, 6, 8], [2, 3, 4, 5, 6]):
            if d <= max_dim and r <= max_rank:
                A = SymTensor(rank=r, dim=d)
                # Assign some values so we aren’t just testing with a zero vector
                A["i"*r] = np.random.normal(size=utils._get_permclass_size((r,), d))
                ci = r//2; cj = r - r//2
                A["i"*ci + "j"*cj] = np.random.normal(size=utils._get_permclass_size((ci, cj), d))
                yield A

    @pytest.fixture
    def test_tensor(self, SymTensor):
        return next(iter(self.get_test_tensors(SymTensor)))
         


    # %%
    def test_perm_classes(self, SymTensor):
        assert SymTensor(rank=2, dim=3, data=None).perm_classes == \
            ('ii', 'ij')
        assert SymTensor(rank=4, dim=3, data=None).perm_classes == \
            ('iiii', 'iiij', 'iijj', 'iijk', 'ijkl')
        assert SymTensor(rank=5, dim=3, data=None).perm_classes == \
            ('iiiii', 'iiiij', 'iiijj', 'iiijk', 'iijjk', 'iijkl', 'ijklm')

# %% [markdown]
# ## Instantiation & dtypes
#
# Can create SymmetricTensors of `float`, `int` and `bool` types.

    # %%
    def test_creation_with_dtype(self, SymTensor):
        assert SymTensor(rank=3, dim=3).dtype == 'float64'
        assert SymTensor(rank=3, dim=3, dtype=int).dtype == 'int64'
        assert SymTensor(rank=3, dim=3, dtype=np.int32).dtype == 'int32'
        assert SymTensor(rank=3, dim=3, dtype=bool).dtype == bool
        with pytest.raises(TypeError):
            SymTensor(rank=3, dim=3, dtype=str, data="foo")


# %% [markdown]
# Initializing with data.
# (Marked as an abstract test because it must be specialized for each data format)

    # %%
    @abc.abstractmethod
    def test_initialization_with_data(self, SymTensor):
        
        data = np.array([[1, 2],[2, 1]])

        # Init with scalar
        A = SymTensor(rank=2, dim=2, data=1., dtype=np.int16)
        assert A.dtype == "int16"
        # vvv SPECIFY TEST IN SUBCLASS vvv
        assert np.array_equal(A._data, np.array([[1,1],[1,1]]))

        # Init with ndarray
        A = SymTensor(rank=2, dim=2, data=data)
        assert A.dtype == data.dtype
        # vvv SPECIFY TEST IN SUBCLASS vvv
        assert np.array_equal(A._data, data)

        # Init with list
        A = SymTensor(rank=2, dim=2, data=data.tolist(), dtype=float)
        assert A.dtype == "float64"
        # vvv SPECIFY TEST IN SUBCLASS vvv
        assert np.array_equal(A._data, data)

# %% [markdown]
# Illegal initializations
# (Split from test above since these tests are reusable)

    # %%
    def test_illegal_initializations(self, SymTensor):

        data = np.array([[1, 2],[2, 1]])

        # TypeError if dim or rank are missing
        with pytest.raises(TypeError):
            SymTensor(rank=2)
        with pytest.raises(TypeError):
            SymTensor(dim=2)
        
        # Rank & dim are required when initializing with a scalar
        with pytest.raises(TypeError):
            SymTensor(data=5.)
        SymTensor(rank=0, dim=3, data=5.)
        # But we can infer a rank if we initialize with a 0-rank array
        # (inferred rank is 0, which can be overwritten by passing the `rank` argument)
        with pytest.raises(TypeError):
            SymTensor(data=np.array(5.))  # Dim is still required, because all data have the same () shape when rank = 0
        SymTensor(dim=3, data=np.array(5.))

        # ValueError if dim/rank are not compatible with data
        with pytest.raises(ValueError):
            SymTensor(rank=2, dim=3, data=data)
        with pytest.raises(ValueError):
            SymTensor(rank=1, dim=3, data=data)

        # Broadcasting to higher rank is theoretically allowed,
        # but one must make sure that the resulting array is symmetric
        with pytest.raises(ValueError):
            A = SymTensor(rank=3, dim=2, data=data)
        A = SymTensor(rank=3, dim=2, data=data, symmetrize=True)


# %% [markdown]
# ## Iteration
#
# Test the index iterators.

# %% [markdown]
# - By default, `SymTensor` gets initialized as a zero tensor.
# - `flat*` return iterators all $d^r$ values, in the same order as a NumPy array.
# - `indep_iter*` iterators return $\binom{d + r - 1}{r}$ values
# - Iteration returns either $\binom{d + r - 1}{r}$ or $d^r$ values (depending on whether it returns permutations of symmetric terms).
# - `*_repindex` iterators return one representative index for each index class.
# - `*_index` iterators return all symmetry-equivalent indices, as one advanced index.

# %% [markdown]
# `permcls_indep_iter_repindex`

    # %%
    def nested_sort(self, list_of_lists):
        return sorted([tuple(sorted(l)) for l in list_of_lists])

    # %%
    def test_permcls_indep_iter_repindex(self, SymTensor):
        A33 = SymTensor(rank=3, dim=3, data=None)
        A32 = SymTensor(rank=3, dim=2, data=None)
        A43 = SymTensor(rank=4, dim=3, data=None)
        assert self.nested_sort(A33.permcls_indep_iter_repindex('iii'))  == [(0,0,0), (1,1,1), (2,2,2)]
        assert self.nested_sort(A32.permcls_indep_iter_repindex('iij'))  == [(0,0,1), (0,1,1)]
        assert self.nested_sort(A33.permcls_indep_iter_repindex('iij'))  == [(0, 0, 1), (0, 0, 2), (0, 1, 1), (0, 2, 2), (1, 1, 2), (1, 2, 2)]
        assert self.nested_sort(A43.permcls_indep_iter_repindex('iijj')) == [(0,0,1,1), (0,0,2,2), (1,1,2,2)]
        # permcls_indep_iter_repindex returns a unique index I for each index class
        A = SymTensor(rank=4, dim=8)
        assert len(list(A.indep_iter_index())) == len(list(A.indep_iter()))
        # # TODO: The following asserts could be added to the PermClsSymmetricTensor test
        # assert len(list(A.indep_iter_index())) == len(list(A.indep_iter())) == A.size
        # assert len({str(Counter(sorted(idx))) for idx in A.indep_iter_repindex()}) == A.size


# %% [markdown]
# `permcls_indep_iter_index`

    # %%
    def test_permcls_indep_iter_index(self, SymTensor):
        A33 = SymTensor(rank=3, dim=3, data=None)
        A32 = SymTensor(rank=3, dim=2, data=None)
        assert self.nested_sort(A33.permcls_indep_iter_index('iii')) == [
            ([0], [0], [0]),
            ([1], [1], [1]),
            ([2], [2], [2])
        ]
        assert self.nested_sort(A32.permcls_indep_iter_index('iij')) == [
            ([0, 0, 1], [0, 1, 0], [1, 0, 0]),
            ([0, 1, 1], [1, 0, 1], [1, 1, 0])
        ]
        assert self.nested_sort(A33.permcls_indep_iter_index('iij')) == [
            ([0, 0, 1], [0, 1, 0], [1, 0, 0]),
            ([0, 0, 2], [0, 2, 0], [2, 0, 0]),
            ([0, 1, 1], [1, 0, 1], [1, 1, 0]),
            ([0, 2, 2], [2, 0, 2], [2, 2, 0]),
            ([1, 1, 2], [1, 2, 1], [2, 1, 1]),
            ([1, 2, 2], [2, 1, 2], [2, 2, 1])
        ]
        A = SymTensor(rank=4, dim=8)
        assert all(len(list(A.permcls_indep_iter_index(σcls))) == len(list(A.permcls_indep_iter(σcls)))
                   for σcls in A.perm_classes)


# %% [markdown]
# `indep_iter_repindex`

    # %%
    def test_indep_iter_repindex(self, SymTensor):
        A33 = SymTensor(rank=3, dim=3, data=None)
        A32 = SymTensor(rank=3, dim=2, data=None)
        assert self.nested_sort(A32.indep_iter_repindex())  == [(0,0,0), (0,0,1), (0,1,1), (1,1,1)]
        assert self.nested_sort(A33.indep_iter_repindex())  == [(0,0,0), (0,0,1), (0,0,2), (0,1,1), (0,1,2), (0,2,2),
                                                    (1,1,1), (1,1,2), (1,2,2), (2,2,2)]


# %% [markdown]
# Correspondence of index and value iterators.

    # %%
    def test_correspondence_index_value_iterators(self, SymTensor):
        for A in self.get_test_tensors(SymTensor):
            # flat
            Adense = A.todense()
            assert all(Adense[idx] == val for val, idx in zip(A.flat, A.flat_index))
            assert (len(list(A.flat))
                    == len(list(A.flat_index))
                    == A.dim**A.rank)
            # indep_iter
            assert (len(list(A.indep_iter()))
                    == len(list(A.indep_iter_index()))
                    == len(list(A.indep_iter_repindex()))
                    == A.indep_size)
            # permcls_indep_iter
            assert all(len(list(A.permcls_indep_iter(σcls)))
                       == len(list(A.permcls_indep_iter_index(σcls)))
                       == len(list(A.permcls_indep_iter_repindex(σcls)))
                       for σcls in A.perm_classes)


# %% [markdown]
# ## Indexing, assignment, reshaping

# %% [markdown]
# Block assignment of already symmetrized data.

    # %%
    def test_block_assignment(self, SymTensor):
        A = SymTensor(3, 5)
        Adata = utils.symmetrize(np.arange(A.dim**A.rank).reshape(A.shape))
        A[:] = Adata
        assert np.array_equal(A.todense(), Adata)


# %% [markdown]
# Element-wise assignement: Assigning one value modifies all associated symmetric components.

    # %%
    def test_elementwise_assignment(self, SymTensor):
        A = SymTensor(3, 3)
        A[1, 2, 0] = 1
        assert np.array_equal(
            A.todense(),
            np.array([[[0., 0., 0.],
                       [0., 0., 1.],
                       [0., 1., 0.]],

                      [[0., 0., 1.],
                       [0., 0., 0.],
                       [1., 0., 0.]],

                      [[0., 1., 0.],
                       [1., 0., 0.],
                       [0., 0., 0.]]])
        )


# %% [markdown]
# Indexing & assignment of σ-classes

    # %%
    def test_σcls_assignment(self, SymTensor):
        A = SymTensor(3, 5, data=np.arange(5**3, dtype=np.int16).reshape(5,5,5), symmetrize=True)
        assert len(A['iij']) == utils.get_permclass_size('iij', A.dim)

        b = 0
        sizes = (utils.get_permclass_size(σcls, A.dim) for σcls in A.perm_classes)
        for σcls, size in zip(A.perm_classes, sizes):
            if σcls == "iii":
                # Test indexing with both scalar and list entries
                A[σcls] = 0
            else:
                A[σcls] = np.arange(b, b+size)
            b += size

        assert all(A[i,i,i] == 0 for i in range(A.dim))
        assert A[1, 1, 1] == 0
        assert A[0, 0, 3] == A['iij'][2]   # Preceded by: (0,0,1),(0,0,2)
        k = list(A.permcls_indep_iter_repindex('iij')).index((2,2,3))
        assert A[2, 2, 3] == A['iij'][k]  # Preceded by: (0,0,1–4),(0,1–4,1–4),(1,1,2–4),(1,2–4,2–4)
        k = list(A.permcls_indep_iter_repindex('ijk')).index((1,2,3))
        assert A[1, 2, 3] == A['ijk'][6]   # Preceded by: (0,1,2—4),(0,2,3–4),(0,3,4)


# %% [markdown]
# Subtensors: indexing with partial indices, returning a lower rank SymmetricTensor

    # %%
    def test_partial_indexing(self, SymTensor):
        # Subtensor generation with 1 index
        for A in self.get_test_tensors(SymTensor):
            for i in range(A.dim):
                assert np.array_equal(A[i].todense(), A.todense()[i])

        # Subtensor generation with multiple indices
        dim = 4
        rank = 4
        #test is_equal
        diagonal = np.random.rand(dim)
        odiag1 = np.random.rand()
        odiag2 = np.random.rand()
        A = SymTensor(rank=rank, dim=dim)
        A['iiii'] = diagonal
        A['iiij'] = odiag1
        A['iijj'] = odiag2

        assert np.array_equal(A[0,1,:,:].todense(), A.todense()[0,1,:,:])
        assert np.array_equal(A[0,1,:,:], A[1,0,:,:])
        assert np.array_equal(A[0,1,1,:], A[1,1,0,:])
        assert all([A[0,0,0,:][i] == A[0,0,0,i] for i in range(dim)])


# %% [markdown]
# Transposition is always equal to the identity.

    # %%
    def test_transpose(self, SymTensor):
        for A in self.get_test_tensors(SymTensor):
            # Default transposition
            assert np.array_equal(A, A.transpose())    # Method interface
            assert np.array_equal(A, np.transpose(A))  # Array function interface
            # Transposition with a random permutation of the axes
            assert np.array_equal(A, A.transpose(np.random.permutation(np.arange(A.rank))))
            assert np.array_equal(A, np.transpose(A, np.random.permutation(np.arange(A.rank))))

# %% [markdown]
# ## Comparisons
# Test `array_equal`, `allclose`, `isclose` array functions

    # %%
    def test_comparisons(self, SymTensor):
        rank = 4
        dim = 15
        #test array_equal, allclose
        diagonal = np.random.rand(dim)
        odiag1 = np.random.rand()
        odiag2 = np.random.rand()
        A = SymTensor(rank=rank, dim=dim)
        B = SymTensor(rank=rank, dim=dim)
        A['iiii'] = diagonal
        B['iiii'] = diagonal
        A['iiij'] = odiag1
        B['iiij'] = odiag1
        A['iijj'] = odiag2
        B['iijj'] = odiag2
        
        assert np.array_equal(A, B)
        assert np.allclose(A, B)
        assert np.all(np.isclose(A, B) == SymTensor(rank=rank, dim=dim, data=True))


# %% [markdown]
# ## Copying

    # %%
    def test_copy(self, test_tensor):
        A = test_tensor
        C = A.copy()
        assert np.array_equal(C, A)


# %% [markdown]
# ## Serialization

    # %%
    def test_serialization(self, test_tensor):
        A = test_tensor

        A_json = SymmetricTensor.json_encoder(A)
        A_deserialized = SymmetricTensor.validate(A_json)
        assert str(A_deserialized) == str(A)
        assert str(SymmetricTensor.json_encoder(A_deserialized)) == str(A_json)

        class Foo(BaseModel):
            A: SymmetricTensor
        foo = Foo(A=A)
        foo2 = Foo.parse_raw(foo.json())
        assert foo2.json() == foo.json()


# %% [markdown]
# ## Avoiding array coercion
#
# `asarray` works as one would expect (converts to dense array by default, does not convert if `like` argument is used).

    # %%
    def test_asarray(self, SymTensor):
        A = SymTensor(rank=2, dim=3)
        with pytest.warns(UserWarning):
            assert type(np.asarray(A)) is np.ndarray
        # `like` argument is supported and avoids the conversion to dense array
        with does_not_warn(UserWarning):
            assert type(np.asarray(A, like=SymTensor(0,0))) is SymTensor


# %% [markdown]
# ## Arithmetic

    # %%
    def test_arithmetic(self, SymTensor):
        rank = 4
        dim = 2
        test_tensor_1 = SymTensor(rank=rank, dim=dim)
        test_tensor_1['iiii'] = np.random.rand(2)
        test_tensor_3 = SymTensor(rank=rank, dim=dim, data=1.0)
        
        #test addition
        test_tensor_2 = np.add(test_tensor_1, 1.0)
        assert np.array_equal(test_tensor_2, test_tensor_1 + 1.0)
        test_tensor_4 =  test_tensor_2 - test_tensor_3
        assert np.allclose(test_tensor_4, test_tensor_1)
        
        #test multiplication
        test_tensor_5 = np.multiply(test_tensor_2, -1)
        test_tensor_6 = np.multiply(test_tensor_5, -1)
        assert np.allclose(test_tensor_6, test_tensor_2)
        
        #test log, exp
        test_tensor_7 = np.exp(test_tensor_2)
        test_tensor_8 = np.log(test_tensor_7)
        assert np.allclose(test_tensor_8, test_tensor_2)


# %% [markdown]
# ## Outer product

    # %%
    def test_outer_product(self, SymTensor):
        for A, B in zip(self.get_test_tensors(SymTensor, max_dim=2, max_rank=4),
                        self.get_test_tensors(SymTensor, max_dim=2, max_rank=4)):
            Ad = A.todense()
            Bd = B.todense()
            # assert np.allclose(np.multiply.outer(A,B).todense(),
            #                    np.multiply.outer(Ad,Bd))
            with pytest.raises(TypeError):
                # We prevent outer products between symmetric tensors because
                # calling the non-symmetrized op on symmetrized tensors is
                # likely a mistake
                np.multiply.outer(A,B)
            assert np.allclose(st.multiply.outer(A,B).todense(),
                               utils.symmetrize(np.multiply.outer(Ad,Bd)))
        
        for test_tensor_1 in self.get_test_tensors(SymTensor, max_dim=2, max_rank=4):
            dim = test_tensor_1.dim
            rank = test_tensor_1.rank
            test_tensor_2 = test_tensor_1 + 1.0
            test_tensor_3 = SymTensor(rank=rank, dim=dim, data=1.0)
        
            test_tensor_1d = test_tensor_1.todense()
            test_tensor_2d = test_tensor_2.todense()
            test_tensor_3d = test_tensor_3.todense()

            test_tensor_8 = st.multiply.outer(test_tensor_2, test_tensor_3)
            assert np.allclose(test_tensor_8.todense(),
                               utils.symmetrize(np.multiply.outer(test_tensor_2d, test_tensor_3d)))
            test_tensor_9 = st.multiply.outer(test_tensor_1, test_tensor_3)
            assert np.allclose(test_tensor_9.todense(),
                               utils.symmetrize(np.multiply.outer(test_tensor_1d, test_tensor_3d)))

            test_tensor_10 = SymTensor(rank=1, dim=2)
            test_tensor_10['i'] = [1,0]
            test_tensor_11 = SymTensor(rank=1, dim=2)
            test_tensor_11['i'] = [0,1]
            test_tensor_12 = st.multiply.outer(test_tensor_10,test_tensor_11)
            assert test_tensor_12[0,0] == 0 and test_tensor_12[1,1] == 0
            assert test_tensor_12['ij'] == 0.5


# %% [markdown]
# ## Tensordot
# Test outer producing with tensordot

    # %% tags=[]
    def _test_tensordot(self, tensor_1, tensor_2):
        test_tensor_13 = st.tensordot(tensor_1, tensor_2, axes=0)
        assert np.allclose(test_tensor_13, st.multiply.outer(tensor_1,tensor_2))

        #Contract over first and last indices:
        test_tensor_14 =  st.tensordot(tensor_1, tensor_2, axes=1)
        dense_tensor_14 = utils.symmetrize(np.tensordot(
            tensor_1.todense(), tensor_2.todense(), axes=1 ))
        assert np.allclose(test_tensor_14.todense(), dense_tensor_14)

        test_tensor_141 =  st.tensordot(tensor_1, tensor_2, axes = (0,1))
        assert np.allclose(test_tensor_14, test_tensor_141)

        #Contract over two first and last indices:
        test_tensor_15 = st.tensordot(tensor_1, tensor_2, axes = 2)
        dense_tensor_15 = utils.symmetrize(np.tensordot(
            tensor_1.todense(), tensor_2.todense(), axes = 2 ))
        if isinstance(test_tensor_15, SymmetricTensor):
            assert np.allclose(test_tensor_15.todense(), dense_tensor_15)
        else:
            assert np.array_equal(test_tensor_15, dense_tensor_15)

        if tensor_1.rank > 2 and tensor_2.rank > 2:
            test_tensor_16 =  st.tensordot(tensor_1, tensor_2, axes=((0,1,2),(0,1,2)))
            dense_tensor_16 = utils.symmetrize(np.tensordot(
                tensor_1.todense(), tensor_2.todense(), axes=((0,1,2),(0,1,2)) ))
            dense_tensor_161 = utils.symmetrize(np.tensordot(
                tensor_1.todense(), tensor_2.todense(), axes=((0,1,2),(2,1,0)) ))
            dense_tensor_162 = utils.symmetrize(np.tensordot(
                tensor_1.todense(), tensor_2.todense(), axes=((0,1,2),(2,0,1)) ))
            assert np.allclose(test_tensor_16.todense(), dense_tensor_16)
            assert np.allclose(test_tensor_16.todense(), dense_tensor_161)
            assert np.allclose(test_tensor_16.todense(), dense_tensor_162)


    # %% tags=[]
    def test_tensordot(self, SymTensor):
        test_tensor_list = list(self.get_test_tensors(SymTensor, max_dim=2, max_rank=4))
        for A, B in tqdm(itertools.combinations(test_tensor_list, 2),
                         total=math.comb(len(test_tensor_list), 2)):
            # Using itertools instead of nested loops avoids generating duplicate combinations
            # Restrictions:
            #    - Test will create a tensor of rank A.rank + B.rank  (=> D**(A.rank+B.rank) elements)
            #    - Summing over all permutations of a rank 12 tensor is *extremely* slow (4096 permutations). We stop at 9 here, and for laptops this is still probably too high
            #    - In addition, arrays are converted to dense, and 1 million entries is about as much as a 16GB machine can cope with
            if A.rank + B.rank <= 9 and A.dim**(A.rank+B.rank) <= 1e6:
                self._test_tensordot(A, B)


# %% [markdown]
# ## Contractions

# %% [markdown]
# ### Contraction with matrix along all indices

    # %%
    def test_contract_all_indices_with_matrix(self, SymTensor):
        A = SymTensor(rank=3, dim=3)
        A[0,0,0] =1
        A[0,0,1] =-12
        A[0,1,2] = 0.5
        A[2,2,2] = 1.0
        A[0,2,2] = -30
        A[1,2,2] = 0.1
        A[1,1,1] =-0.3
        A[0,1,1] = 13
        A[2,1,1] = -6
        W = np.random.rand(3,3)
        W1 = np.random.rand(3,3)
        W2 = np.random.rand(3,3)
        assert np.allclose(symalg.contract_all_indices_with_matrix(A, W).todense(),
                           utils.symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W,W,W)))
        assert np.allclose(symalg.contract_all_indices_with_matrix(A, W1).todense(),
                           utils.symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W1,W1,W1)))
        assert np.allclose(symalg.contract_all_indices_with_matrix(A, W2).todense(),
                           utils.symmetrize(np.einsum('abc, ai,bj,ck -> ijk', A.todense(), W2,W2,W2)))

        B = SymTensor(rank=4, dim=4)
        B['iiii'] = np.random.rand(4)
        B['ijkl'] =12
        B['iijj'] = np.random.rand(6)
        B['ijkk'] =-0.5
        W = np.random.rand(4,4)
        C = symalg.contract_all_indices_with_matrix(B, W)
        W1 = np.random.rand(4,4)
        W2 = np.random.rand(4,4)
        assert np.allclose(symalg.contract_all_indices_with_matrix(C, W).todense(),
                           utils.symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W,W,W,W)))
        assert np.allclose(symalg.contract_all_indices_with_matrix(C, W1).todense(),
                           utils.symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W1,W1,W1,W1)))
        assert np.allclose(symalg.contract_all_indices_with_matrix(C, W2).todense(),
                           utils.symmetrize(np.einsum('abcd, ai,bj,ck, dl -> ijkl', C.todense(), W2,W2,W2,W2)))


# %% [markdown]
# ### Contraction with list of SymmetricTensors

    # %%
    def test_contract_tensor_list(self, SymTensor):
        dim = 4
        for dim in [2,3,4,5]: # Not too high dimensionality, because dense tensor operations
            test_tensor = SymTensor(rank =3, dim = dim)
            test_tensor['iii'] = np.random.rand(dim)
            test_tensor['ijk'] = np.random.rand(int(dim*(dim-1)*(dim-2)/6))
            test_tensor['iij'] = np.random.rand(int(dim*(dim-1)))

            tensor_list = []
            chi_dense = np.zeros( (dim,)*3)

            def get_random_symtensor_rank2(dim):
                tensor = SymTensor(rank=2, dim =dim)
                tensor['ii'] = np.random.rand(dim)
                tensor['ij'] = np.random.rand(int((dim**2 -dim)/2))
                return tensor
            
            for i in range(dim):
                random_tensor = get_random_symtensor_rank2(dim)
                tensor_list.append(random_tensor)
                chi_dense[i,:,:] = random_tensor.todense()

            contract_1 = symalg.contract_tensor_list(test_tensor, tensor_list,
                                                     n_times=1, rule='all')
            contract_2 = symalg.contract_tensor_list(test_tensor, tensor_list,
                                                     n_times=2, rule='all')

            assert  np.allclose(contract_1.todense(),
                                utils.symmetrize(np.einsum('ija, akl -> ijkl',
                                                           test_tensor.todense(), chi_dense)))
            assert  np.allclose(contract_2.todense(),
                                utils.symmetrize(np.einsum('iab, ajk, blm -> ijklm',
                                                           test_tensor.todense(), chi_dense, chi_dense)))


# %% [markdown]
# ### Contraction with vector

    # %%
    def test_contract_all_indices(self, SymTensor):
        A = SymTensor(rank=3, dim=3)
        A[0,0,0] =1
        A[0,0,1] =-12
        A[0,1,2] = 0.5
        A[2,2,2] = 1.0
        A[0,2,2] = -30
        A[1,2,2] = 0.1
        x = np.random.rand(3)
        x1 = np.random.rand(3)
        x2 = np.zeros(3)
        assert np.isclose(symalg.contract_all_indices_with_vector(A, x),
                          np.einsum('abc, a,b,c -> ', A.todense(), x,x,x))
        assert np.isclose(symalg.contract_all_indices_with_vector(A, x1),
                          np.einsum('abc, a,b,c -> ', A.todense(), x1,x1,x1))
        assert np.isclose(symalg.contract_all_indices_with_vector(A, x2), 0)
