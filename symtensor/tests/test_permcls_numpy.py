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
# # Testing `PermClsSymmetricTensor`

# %%
import pytest
import numpy as np

from symtensor import utils
from symtensor.permcls_symtensor import (
    PermClsSymmetricTensor, get_index_representative, σindex_iter)
from symtensor.testing.api import SymTensorAPI
from symtensor.testing.utils import Code, NBTestRunner


# %% [markdown] tags=["remove-cell"]
# **Explanation of this document structure**
# This file is meant for two different types of consumption:
# - As the file `pytest` inspects to obtain the list of tests to run on this SymmetricTensor class
# - As an exhaustive list of usage examples, which uses these tests as sources.
#   (This list requires executing the code cells marked for notebook only;
#    it shows up in the documentation built by Jupyter Book and
#    when the file is executed in a Jupyter notebook.)
#
# **HINT**: This file is most easily edited when opened as a notebook in Jupyter.
#

# %% [markdown]
# To ensure a consistent API across formats and backends, the API usage tests are defined in the generic test class `SymTensorAPI` (it is not prefixed with "Test" to avoid it being picked up by pytest).
# Adding a new format or backend to the test requires only creating a subclass with a name starting with "Test". This class must define the `SymTensor` fixture as returning the symmetric tensor type to test:

# %% tags=["remove-cell"]
class TestPermClsSymtensorAPI(SymTensorAPI):
    @pytest.fixture
    def SymTensor(self):
        return PermClsSymmetricTensor

# %% [markdown] tags=["remove-cell"]
# In theory the code above is sufficient to run the full battery of standardized API tests on the this SymmetricTensor subclass when running `pytest` on the command line.
# In practice, certain subclasses might require that some tests need be specialized or deactivated; this we do by redefining them in code cells below.
# We can also add tests specific to a symmetric tensor type.
#
# **Important: All code cells must be indented, in order to stay defined with the Test class.**
#
# Keeping test definitions in separate cells allows us to intersperse them with explanations, in order to fulfill the second objective of this file, which is to document usage.
# This objective requires also also displaying the code for tests which are not modified, which should be the majority.
# We do this by instantiating a `NBTestRunner` in the next cell, which provides a callable which displays the test code and runs it.
#
# The next code cell, as well as all cells with a `show_test` call, are only executed when opened in notebook format, and when generating the docs. They are not seen by `pytest`.
# We achieve this by [tagging](https://jupytext.readthedocs.io/en/latest/formats.html#active-and-inactive-cells) them with `"active-ipynb"`.

# %% tags=["active-ipynb", "remove-input"]
#     API = TestPermClsSymtensorAPI()
#     show_test = NBTestRunner(TestPermClsSymtensorAPI, PermClsSymmetricTensor, display=True)
#     run_test = NBTestRunner(TestPermClsSymtensorAPI, PermClsSymmetricTensor, display=False)

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_perm_classes)

# %% [markdown]
# ## Instantiation & dtypes
#
# Can create SymmetricTensors of `float`, `int` and `bool` types.

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_creation_with_dtype)

# %% [markdown]
# Initializing with data.
# This test is format-specific, so needs to be overridden

    # %%
    def test_initialization_with_data(self, SymTensor):
        
        x = 3.14
        data = np.array([[1, 2],[2, 1]])

        # Init with scalar
        A = SymTensor(rank=2, dim=2, data=x, dtype=np.int16)
        assert A.dtype == "int16"
        assert A._data.keys() == {(2,), (1,1)}
        assert all(np.all(arr == int(x)) for arr in A._data.values())

        # Init with ndarray
        A2 = SymTensor(rank=2, dim=2, data=data)
        assert A2.dtype == data.dtype
        assert A2._data.keys() == {(2,), (1,1)}
        assert np.array_equal(A2._data[(2,)], [1, 1])   # 'ii'
        assert np.array_equal(A2._data[(1,1)], [2])  # 'ij'
        assert np.array_equal(A2.todense(), data)

        # Init with list
        A3 = SymTensor(rank=2, dim=2, data=A2._data, dtype=float)
        assert A3.dtype == "float64"  # When `data` doesn’t provide dtype, default is float64
        assert A3._data.keys() == set(utils._perm_classes(A3.rank)) == {(2,), (1,1)}
        assert np.array_equal(A3._data[(2,)], [1, 1])   # 'ii'
        assert np.array_equal(A3._data[(1,1)], [2])  # 'ij'
        assert np.array_equal(A3.todense(), data)

# %% tags=["active-ipynb", "remove-input"]
#     run_test(test_initialization_with_data)

# %% [markdown]
# Illegal initializations

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_illegal_initializations)

# %% [markdown]
# ## Iteration
#
# Test the index iterators.

# %% [markdown]
# - By default, `DenseSymmetricTensor` gets initialized as a zero tensor.
# - `flat*` return iterators all $d^r$ values, in the same order as a NumPy array.
# - `indep_iter*` iterators return $\binom{d + r - 1}{r}$ values
# - Iteration returns either $\binom{d + r - 1}{r}$ or $d^r$ values (depending on whether it returns permutations of symmetric terms).
# - `*_repindex` iterators return one representative index for each index class.
# - `*_index` iterators return all symmetry-equivalent indices, as one advanced index.

# %% [markdown]
# `permcls_indep_iter_repindex`
#
# This test is overridden because the order of representative indices is format-dependent.

# %% tags=["active-ipynb", "remove-input"]
#     show_test(API.test_permcls_indep_iter_repindex)

# %% [markdown]
# `permcls_indep_iter_index`
#
# This test is overridden because the order of representative indices is format-dependent.

# %% tags=["active-ipynb", "remove-input"]
#     show_test(API.test_indep_iter_repindex)

# %% [markdown]
# Correspondence of index and value iterators.

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_correspondence_index_value_iterators)


# %% [markdown]
# `σindex_iter`

    # %%
    def test_σindex_iter(self):
        assert list(σindex_iter((3,), 3))  == [(0,0,0), (1,1,1), (2,2,2)]
        assert list(σindex_iter((2,1), 2)) == [(0,0,1), (1,1, 0)]
        assert list(σindex_iter((2,1), 3)) == [(0,0,1), (0,0,2), (1,1,0), (1,1,2), (2,2,0), (2,2,1)]
        assert list(σindex_iter((2,2), 3)) == [(0,0,1,1), (0,0,2,2), (1,1,2,2)]

# %% [markdown]
# ## Indexing, assignment, reshaping

# %% [markdown]
# Test standardization of index class representatives: `get_index_representative`

    # %%
    def test_standardization_indexrep_dense(self):
        assert get_index_representative((2,1,2))         == (2,2,1)
        assert get_index_representative((1,1,2))         == (1,1,2)
        assert get_index_representative((0,0))           == (0,0)
        assert get_index_representative((5,4,3,3,2,1))   == (3, 3, 1, 2, 4, 5)


# %% tags=["active-ipynb", "remove-input"]
#     run_test(test_standardization_indexrep_dense)

# %% [markdown]
# Block assignment of already symmetrized data.

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_block_assignment)

# %% [markdown]
# Element-wise assignment: Assigning one value modifies all associated symmetric components.

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_elementwise_assignment)

# %% [markdown]
# Indexing & assignment of σ-classes

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_σcls_assignment)

# %% [markdown]
# Subtensors: indexing with partial indices, returning a lower rank SymmetricTensor

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_partial_indexing)

# %% [markdown]
# Transposition is always equal to the identity.

# %% tags=["active-ipynb"]
#     show_test(API.test_transpose)

# %% [markdown]
# ## Comparisons
# Test `array_equal`, `allclose`, `isclose` array functions

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_comparisons)

# %% [markdown]
# ## Copying

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_copy)

# %% [markdown]
# ## Serialization

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_serialization)

# %% [markdown]
# ## Avoiding array coercion
#
# `asarray` works as one would expect (converts to dense array by default, does not convert if `like` argument is used).

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_asarray)

# %% [markdown]
# ## Arithmetic

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_arithmetic)

# %% [markdown]
# ## Outer product

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_outer_product)

# %% [markdown]
# ## Tensordot

# %% tags=["remove-input", "active-ipynb"]
#     display(Code(API._test_tensordot))
#     show_test(API.test_tensordot)

# %% [markdown]
# ## Contractions

# %% [markdown]
# ### Contraction with matrix along all indices

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_contract_all_indices_with_matrix)

# %% [markdown]
# ### Contraction with list of SymmetricTensors

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_contract_tensor_list)

# %% [markdown]
# ### Contraction with vector

# %% tags=["remove-input", "active-ipynb"]
#     show_test(API.test_contract_all_indices)
