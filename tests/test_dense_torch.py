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
# # Testing `DenseTorchSymmetricTensor`

# %%
from mackelab_toolbox.utils import Code

# %%
import statGLOW.typing
from statGLOW.stats.symtensor.symtensor.torch_symtensor import DenseTorchSymmetricTensor
from statGLOW.stats.symtensor.symtensor.testing import unittests

# %% [markdown]
# ## Instantiation & dtypes
#
# Can create SymmetricTensors of `float`, `int` and `bool` types.

# %% tags=["remove-input"]
display(Code(unittests.test_perm_classes))
unittests.test_perm_classes(DenseTorchSymmetricTensor)

# %% [markdown]
# Initializing with data.

# %% tags=["remove-input"]
display(Code(unittests.test_initialization_with_data))
unittests.test_initialization_with_data(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Iteration
#
# Test the index iterators.

# %% [markdown]
# - By default, `DenseTorchSymmetricTensor` gets initialized as a zero tensor.
# - `flat*` return iterators all $d^r$ values, in the same order as a NumPy array.
# - `indep_iter*` iterators return $\binom{d + r - 1}{r}$ values
# - Iteration returns either $\binom{d + r - 1}{r}$ or $d^r$ values (depending on whether it returns permutations of symmetric terms).
# - `*_repindex` iterators return one representative index for each index class.
# - `*_index` iterators return all symmetry-equivalent indices, as one advanced index.

# %% [markdown]
# `permcls_indep_iter_repindex`

# %% tags=["remove-input"]
display(Code(unittests.test_permcls_indep_iter_repindex))
unittests.test_permcls_indep_iter_repindex(DenseTorchSymmetricTensor)

# %% [markdown]
# `permcls_indep_iter_index`

# %% tags=["remove-input"]
display(Code(unittests.test_indep_iter_repindex))
unittests.test_indep_iter_repindex(DenseTorchSymmetricTensor)

# %% [markdown]
# Correspondence of index and value iterators.

# %% tags=["remove-input"]
display(Code(unittests.test_correspondence_index_value_iterators))
unittests.test_correspondence_index_value_iterators(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Indexing & assignment

# %% [markdown]
# Test standardization of index class representatives: `get_index_representative` (skipped; [tested with base class](./test_dense_numpy.py)).

# %% [raw]
# def test_standardization_indexrep_dense():
#     assert get_index_representative((2,1,2))         == (1,2,2)
#     assert get_index_representative((1,1,2))         == (1,1,2)
#     assert get_index_representative((0,0))           == (0,0)
#     assert get_index_representative((5,4,3,3,2,1))   == (1,2,3,3,4,5)

# %% [markdown]
# Block assignment of already symmetrized data.

# %% tags=["remove-input"]
display(Code(unittests.test_block_assignment))
unittests.test_block_assignment(DenseTorchSymmetricTensor)

# %% [markdown]
# Element-wise assignement: Assigning one value modifies all associated symmetric components.

# %% tags=["remove-input"]
display(Code(unittests.test_elementwise_assignment))
unittests.test_elementwise_assignment(DenseTorchSymmetricTensor)

# %% [markdown]
# Indexing & assignment of σ-classes

# %% tags=["remove-input"]
display(Code(unittests.test_σcls_assignment))
unittests.test_σcls_assignment(DenseTorchSymmetricTensor)

# %% [markdown]
# Subtensors: indexing with partial indices, returning a lower rank SymmetricTensor

# %% tags=["remove-input"]
display(Code(unittests.test_partial_indexing))
unittests.test_partial_indexing(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Comparisons
# Test `array_equal`, `allclose`, `isclose` array functions

# %% tags=["remove-input"]
display(Code(unittests.test_comparisons))
unittests.test_comparisons(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Copying

# %% tags=["remove-input"]
display(Code(unittests.test_copy))
unittests.test_copy(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Serialization

# %% tags=["remove-input"]
display(Code(unittests.test_serialization))
unittests.test_serialization(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Avoiding array coercion
#
# `asarray` works as one would expect (converts to dense array by default, does not convert if `like` argument is used).

# %% tags=["remove-input"]
display(Code(unittests.test_asarray))
unittests.test_asarray(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Arithmetic

# %% tags=["remove-input"]
display(Code(unittests.test_arithmetic))
unittests.test_arithmetic(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Outer product

# %% tags=["remove-input"]
display(Code(unittests.test_outer_product))
unittests.test_outer_product(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Tensordot

# %% tags=["remove-input"]
display(Code(unittests._test_tensordot))
display(Code(unittests.test_tensordot))
unittests.test_tensordot(DenseTorchSymmetricTensor)

# %% [markdown]
# ## Contractions

# %% [markdown]
# ### Contraction with matrix along all indices

# %% tags=["remove-input"]
display(Code(unittests.test_contract_all_indices_with_matrix))
unittests.test_contract_all_indices_with_matrix(DenseTorchSymmetricTensor)

# %% [markdown]
# ### Contraction with list of SymmetricTensors

# %% tags=["remove-input"]
display(Code(unittests.test_contract_tensor_list))
unittests.test_contract_tensor_list(DenseTorchSymmetricTensor)

# %% [markdown]
# ### Contraction with vector

# %% tags=["remove-input"]
display(Code(unittests.test_contract_all_indices))
unittests.test_contract_all_indices(DenseTorchSymmetricTensor)
