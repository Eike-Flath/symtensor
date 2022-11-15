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
# # Test utilities

# %%
import pytest
import math
import numpy as np
import torch

# %%
import symtensor as st
from symtensor import utils
import symtensor.torch_symtensor

# %% [markdown]
# ## `symmetrize`

# %%
def test_symmetrize():
    dt = np.dtype("int8")
    for r in (2, 3, 4, 5):
        for S in (0, 1, 10, 300):
            size_in_MiB = S**r * dt.itemsize / 1e6
            if size_in_MiB > 1:
                continue
            A = np.random.randint(np.iinfo(dt).min, np.iinfo(dt).max,
                                  size=(S,)*r, dtype=dt)
            At= torch.tensor(A)

            # Standard symmetrize
            if S > 1:
                assert not utils.is_symmetric(A)
            Asym = utils.symmetrize(A)
            assert utils.is_symmetric(Asym)
            utils.symmetrize(A, out=A)
            assert np.array_equal(A, Asym)

            # Torch symmetrize
            Atsym = utils.symmetrize(At)
            assert isinstance(Atsym, torch.Tensor)
            utils.symmetrize(At, out=At)
            assert np.array_equal(At, Atsym)

            # SymmetricTensor symmetrize
            As = st.PermClsSymmetricTensor(data=Asym)
            assert utils.symmetrize(As) is As
            Bs = st.PermClsSymmetricTensor(rank=As.rank, dim=As.dim)
            utils.symmetrize(As, out=Bs)
            assert np.array_equal(As, Bs)

# %% tags=["active-ipynb"]
# test_symmetrize()

# %% [markdown]
# ## Combinatorics
#
# The two tests below check the functions `utils.get_permclass_size`, `utils.get_permclass_multiplicity` and `utils._perm_classes`,
# by verifying:
# - That the permutation classes form a partition of tensor indices: the sum of their class sizes equals the total number of independent components in a symmetric tensor, which is given by
#   $$\binom{d + r - 1}{r}\,.$$
#   This is a well-known expression; it can be found for example [here](http://www.physics.mcgill.ca/~yangob/symmetric%20tensor.pdf).
# - The identity
#   $$\sum_\hat{σ} s_\hat{σ} = d^r\,,$$
#   where $\hat{σ}$ is a permutation class and $s_{\hat{σ}}$ the size of that class.
#

# %%
def test_permclass_size():
    for r in range(9):
        for d in [0, 2, 10, 400]:
            assert (sum(utils.get_permclass_size(σcls, d)
                        for σcls in utils._perm_classes(r))
                    == math.prod(range(d, d+r)) / math.factorial(r))

            assert (sum(utils.get_permclass_size(σcls, d) * utils.get_permclass_multiplicity(σcls)
                        for σcls in utils._perm_classes(r))
                    == d**r)
