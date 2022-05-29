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
# # Example usage

# %%
import statGLOW.typing
from statGLOW.stats.symtensor import DenseSymmetricTensor

import numpy as np

s = DenseSymmetricTensor(rank = 4, dim = 5)
#alle diagonaleinträge setzen
s['iiii'] = 1.0
#alle einträge mit allen indices unterschiedlich setzen

s['ijkl'] = [np.random.rand() for index in s.index_class_iter(class_label='ijkl')]
s['ijjj'] = [np.random.rand() for index in s.index_class_iter(class_label='ijjj')]

#zeige alle permutationsklassen:
print([permcls for permcls in s.perm_classes])

#mache numpy tensor daraus

s_dense = s.todense()

print(s_dense)

# %%
