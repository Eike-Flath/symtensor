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
# # Use-case timings
#
# Various timings to help get a sense of computation times for different use cases.

# %% [markdown]
# ::: {note}  
# For comparitive micro-timings, which informed some design choices, see the corresponding page in the [developer docs](developers/comparitive_timings).  
# :::

# %%
from mackelab_toolbox.utils import timeit
import itertools
import functools
import math
from tabulate import tabulate
import numpy as np
import pint; ureg = pint.UnitRegistry()

import statGLOW.typing
from statGLOW.stats.symtensor import DenseSymmetricTensor

ureg.define('μs = 1 * microsecond = μs')  # timeit uses 'μs' in its output strings
# ureg.define("@alias microsecond = μs")  # This would be better, but doesn't work – https://github.com/hgrecco/pint/issues/1076

# %% [markdown]
# ## Iterating over independent components of a tensor
#
# This is the time it takes just to iterate through independent tensor components.
# The loop is performed with a single call to the C function `itertools.combinations_with_replacement`, so this is very close to single thread optimal speed.
#
# Some rank/dim combinations are skipped because they can take *a lot* of time.

# %%
cwr = itertools.combinations_with_replacement
res_data = []
for rank, dim in itertools.product([2, 4, 6, 8], [5, 10, 15, 20, 25, 30, 100, 1000]):
    size = math.comb(dim+rank-1, rank)
    if size >= 1e8:  # 1e10 is the threshold where counting the list takes noticeable time
        res_data.append([rank, dim, size, None, None])
    else:
        dim_idcs = range(dim)
        L = sum(1 for _ in cwr(dim_idcs, rank))
        res = timeit("cwr(dim_idcs, rank)", number=5)
        res_data.append([rank, dim, size,
                         (res.min*ureg(res.unit)).to('μs').magnitude,
                         (res.min*ureg(res.unit)).to('ps').magnitude/L])

# %%
print(tabulate(res_data, headers=["rank", "dim", "size", "time (μs)", "time / index (ps)"],
               floatfmt=".3f"))

# %% [markdown]
#       rank    dim                  size    time (μs)    time / index (ps)
#     ------  -----  --------------------  -----------  -------------------
#          2      5                    15        0.366            24400.021
#          2     10                    55        0.414             7523.644
#          2     15                   120        0.459             3823.334
#          2     20                   210        0.515             2452.379
#          2     25                   325        0.551             1694.154
#          2     30                   465        0.583             1254.624
#          2    100                  5050        1.180              233.624
#          2   1000                500500       18.565               37.093
#          4      5                    70        0.379             5411.431
#          4     10                   715        0.418              585.174
#          4     15                  3060        0.450              146.993
#          4     20                  8855        0.511               57.708
#          4     25                 20475        0.560               27.360
#          4     30                 40920        1.136               27.766
#          4    100               4421275        1.223                0.277
#          4   1000           41917125250
#          6      5                   210        0.368             1752.383
#          6     10                  5005        0.413               82.597
#          6     15                 38760        0.459               11.852
#          6     20                177100        0.518                2.926
#          6     25                593775        0.566                0.953
#          6     30               1623160        0.585                0.361
#          6    100            1609344100
#          6   1000      1409840590658500
#          8      5                   495        0.373              753.536
#          8     10                 24310        0.425               17.466
#          8     15                319770        0.455                1.424
#          8     20               2220075        0.503                0.227
#          8     25              10518300        0.556                0.053
#          8     30              38608020        0.594                0.015
#          8    100          325949656825
#          8   1000  25504066636461931375

# %% [markdown]
# ### Slicing
#
# **[TODO]** Make a table as above.

# %%
for rank in [3]:
    for dim in [50]:
        A = DenseSymmetricTensor(rank=rank, dim=dim)
        display(timeit("A[0]", number=1000))


# %% [markdown]
# ### Outer product:
#
# **[TODO]** Make a table as above.

# %%
for rank in [3]:
    for dim in [50]:
        vect = DenseSymmetricTensor(rank=1, dim=dim)
        vect['i'] = np.random.rand(dim)
        print(f"rank: {rank}, dim: {dim}")

        # # pos dict creation
        # # %timeit x = pos_dict[rank,dim]

        # vect x vect x vect ... x vect
        display(timeit("functools.reduce(np.multiply.outer, (vect,)*rank)",
                       number=10))
