# -*- coding: utf-8 -*-
# # Comparitive timings
#
# Comparitive timings of two or more implementations, supporting certain design choices.

from mackelab_toolbox.utils import timeit
import itertools

# ## Comparing permutation classes
#
# Comparisons with strings are overall slightly faster.

# +
counts = [(2,1,1,1,1,1,1),
          (4,4),
          (8,0)]
labels = ["iijklmno", "iiiijjjj", "iiiiiiii"]

c = counts[0]
l = labels[0]


# -

# ### Equality

# +
# %timeit (2,) == c                 # False
# %timeit (2,1,1,1) == c            # False
# %timeit (2,1,1,1,1,1,1) == c      # True
# %timeit (2,1,1,1,1,1,1,1,1) == c  # False

# %timeit "i" == l           # False
# %timeit "iijkl" == l       # False
# %timeit "iijklmno" == l    # True
# %timeit "iijklmnopq" == l  # False
# -

# ### Sub-permutation class

# +
# %timeit (2,) in c                 # False
# %timeit (2,1,1,1) in c            # False
# %timeit (2,1,1,1,1,1,1) in c      # True
# %timeit (2,1,1,1,1,1,1,1,1) in c  # False

# %timeit "i" in l           # False
# %timeit "iijkl" in l       # False
# %timeit "iijklmno" in l    # True
# %timeit "iijklmnopq" in l  # False
# -

# ## Inlining a function
#
# **Result**: Inlining a function saves about 28 ns / function call vs calling another function
#
# **Note**: Result applies for functions in the same local namespace. Overhead for things like `utils.my_func` should be slightly higher.

# +
def inlined(n):
    for _ in range(n):
        pass
def inner():
    pass
def not_inlined(n):
    for _ in range(n):
        inner()
        
# %timeit inlined(100)
# %timeit not_inlined(100)
# %timeit inlined(1000000)
# %timeit not_inlined(1000000)


# -

#     774 ns ± 4.31 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     3.54 µs ± 10.7 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     11.1 ms ± 395 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)
#     37.7 ms ± 397 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

print("n=100:     ", round((2.8 * 1000) / 100, 1), "ns / iteration")
print("n=1000000: ", round((26.6 * 1000000) / 1000000, 1), "ns / iteration")

# ## Iterating over *independent* elements of a dense tensor
# np.nditer + independent filter – vs – element-access with only indep rep indices
#
# **Result** Element-wise access is >100 times faster.

import numpy as np
from itertools import combinations_with_replacement
from more_itertools import consume

# Ensure both methods yield the same indices

# +
rank = 4; dim = 4

A = np.zeros((dim,)*rank)


comb = combinations_with_replacement(range(dim), rank)  
it = np.nditer(A, flags=['multi_index'], op_flags=[['readonly']])

l = []
while not it.finished:
    if (np.diff(it.multi_index) >= 0).all():
        l.append(it.multi_index)
    it.iternext()

assert list(comb) == l


# -

# Timings

# +
def with_comb(A, dim, rank):
    for idx in combinations_with_replacement(range(dim), rank):
        yield A[idx]

def with_nditer(A):
    it = np.nditer(A, flags=['multi_index'], op_flags=[['readonly']])
    while not it.finished:
        if (np.diff(it.multi_index) >= 0).all():
            yield it[0]
        it.iternext()

for rank in [2, 4, 8]:
    for dim in [10, 100]:
        if dim**rank > 10000000:  # 10^8
            continue
        A = np.zeros((dim,)*rank)
        # res = %timeit -o -q consume(with_comb(A, dim, rank))
        print(f"Unique element-wise access, {dim:>5}^{rank} tensor: {res}")
        # res = %timeit -o -q consume(with_nditer(A))
        print(f"Filtered nditer,            {dim:>5}^{rank} tensor: {res}")
# -

#     Unique element-wise access,    10^2 tensor: 5.95 µs ± 40.5 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
#     Filtered nditer,               10^2 tensor: 470 µs ± 5.35 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#     Unique element-wise access,   100^2 tensor: 461 µs ± 2.17 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
#     Filtered nditer,              100^2 tensor: 46.3 ms ± 153 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)
#     Unique element-wise access,    10^4 tensor: 82.3 µs ± 814 ns per loop (mean ± std. dev. of 7 runs, 10000 loops each)
#     Filtered nditer,               10^4 tensor: 46.6 ms ± 280 µs per loop (mean ± std. dev. of 7 runs, 10 loops each)

# ## 10x overhead of `singledispatch`
#
# We considered using functool's *singledispatchmethod* to allow methods to accept σ-classes in both counts and label formats, but that incurs a pretty high overhead if the code inside the function is short.
# Since it's conceivable for these functions to be used inside loops, the extra convenience doesn't seem worth it.

from functools import singledispatchmethod


class Foo:
    @singledispatchmethod
    def get_class_size(self, σcls):
        raise TypeError("Perm class identifier must be either tuple of "
                        f"repeats or string.\nReceived: {σcls}.")
    @get_class_size.register
    def _(self, σcls: str):
        pass
    @get_class_size.register
    def _(self, σcls: tuple):  # TODO: With 3.9, we can do tuple[int]
        pass
    
    def get_without_overload(self, σcls):
        if isinstance(σcls, str):
            return self.get_str(σcls)
        elif isinstance(σcls, tuple):
            return self.get_tup(σcls)
        else:
            raise TypeError("Perm class identifier must be either tuple of "
                            f"repeats or string.\nReceived: {σcls}.")
    def get_str(self, σcls):
        pass
    def get_tup(self, σcls):
        pass


# +
foo = Foo()
# %timeit foo.get_class_size("ii")
# %timeit foo.get_class_size((1,))

# %timeit foo.get_without_overload("ii")
# %timeit foo.get_without_overload((1,))
# -

#     1.51 µs ± 5.35 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     1.51 µs ± 7.19 ns per loop (mean ± std. dev. of 7 runs, 1000000 loops each)
#     102 ns ± 0.21 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)
#     135 ns ± 0.264 ns per loop (mean ± std. dev. of 7 runs, 10000000 loops each)

from mackelab_toolbox.utils import GitSHA    # Keep this cell below all others
GitSHA(show_hostname=True)
