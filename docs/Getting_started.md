---
jupytext:
  formats: md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (statGLOW)
  language: python
  name: statglow
---

# Getting started

```{code-cell} ipython3
:tags: [remove-input]

from __future__ import annotations
import statGLOW.typing
```

## Rationale

If we naively store tensors as high-dimensional arrays, they quickly grow quite large. However, reality does not need to be so bad:

1.  Since tensors must be symmetric, a fully specified tensor contains a lot of redundant information.

We want to avoid storing each element of the tensor, while still being efficient with those elements we do store.

## Intended usage

### Situations where a `SymmetricTensor` may be well suited

In general, with a `SymmetricTensor` one gains a much reduced memory footprint for large tensors. Efficiency of operations strongly depends on whether they can be written in terms of fast, memory-aligned operations exploiting the memory layout.

**Construction of symmetric tensors.**  


**Storing symmetric tensors in a memory-efficient manner.**  


**Computing combinatorics**  
Number of equivalent permutations.

Number of independent components within a *permutation class*.

**Iterating over the entire tensor, when the order does not matter**  


**Random access of a specific entry.**  
Not all memory layouts allow efficient computation of memory position from a dense index, although they all benefit from the reduce memory size compared to dense storage.

**Broadcasted operations with scalars operands.**  


**Contractions with arrays or symmetric tensors**  
Because of the compressed memory layout, dot products or contractions require looping over indices in non-trivial order. Blocked layouts (as proposed by Schatz et al (2014)), or a fully lexicographic layout, should allow for much faster memory-aligned operations at least in some cases.

### Situations for which a `SymmetricTensor` may be less well suited:

**Slicing a tensor as a normal array (per axis).**  
In some cases (such as lexicographic storage), efficient slicing along one dimension might be possible.

**Broadcasted operations on the whole tensor with non-scalar operands.**  
While it might possible to get this to work, it likely would not be much faster than just looping.

Here again, an efficient solution might be possible with lexicographic storage. Whether it’s worth our time implementing though is another question.

### Supported array operations.

See the [implementation page](./sources/base.py#Implementation-of-the-__array_function__-protocol).

## Notation

**Multi-index**  
~ A particular element of a tensor $A$ of rank $r$ is denoted $A_{i_1,\dotsc,i_r}$. For notational brevity, we use $I$ to denote the multi-index, i.e.
  
  $$I := i_1,\dotsc,i_r\,.$$
  
  When the rank of the multi-index is important, we may use $I_r$.

**Index class**  
~ The basic property of a symmetric tensor $A$ of rank $r$ is that for any permutation $σ \in S_r$,
  
  $$A_{I} = A_{σ(I)} \,.$$
  
  Thus the existence of a permutation such that $I' = σ(I)$ defines an equivalence relation on the set of multi-indices. An *index class* is the set of all indices satisfying this relation, and is denoted by with a representative element of that classes:
  
  $$\hat{I} := \{I' \in \mathbb{R}^r | \exists σ \in S_r, I' = σ(I)\} \,.$$

~ The representative depends on the data format. For dense tensors, we sort indices lexicographically. For σ-class tensors, we groupe equal indices, sorting groups first in decreasing multiplicity, and then in increasing index value. In both formats, the first of these sorted indices is the representative index.

**Permutation class (aka σ-class)**  
~ A *permutation class* is a set of index classes which have the same index repetition pattern; for example, $\widehat{1110}$ and $\widehat{2220}$ both have four indices, with one repeated three times. We can represent these classes in two different ways:

  - A string: `'iiij'`;
  - A tuple of repetitions: `(3, 1)`

~ The first notation is used for public-facing functions. The second is used internally and is described in more detail [below](#Indexing-utilities).  
  There are a few advantages to grouping permutation classes:

  - The number of permutation classes is independent of rank (if we allow for empty classes).
  - Index classes in the same permutation class all have the same size.

~ We also define

  - The **multiplicity** of a permutation class as the size of each of its index classes.
  - The **size** of a permutation class as the number of index classes it contains.

### Notation summary

| Symbol        | Desc                                                   | Examples                             |
|:--------------|:-------------------------------------------------------|:-------------------------------------|
| $d$           | dimension                                              | 2                                    |
| $r$           | rank                                                   | 4                                    |
| $I$           | Multi-index                                            | $1010$<br>$1011$                     |
| $\hat{I}$     | Index class                                            | $\widehat{0011}$<br>$\widehat{1110}$ |
| $\hat{σ}$     | Permutation class                                      | `iijj`<br>`iiij`                     |
| $γ_{\hat{σ}}$ | multiplicity                                           | 6<br>4                               |
| $s_{\hat{σ}}$ | size                                                   | 1<br>2                               |
| $l$           | Given $\hat{σ}$, number of different indices           | 2<br>2                               |
| $m_n$         | Given $\hat{σ}$, number of indices repeated $n$ times  | See below                            |
| $n_k$         | Given $\hat{σ}$, number of times index $k$ is repeated | See below                            |

More examples:

| $\hat{I}$ | $\hat{I}$ (str) | $l$ | $m$           | $n$     |
|:----------|:----------------|:----|:--------------|:--------|
| `(3,2)`   | `iiijj`         | 2   | 0, 1, 1, 0, 0 | 3, 2    |
| `(1,1,1)` | `ijk`           | 3   | 3, 0, 0       | 1, 1, 1 |

### Identities

-   $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} γ_{\hat{σ}} = d^r$
-   $\displaystyle \sum_{\hat{σ}} s_{\hat{σ}} = \binom{d + r - 1}{r}$
-   $\displaystyle s_{\hat{σ}} = \frac{d(d-1)\dotsb(d-l+1)}{m_1!m_2!\dotsb m_r!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $m_n$ is the number of different indices which appear $n$ times.
-   $\displaystyle γ_{\hat{σ}} = \binom{r}{m_1,m_2,\dotsc,m_r} = \frac{r!}{n_1!n_2!\dotsb n_l!}$, where $l$ is the number of different indices in the permutation class $\hat{σ}$ and $n_k$ is the number of times index $\hat{I}_k$ appears.

## Usage

:::{caution}  
Symmetric tensors with [different memory layouts](./symmetric_formats.md) are provided. For the examples below we use the [dense](sources/dense_symtensor) layout; this is in some sense the reference format: since it simply wraps a NumPy `ndarray` with the `SymmetricTensor` API, support for NumPy functionality is often simple. However it also provides no data compression at all, so in practice one of the compressed formats are generally better suited.  
:::

```{code-cell} ipython3
from statGLOW.stats.symtensor import SymmetricTensor, DenseSymmetricTensor
```

```{code-cell} ipython3
import pytest
from statGLOW.stats.symtensor import SymmetricTensor, DenseSymmetricTensor

import holoviews as hv
hv.extension("bokeh")
```

The base class `SymmetricTensor` is an [*abstract base class*](https://docs.python.org/3/library/abc.html): it prescribes a certain set of methods, but since it does not specify a memory layout, some methods are left unimplemented. It therefore cannot be instantiated directly:

```{code-cell} ipython3
with pytest.raises(TypeError):
    SymmetricTensor(rank=3, dim=3)
```

Subclasses define a memory layout, and provide implementations for the methods which are layout-specific.

- `DenseSymmetricTensor` simply uses a dense array to store values; this can be useful for defining the API and testing, but obviously offers little benefit compared to a normal array.
- `LexSymmetricTensor` (Not implemented): Provides optimal storage for symmetric tensors with no additional symmetries (each index permutation is stored exactly once).
- `BlockedSymmetricTensor` (Not implemented; name TBD): This is based on the work of Schatz et al (2014). A trade-off is made between computation and memory efficiency: smaller block sizes approach the efficiency of lexicographic storage.
- `PermClsSymmetricTensor` stores data as a dictionary of 1-D arrays, one array per permutation class. With the ability to represent entire permutation classes with a scalar, there is potential for higher compression than lexicographic storage.
    For obvious reasons, this format is the most efficient for operating on permutation classes (e.g. setting all element values for a particular class).

```{code-cell} ipython3
A = DenseSymmetricTensor(rank=3, dim=3)
if exenv in {"notebook", "jbook"}: display(A)
```

Values can be assigned to an entire permutation class at a time.
Note the use of `get_permclass_size` to avoid having to compute how many independent `'iijj'` terms there are.

```{code-cell} ipython3
A['iiii'] = 1
A['iijj'] = np.arange(utils.get_permclass_size('iijj', A.dim))
if exenv in {"notebook", "jbook"}: display(A)
```

The `indep_iter` and `indep_iter_index` methods can be used to obtain a list of values where *each independent component appears exactly once*.

```{code-cell} ipython3
hv.Table(zip((str(idx) for idx in A.indep_iter_index()),
             A.indep_iter()),
         kdims=["broadcastable index"], vdims=["value"])
```

Conversely, the `flat` method will return as many times as it appears in the full tensor, as though it was called on a dense representation of that tensor (although with possibly different ordering, depending on the memory layout).

`flat_index` returns the index associated to each value.

```{code-cell} ipython3
hv.Table(zip(A.flat_index, A.flat), kdims=["index"], vdims=["value"])
```

The number of independent components can be retrieved with the `indep_size` attribute.

```{code-cell} ipython3
A.indep_size
```

Note that this may be more or less than the number of actually store memory elements: certain memory layouts allow a scalar to represent multiple components, and others (like `DenseSymmetricTensor` and `BlockedSymmetricTensor`) store redundant data.
To get the number of *allocated* memory elements, use the `size` attribute.

```{code-cell} ipython3
A.size
```

The size of the full non-symmetric tensor is most simply calculated as `dim**rank`, which is equivalent to multiplying each permutation class $\hat{σ}$ size by its multiplicity $γ_{\hat{σ}}$ (the number of times components of that class are repeated due to symmetry).

```{code-cell} ipython3
(math.prod(A.shape)
 == A.dim**A.rank
 == sum(utils.get_permclass_size(σcls, A.dim)*utils.get_permclass_multiplicity(σcls)
        for σcls in A.perm_classes)
 == 27)
```

Like the sparse arrays of *scipy.sparse*, a `SymmetricTensor` has a `.todense()` method which returns the equivalent dense NumPy array.
For the base class this is a no-op, since data are already stored as a dense array.

```{code-cell} ipython3
Adense = A.todense()
```

```{code-cell} ipython3
l1 = str(Adense[:1,:2])[:-1]
l2 = str(Adense[1:2,:2])[1:]
print(l1[:-1] + "\n\n   ...\n\n  " + l1[-1] + "\n\n"
      + " " + l2[:-2] + "\n\n   ...\n\n  " + l2[-2] + "\n\n...\n\n" + l2[-1])
```

To instantiate a tensor without allocating any memory for data, pass `data=None`. This can be useful to access the methods for combinatorics, or compute the size that would be required to store a tensor in this format (perhaps to check if this is not too large).

```{code-cell} ipython3
B = DenseSymmetricTensor(rank=6, dim=1000, data=None)  # 10¹⁸ elements — would exceed memory if allocated
print("# indep elements      :", B.size)
print("\n".join(textwrap.wrap(f"σ classes             : {B.perm_classes}",
                              subsequent_indent=" "*23)))
print("size of class 'iiijjk':", utils.get_permclass_size('iiijjk', B.dim))
```

Symmetric tensors main contain bools instead of numbers. This is useful to store the result of comparisons:

```{code-cell} ipython3
A = DenseSymmetricTensor(rank=3, dim=3)
B = DenseSymmetricTensor(rank=3, dim=3)
A['iii'] = 1
B['iii'] = 1; B['ijj'] = -1

display(A == B)
```
