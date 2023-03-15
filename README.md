# Symtensor: for really big symmetric tensors

[![pipeline status](https://jugit.fz-juelich.de/explainable-ai/symtensor/badges/develop/pipeline.svg)](https://jugit.fz-juelich.de/explainable-ai/symtensor/-/commits/develop)  [![coverage report](https://jugit.fz-juelich.de/explainable-ai/symtensor/badges/develop/coverage.svg)](https://jugit.fz-juelich.de/explainable-ai/symtensor/-/commits/develop)

## Rationale

A symmetric tensor $A$ of rank $r$ (an $r$-dimensional array in NumPy parlance) is one for which

$$A_{i_1,i_2,i_3,\dotsc,i_r} = A_{i_2,i_3,i_1,\dotsc} = A_{i_2,i_1,i_r,\dotsc,i_3} = \dotsb$$

These arise in a variety of fields within chemistry and physics, where the symmetry is prescribed by the physics. The physics often also prescribe that the tensor be high dimensional and/or high rank, which can quickly exceed memory capacity if the tensor is stored in a normal NumPy array. Thankfully, symmetry also presents a solution to this problem: since many of the tensor components are redundant, they can be stored in a compressed format. However, it is not sufficient to simply store these tensors: we also need to use them. Therefore the interest in wrapping their compressed storage with a NumPy-compatible API, exposing standard slicing and arithmetic semantics, allowing mixture and replacement with standard NumPy arrays, and completely hiding the compressed memory layout.[^numpy-API] Crucially, whenever possible, *operations are performed directly on the underlying compressed memory*, which in some case can lead order-of-magnitude gains in efficiency – or even make the impossible, possible.[^like-sparse]

This package evolved out of a desire to apply field-theory methods to neural networks, which required tensors of both high rank and very high dimension to represent polynomials. Since evaluating these polynomials implicitely symmetrizes all the tensors (for example, $A_{ij} x_i x_j + A_{ji} x_j x_i = \frac{A_{ij} + A_{ji}}{2} x_i x_j$), we can save a lot of storage and computation cost by working directly with *symmetric tensors* and *symmetrized operations*. 

This package therefore addresses three complementary needs:

1) Provide ready-to-use implementations of symmetric tensors, which behave like standard arrays wherever possible.
2) Add support for a selection of symmetrized linear algebra operations, which exploit the symmetric structure of tensors for sometimes massive computational gains.
   Generic fallback operations are provided for non-symmetric arrays.
2) Make it easy to extend support for:
    - additional functions;
    - *additional array backends* (currently NumPy and PyTorch are suported)
    - *new storage compressed storage formats*.

## Modular, extensible design

Just like with sparse arrays, different applications may be best served by different compressed storage formats. This is possible by subclassing the base *SymmetricTensor* class and implementing format-specific methods. The resulting class can be made completely interoperable with NumPy arrays and other SymmetricTensor classes.

In addition to compression formats, one may also want to use different backends than NumPy for the data arrays, such as PyTorch or JaX. This is done via *mixin* classes. By combining mixins with SymmetricTensor subclasses, a wide array of tensor formats can be supported with minimal repeated code.

### Symmetric algebra

The functions below define *symmetrized* versions of the equivalent NumPy function. In other words, if a NumPy function  can be written $f(A, B)$, where $A$, $B$ are $n$-d arrays, then the symmetrized version would be

$$\mathtt{Symmetrize}\left(f(A,B)\right) \,,$$

where $\mathtt{Symmetrize}$ is an average over all symmetric permutations of its argument.
A default implementation of $f$ would use some form of the definition above, which is generic and is well-defined also for non-symmetric tensors. However, since the number of permutations grows rapidly with the rank of the tensor, this quickly becomes computationally prohibitie. Therefore each storage format will normally provide an optimized version which exploits its compressed structure, and allows for application with much larger tensors. The use of optimized function implementations, and falling back to the default one using permutations, is handled transparently by the library.

Since symmetrized functions are closed over symmetric tensors, we can say that they form a *symmetric algebra*; they are correspondingly defined in [`symtensor.symalg`](symtensor/symalg.py). Note that you should always use the functions defined in this module: they will automatically dispatch to an optimized implementation when one is available.

Currently `symtensor.symalg` defines the following functions:

- `add`
   + `add`
   + `add.outer`
- `subtract`
   + `subtract`
   + `subtract.outer`
- `multiply`
   + `multiply`
   + `multiply.outer`
- `transpose` (optimized to a no-op on symmetric tensors)
- `tensordot`
- `contract_all_indices_with_matrix`
- `contract_all_indices_with_vector`
- `contract_tensor_list`

### Currently supported formats

| Format →<br>Backend ↓  | [Dense](symtensor/dense_symtensor.py) | [Permutation classes](symtensor/permcls_symtensor.py) | [Outer product decomposition](symtensor/decomp_symmtensor.py) | Blocked arrays |
| -----------------------| -------------| ---------------------------|--------------------------- | -------------- |
| Numpy                  | ✔            | ✔                          | (WIP)                      | ✘              |
| [PyTorch](symtensor/torch_symtensor)         | ✔            | ✔                          | prototype                  | ✘              |
| tlash[^tlash]          | ✘            | ✘                          | ✘                          | (planned)      |


### Standardized API

We use a [standardized API suite](symtensor/testing/api.py) of unit tests: most tests are written generically enough to be applied to each format/backend implementation *without any modification*. This ensures a consistent API throughout the library, and massively simplifies the writing of tests for new implementations. In brief, porting the test suite involves:
   1. Copying one of the existing tests under [*tests*](./symtensor/tests).
   2. Opening the file as a notebook via Jupytext. This provides a lot of inlined documentation, and organizational sections to aid navigation.
   3. Changing the few lines at the top which specify the subclass of *SymmetricTensor* to test.
   4. Adding/deactivating/modifying tests as needed for the particulars of your format and backend. Typically, out of about 25 tests, 3-4 need to be specialized.

[^numpy-API]: We implement both the [universal function](https://numpy.org/neps/nep-0013-ufunc-overrides.html) and [array function](https://numpy.org/neps/nep-0018-array-function-protocol.html#nep18) protocols. These are also called [dispatch mechanisms](https://numpy.org/doc/stable/user/basics.dispatch.html).
[^like-sparse]: These requirements are very similar to those for sparse arrays, and this package makes extensive use of NumPy improvements introduced to address limitations of *scipy.sparse*. In fact our base class could serve as a basis for a new implementation of sparse arrays, if one were the type inclined to reinvent the wheel.
[^tlash]: `tlash` is a C implementation of the blocked arrays format described by Schatz et al. (*Exploiting Symmetry in Tensors for High Performance: Multiplication with Symmetric Tensors*, 2014). [[GitHub]](https://github.com/mdschatz/tlash) [[doi:10.1137/130907215]](https://doi.org/doi:10.1137/130907215)
