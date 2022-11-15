# Symtensor: for really big symmetric tensors

[![pipeline status](https://jugit.fz-juelich.de/explainable-ai/symtensor/badges/develop/pipeline.svg)](https://jugit.fz-juelich.de/explainable-ai/symtensor/-/commits/develop)  [![coverage report](https://jugit.fz-juelich.de/explainable-ai/symtensor/badges/develop/coverage.svg)](https://jugit.fz-juelich.de/explainable-ai/symtensor/-/commits/develop)

## Rationale

A symmetric tensor $A$ of rank $r$ (an $r$-dimensional array in NumPy parlance) is one for which

$$A_{i_1,i_2,i_3,\dotsc,i_r} = A_{i_2,i_3,i_1,\dotsc} = A_{i_2,i_1,i_r,\dotsc,i_3} = \dotsb$$

These arise in a variety of fields within chemistry and physics, where the symmetry is prescribed by the physics. The physics often also prescribe that the tensor be high dimensional and/or high rank, which can quickly exceed memory capacity if the tensor is stored in a normal NumPy array. Thankfully, symmetry also presents a solution to this problem: since many of the tensor components are redundant, they can be stored in a compressed format. However, it is not sufficient to simply store these tensors: we also need to use them. Therefore the interest in wrapping their compressed storage with a NumPy-compatible API, exposing standard slicing and arithmetic semantics, allowing mixture and replacement with standard NumPy arrays, and completely hiding the compressed memory layout.[^numpy-API] Crucially, whenever possible, *operations are performed directly on the underlying compressed memory*, which in some case can lead order-of-magnitude gains in efficiency – or even make the impossible, possible.[^like-sparse]

This package evolved out of a desire to apply field-theory methods to neural networks, which required tensors of both high rank and very high dimension. It addresses two complementary needs:

1) Provide ready-to-use implementations of symmetric tensors, which behave like standard arrays wherever possible.
2) Make it easy to extend support for:
    - additional functions;
    - *additional array backends* (currently NumPy and PyTorch are suported)
    - *new compression formats*.

## Modular, extensible design

Just like with sparse arrays, different applications may be best served by different compression formats. This is possible by subclassing the base *SymmetricTensor* class and implementing format-specific methods. The resulting class can be made completely interoperable with NumPy arrays and other SymmetricTensor classes.

In addition to compression formats, one may also want to use different backends than NumPy for the data arrays, such as PyTorch or JaX. This is done via *mixin* classes. By combining mixins with SymmetricTensor subclasses, a wide array of tensor formats can be supported with minimal repeated code.

We use a [standardized API suite]() of unit tests: most tests are written generically enough to be applied to each format/backend implementation *without any modification*.[^how-to-write-tests] This ensures a consistent API throughout the library, and massively simplifies the writing of tests for new implementations.

## Currently supported formats

| Format →<br>Backend ↓              | [Dense](sources/dense_symtensor.py) | [Permutation classes](sources/permcls_symtensor.py) | Blocked arrays |
| ---------------------------------- | ----------------------------------- | --------------------------------------------------- | -------------- |
| Numpy                              | ✔                                   | ✔                                                   | ✘              |
| [PyTorch](sources/torch_symtensor) | ✔                                   | ✔                                                   | ✘              |
| tlash[^tlash]                      | ✘                                   | ✘                                                   | (planned)      |


[^numpy-API]: We implement both the [universal function](https://numpy.org/neps/nep-0013-ufunc-overrides.html) and [array function](https://numpy.org/neps/nep-0018-array-function-protocol.html#nep18) protocols. These are also called [dispatch mechanisms](https://numpy.org/doc/stable/user/basics.dispatch.html).
[^like-sparse]: These requirements are very similar to those for sparse arrays, and this package makes extensive use of NumPy improvements introduced to address limitations of *scipy.sparse*. In fact our base class could serve as a basis for a new implementation of sparse arrays, if one were the type inclined to reinvent the wheel.
[^how-to-write-tests]: The general procedure for building a test suite for a new implementation:
  - Copy one of the existing tests under [*tests*](./symtensor/tests).
  - Open the file as a notebook via Jupytext. This provides a lot of inlined documentation, and organizational sections to aid navigation.
  - Change the few lines at the top which specify the subclass of *SymmetricTensor* to test.
  - Add/deactivate/modify tests as needed for the particulars of your format and backend. Typically, out of the 25-odd tests, 3-4 need to be specialized.
[^tlash]: [GitHub](https://github.com/mdschatz/tlash)
