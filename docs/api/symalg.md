---
jupytext:
  formats: md:myst
  notebook_metadata_filter: -jupytext.text_representation.jupytext_version
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
kernelspec:
  display_name: Python (symtensor)
  language: python
  name: symtensor
---

# Symmetric algebra functions

```{eval-rst}
.. automodule:: symtensor.symalg
```

## Binary operations

```{code-cell} ipython3
:tags: [remove-cell]

from myst_nb import glue
from symtensor import SymmetricTensor, symalg
def get_methods(fun):
   return ", ".join(f"{fun.__name__}.{k}"
                    for k in ("outer", "at", "reduce", "reduceat")
                    if symalg.add in SymmetricTensor._HANDLED_UFUNCS[k])
glue("add-methods", get_methods(symalg.add), display=False)
glue("subtract-methods", get_methods(symalg.subtract), display=False)
glue("multiply-methods", get_methods(symalg.multiply), display=False)
```

```{eval-rst}
.. autofunction:: add
```
Also supported: {glue:text}`add-methods`.

```{eval-rst}
.. autofunction:: subtract
```
Also supported: {glue:text}`subtract-methods`.

```{eval-rst}
.. autofunction:: multiply
```
Also supported: {glue:text}`multiply-methods`.

## Tensor contractions

```{eval-rst}
.. autofunction:: symtensor.symalg.tensordot
.. autofunction:: symtensor.symalg.contract_all_indices_with_matrix
.. autofunction:: symtensor.symalg.contract_all_indices_with_vector
.. autofunction:: symtensor.symalg.contract_tensor_list
```

