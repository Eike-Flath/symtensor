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

# Extended array functions

## Unary operations

```{eval-rst}
.. autofunction:: symtensor.utils.symmetrize
```
## Array functions

### Functions supported from the `numpy` namespace

::::{grid}
:gutter: 3

:::{grid-item-card} Description
**`numpy.ndim`**

**`numpy.shape`**
:::

:::{grid-item-card} Creation
**`numpy.asarray`**

**`numpy.asanyarray`**

**`numpy.empty`**
:::

:::{grid-item-card} Type promotion
**`numpy.result_type`**
:::

:::{grid-item-card} Comparison
**`numpy.isclose`**

**`numpy.all`**

**`numpy.any`**

**`numpy.array_equal`**

**`numpy.allclose`**
:::

::::

### Extended functionality

#### Creation

```{eval-rst}
.. autofunction:: symtensor.utils.empty_array_like
```

#### Type promotion

```{eval-rst}
.. autofunction:: symtensor.symalg.result_array
```
