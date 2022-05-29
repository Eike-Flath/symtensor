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
# # Torch mixin for `SymmetricTensor`

# %%
import numpy as np   # To avoid MKL bugs, always import NumPy before Torch
import torch

from typing import Optional, Union
from smttask_ml.scityping import Number, Array, DType

class TorchSymmetricTensorMixin:
    
    def __init__(self, rank: Optional[int]=None, dim: Optional[int]=None,
                 data: Union[Array, Number]=np.float64(0),
                 dtype: Union[str,DType]=None,
                 symmetrize: bool=False,
                 device: Union[str, 'torch.device']="cpu"):
        """
        {{base_docstring}}
        
        Torch-specific parameter
        ------------------------
        device: A string identifying a PyTorch device, either 'cpu' or 'gpu'.
           The default value 'None' will prevent the `torch` library from being
           loaded at all; thus it will work even when PyTorch is not installed.
           PyTorch device objects are also supported.

        """
        # Set device
        self._device_name = str(device).lower()
        if not self._device_name in {"cpu", "gpu"}:
            raise ValueError("`device` should be either 'cpu' or 'gpu'; "
                             f"received '{self._device_name}'.")
                             
        # Let super class do the initialization
        super().__init__(rank, dim, data, dtype, symmetrize)

    @property
    def device(self) -> "torch.device":
        return torch.device(self._device_name)


# %% [markdown]
# ## `DenseTorchSymmetricTensor`

# %%
class DenseTorchSymmetricTensor(TorchSymmetricTensorMixin, DenseSymmetricTensor):
    pass
