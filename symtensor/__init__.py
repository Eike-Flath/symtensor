from .base import SymmetricTensor
from .dense_symtensor import DenseSymmetricTensor
from .permcls_symtensor import PermClsSymmetricTensor
# try:
#     from .torch_symmetric_tensor import TorchSymmetricTensor
# except ModuleNotFoundError:
#     pass

from .symalg import *

# # Set a default symmetric tensor: TorchSymmetricTensor if we can
# # (i.e. 'torch' module is available), otherwise a Numpy-based SymmetricTensor
# if "TorchSymmetricTensor" in locals():
#     # SymmetricTensor = TorchSymmetricTensor  # Temporarily commented out until implementation is finished
#     SymmetricTensor = PermClsSymmetricTensor  # Remove once TorchSymmetricTensor is fully implemented
# else:
#     SymmetricTensor = PermClsSymmetricTensor
