from .permcls_symmetric_tensor import PermClsSymmetricTensor
try:
    from .torch_symmetric_tensor import TorchSymmetricTensor
except ModuleNotFoundError:
    pass
    
# Set a default symmetric tensor: TorchSymmetricTensor if we can
# (i.e. 'torch' module is available), otherwise a Numpy-based SymmetricTensor
if "TorchSymmetricTensor" in locals():
    # SymmetricTensor = TorchSymmetricTensor  # Temporarily commented out until implementation is finished
    SymmetricTensor = PermClsSymmetricTensor  # Remove once TorchSymmetricTensor is fully implemented
else:
    SymmetricTensor = PermClsSymmetricTensor
