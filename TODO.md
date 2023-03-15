### Fix symalg

- [ ] Extend (np.add, np.sub, np.mul) to non-symmetric args, so they aren't completely equivalent functions to (symalg.add, symalg.sub, symalg.mul)


### Package migration & cleaning

- [ ] Replace any use of `exenv` by jupytext tags
- [ ] Replace any use of `__main__` by jupytext tags
- [ ] Figure out what `get_idx` does, update docstring and remove statGLOW dep
- [ ] Remove mackelab_toolbox dependencies (include code here if necessary)
  + [ ] TimeThis (testing / profiling only)
  + [ ] total_size_handler
  + [ ] GitSHA
- [ ] Find how to specify dev / testing requirements
- [ ] Remove Pydantic dependency? We could include the type coercion in the `Data.encode` methods (defined in base.py and torch_symtensor.py)
- [ ] Implement `symmetric_outer` and `symmetric_tensordot` in PermClsSymmetricTensor  
  (Currently they use the non-optimized generic versions.)
- [ ] [PermClsSymmetricTensor] Deserialization of stringified σcls tuples is currently in two places: `Data.decode` and `_validate_data` (dict branch)
  Should be consolidated into one place.
- [ ] Exclude the decomp_symmtensor from initial version, since it is still prototype.

### CPU to GPU:
We want to move heavy calculations from CPU to GPU using `pytorch`.


To do this we must:
  - [ ] Ensure that data is stored on GPU:
    I already added a property `SymmetricTensor.device` which should automatically give us the right pytorch device (CPU or GPU)
    - [ ] if `SymmetricTensor` is initialized with data dictionary, ensure that the data is stored as `torch.Tensor` on the right device
    - [ ] if `__setitem__()` is called, ensure that the data are stored on the right device **(?)**
    - [X] Rewrite `__getitem__` for pytorch **(?)**
    - [X] Rewrite `indep_iter` for pytorch
  - [ ] Ensure data manipulations are done on GPU:
     - [X] Rewrite `__array_ufunc_` for torch functions
     - [X] Rewrite `__array_function_` for torch functions **if necessary?**
          AR doesn't think `__array_function__` needs to be rewritten, but the four associated one-line functions decorated with `SymmetricTensor.implements` we might as well adapt for Pytorch. `asarray` and `tensordot` should be trivial, since PyTorch supports them. The other two functions we can explicitely disallow (by commenting them out basically), and if we really need them later we can add them at that time.
     - [X] Rewrite `tensordot` for pytorch
     - [X] Rewrite `outer_product` for pytorch
     - [X] Rewrite `contract_all_indices_with_matrix` for pytorch in Schatz paper fig 3 way
     - [X] Rewrite `contract_tensor_list` for pytorch
     - [X] Rewrite `contract_all_indices_with_vector` for pytorch


We could do this while preserving the functions which use numpy or do everything new in torch.
So either:
```
def some_func(self, ...):
    if self.use_numpy:
       # some numpy code
    else:
       # some pytorch code
```

or directly do
```
def some_func(self, ...):
    # some pytorch code
```

What do you think?

AR's suggestion:
- One abstract class `SymmetricTensor`
- Multiple subclasses for each implementation; at present, `NumpySymmetricTensor` and `TorchSymmetricTensor`.
- Each subclass in its own module
- All modules under the subpackage `symmetric_tensor` (which would replace this file)
- (Optional) Have an `__init__` file in the subpackage with something like the following code:
  ```python
  try:
      import torch
  except ImportError:
      from .numpy_symmetric_tensor import NumpySymmetricTensor as SymmetricTensor
  else:
      from .torch_symmetric_tensor import TorchSymmetricTensor as SymmetricTensor
      del torch
   ```
   This would ensure that already existing code should continue to work, and will switch to the Torch implementation if possible.
