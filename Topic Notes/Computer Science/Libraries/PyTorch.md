## Tensors

Tenors are a specialized data structure that are very similar to arrays and matrices. In PyTorch, we use tensors to encode the inputs and outputs of a model, as well as the model's parameters. Tensors are similar to [[NumPy#ndarray]], except that tensors can run on GPUs or other specialized hardware to accelerate computing. Tensors can be initialized in various ways, either *directly from data*,  *from a NumPy array*, or *from another tensor*.

### Operations
1. Transposing
2. Indexing
3. Slicing
4. Mathematical Operations
5. Linear Algebra
6. Random Sampling
7. etc.

Operations with a `_` suffix are in-place. For example, `x.copy_(y)` will change `x`