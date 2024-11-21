Convolutional layers mainly have three key properties:
1. *sparse connections*
2. *parameter sharing*
3. *equivariant representation*

## Sparse Connections

In traditional neural network layers, the interactions between the input units and the output units can be described by a matrix. Each element of this matrix defines an independent parameter for the interaction between each input unit and each output unit. However, the convolutional layers usually have sparse connections between layers when the kernel is only nonzero on a limited number of input units. In densely connected layers, a single output unit is affected by all of the input units, whereas in the convolutional  neural network layers, the output unit is only affected by input units within the scope of the kernel, called the *receptive field*.

One of the major advantages of the sparse connectivity is that it can largely improve the computational efficiency. If there are $N$ input units and $M$ output units, there $N\times M$ parameters in the traditional neural network layers. The time complexity for a single computation pass of this layer is $O(N \times M)$. The convolutional layers with the same number of input and output units only have $K \times M$ parameters, when its kernel size is $K$. Typically, the kernel size $K$ is much smaller than the number of input units $N$.

## Parameter Sharing

Parameter sharing means sharing the same set of parameters when performing the calculation for different output units.