In deep learning, activation functions are crucial components that introduce non-linearity into the model, allowing neural networks to learn complex patterns in the data. Without activation functions, the neural network would be equivalent to a linear regression model, unable to capture complex relationships.
## Sigmoid
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
 The sigmoid function $\sigma : \mathbb{R} \to [0,1]$ is useful for binary classification tasks (e.g., predicting probabilities). Because the gradient is smooth, [[Gradient Descent]] is easy to implement with this activation function. The sigmoid function saturates for very large or very small input values, causing gradients to be near zero or slowing down learning.

## Hyperbolic Tangent
$$
\text{tanh}(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 2\sigma(2x) - 1
$$
The **tanh** function $\text{tanh} : \mathbb{R} \to [-1,1]$ is similar to the sigmoid function, but it maps the input to a range between -1 and 1. It has the advantage of being zero-centered, meaning the output can be both positive and negative, which helps with the learning process (especially when dealing with weight updates). Like sigmoid, it suffers from the vanishing gradients problem, especially with very large input values. The **tanh** function is commonly used in the hidden layers of neural networks, especially in [[Recurrent Neural Networks]] (RNNs).

## Rectified Linear Unit
$$\text{ReLU}(x) = \text{max}(0,x)$$
The **ReLU** function $\text{ReLU} : \mathbb{R} \to [0,\infty]$ is one of the most widely used activation functions. It outputs the input directly if it's positive; otherwise, it outputs zero. **ReLU** is simple and computationally efficient, making it a go-to activation function for many deep learning architectures. 

1. 
It is *not bounded*, meaning it can produce very large output values. **ReLU** introduces *sparsity* (many neurons can be inactive at once, outputting 0), which can help with regularization.


 
