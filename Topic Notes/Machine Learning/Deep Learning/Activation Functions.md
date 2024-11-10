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
The **ReLU** function $\text{ReLU} : \mathbb{R} \to [0,\infty]$ is one of the most widely used activation functions. It outputs the input directly if it's positive; otherwise, it outputs zero. **ReLU** is simple and computationally efficient, making it a go-to activation function for many deep learning architectures. Commonly used in hidden layers of deep neural networks, including convolutional neural networks (CNNs) and fully connected networks.

1. **Not bounded**: ReLU can produce very large output values
2. **Sparsity**: ReLU introduces sparsity (many neurons can be inactive at once, outputting 0), which can help with regularization
3. **Avoids vanishing gradient**: Unlike sigmoid and tanh, ReLU does not saturate for positive values, which helps with faster learning and gradient propagation
4. **Dying ReLU Problem**: If a neuron gets stuck in the negative input region, it will always output zero, and thus its gradient will be zero, which can prevent the neuron from learning
## Leaky Rectified Linear Unit

## Parametric ReLU

## Exponential Linear Unit

## Softmax

The standard softmax function $\sigma: \mathbb{R}^K \to (0,1)^K$, where $K \geq 1$ , takes a vector $z = (z_1, \dots z_k) ∈ \mathbb{R}^K$  and computes each component of vector $\sigma(z) ∈ (0,1)^K$ with 
$$ \sigma(z)_i = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
1. Softmax is used primarily in the output layer for multi-class classification problems
2. It converts a vector of raw scores (logits) into probabilities by taking the exponent of each score and normalizing it
3. It ensures that the sum of all output probabilities is 1, making it ideal for classification tasks
## Swish
$$ \text{swish}_\beta(x) = x\;\text{sigmoid}(𝛽 x) = \frac{x}{1 + e^{-𝛽 x}}$$
The swish family of functions was designed to smoothly interpolate between a linear function and the ReLU function. Often defined $𝛽 = 1$, the swish function is smooth and non-monotonic, which can help the model with gradient flow and reduce the likelihood of vanishing gradients. It tends to outperform ReLU in some deep network architectures.
## Summary
1. **Sigmoid**: Good for binary classification, but suffers from vanishing gradients
2. **tanh**: Zero-centered and often better than sigmoid, but still suffers from vanishing gradients
3. **ReLU**: Most popular due to efficiency, though it suffers form the "dying ReLU" problem
4. **Leaky ReLU**: Addresses dying ReLU problem by allowing small negative slope
5. **PReLU**: Allows the negative slope to be learning, potentially improving performance
6. **ELU**: Smooth activation function that avoids some of ReLU's drawbacks, but it is computationally expensive
7. **Softmax**: Used for multi-class classification tasks
8. **Swish**: A newer (2016) activation function that is smooth and can outperform ReLU in certain cases