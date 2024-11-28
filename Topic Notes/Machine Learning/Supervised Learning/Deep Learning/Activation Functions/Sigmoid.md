Formula: 
$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$
 The sigmoid function $\sigma : \mathbb{R} \to [0,1]$ is useful for binary classification tasks (e.g., predicting probabilities). Because the gradient is smooth, [[Gradient Descent]] is easy to implement with this activation function. The sigmoid function saturates for very large or very small input values, causing gradients to be near zero or slowing down learning.