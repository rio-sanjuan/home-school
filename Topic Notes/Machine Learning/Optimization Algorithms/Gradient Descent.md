Gradient Descent is a fundamental optimization algorithm used to minimize the loss function, and in turn, train models like neural networks. It's the primary method used for adjusting the parameters (weights and biases) of a model in order to improve its performance on a given task.

At a high level, gradient descent is an iterative optimization technique that seeks to minimize a function by gradually moving toward the steepest direction of descent (negative gradient) from a given point. In the context of deep learning, the function being minimized is typically the loss function (or cost function), which measures the difference between the predicted output of the model and the actual target output.

The goal is to find the set of parameters (weights and biases) that results in the smallest possible loss. The algorithm uses the gradient (the vector of partial derivatives) of the loss function with respect to the parameters to update the parameters and iteratively minimize the loss.

## Steps of Gradient Descent

1. **Initialization**: Initialize the weights and biases of the model, typically with small random values.
2. **Forward Pass**: Compute the output of the model (e.g., a neural network) based on the current values of the parameters.
3. **Loss Calculation**: Calculate the loss (error) between the predicted output and the actual target values using a loss function (e.g., [[Mean Squared Error]] or [[Cross-Entropy]] for classification).
4. **Backward Pass** ([[Backpropagation]]): Compute the gradient of the loss function with respect to each model parameter. This is done using the *chain rule* in calculus and is known as backpropagation. The gradients indicate how much change in each parameter will reduce the loss.
5. **Update Parameters**: Adjust the parameters in the opposite direction of the gradient to minimize the loss. The size of the step taken in this direction is determined by the *learning rate*.

$$ θ = θ - 𝛼 \:ᐧ\:\nabla L (θ)$$

## ADAM

## Mini-batch Gradient Descent

## Stochastic Gradient Descent