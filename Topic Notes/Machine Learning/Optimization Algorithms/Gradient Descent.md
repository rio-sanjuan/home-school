Gradient Descent is a fundamental optimization algorithm used to minimize the loss function, and in turn, train models like neural networks. It's the primary method used for adjusting the parameters (weights and biases) of a model in order to improve its performance on a given task.

At a high level, gradient descent is an iterative optimization technique that seeks to minimize a function by gradually moving toward the steepest direction of descent (negative gradient) from a given point. In the context of deep learning, the function being minimized is typically the loss function (or cost function), which measures the difference between the predicted output of the model and the actual target output.

The goal is to find the set of parameters (weights and biases) that results in the smallest possible loss. The algorithm uses the gradient (the vector of partial derivatives) of the loss function with respect to the parameters to update the parameters and iteratively minimize the loss.

## Steps of Gradient Descent

1. **Initialization**: Initialize the weights and biases of the model, typically with small random values.
2. **Forward Pass**: Compute the output of the model (e.g., a neural network) based on the current values of the parameters.
3. **Loss Calculation**: Calculate the loss (error) between the predicted output and the actual target values using a loss function (e.g., [[Mean Squared Error]] or [[Cross-Entropy Loss]] for classification).
4. **Backward Pass** ([[Backpropagation]]): Compute the gradient of the loss function with respect to each model parameter. This is done using the *chain rule* in calculus and is known as backpropagation. The gradients indicate how much change in each parameter will reduce the loss.
5. **Update Parameters**: Adjust the parameters in the opposite direction of the gradient to minimize the loss. The size of the step taken in this direction is determined by the *learning rate*.
$$ θ = θ - 𝛼 \;ᐧ\;\nabla L (θ)$$
6. **Repeat**: Repeat the process of forward pass, loss calculation, backpropagation, and parameter update for multiple iterations (epochs) until the loss converges to a minimum or the model achieves acceptable performance.

### Learning Rate

The learning rate ($\alpha$) controls how big the steps are when updating the parameters. If the learning rate is too small, the convergence to the optimal solution can be slow. If it's too large, the updates may overshoot the minimum and cause the algorithm to diverge.

## Challenges with Gradient Descent

### Local Minima

In non-convex optimization problems (like those in deep learning), gradient descent can get stuck in local minima or saddle points, meaning it may not find the best global minimum of the loss function. However, in practice, deep learning models often have so many local minima that they are effectively all "good enough", and the optimization process tends to find a region of low loss.

### Vanishing and Exploding Gradients

In very deep neural networks, gradients can become very small (vanishing gradients) or very large (exploding gradients), making it difficult to train the model effectively. Techniques like weight initialization strategies, activation functions, and gradient clipping are used to mitigate these problems.

### Learning Rate Selection

Choosing an appropriate learning rate is crucial. If it's too large, the algorithm may fail to converge. If it's too small, convergence might take too long. Learning rate schedules or adaptive methods like [[Optimizers#Adaptive Moment Estimation]] can help adjust the learning rate during training.

## Batch Gradient Descent

In Batch Gradient Descent, the gradient is computed using the *entire training dataset* at each iteration. It's precise, but it can be computationally expensive, especially for large datasets, since you need to process the entire dataset before making any updates. The update rule is 

$$
θ = θ - \alpha\hspace{0.3em}ᐧ\hspace{0.3em} \frac{1}{m}\sum_{i=1}^{m}\nabla L(x^{(i)}, y^{(i)})
$$
where $m$ is the number of training examples, and $x^{(i)},y^{(i)}$ are the input and target values of the $i^{th}$ training example.

## Stochastic Gradient Descent

In Stochastic Gradient Descent, the gradient is computed using only a single training example at each iteration. This makes it computationally faster, but the updates are noisy, and the path toward the minimum can be erratic. The update rule is

$$
θ = θ - \alpha\hspace{0.3em}ᐧ\hspace{0.3em} \nabla L(x^{(i)}, y^{(i)})
$$

where $x^{(i)},y^{(i)}$ represent the current training example.

## Mini-Batch Gradient Descent

In Mini-Batch Gradient Descent, the gradient is computed using a small subset (mini-batch) of the training data at each iteration, typically 32, 64, or 128 samples. This provides a balance between the precision of batch gradient descent and the speed of stochastic gradient descent. The update rule is 
$$
θ = θ - \alpha\hspace{0.3em}ᐧ\hspace{0.3em} \frac{1}{b}\sum_{i=1}^{b}\nabla L(x^{(i)}, y^{(i)})
$$
where $b$ is the mini-batch size.
