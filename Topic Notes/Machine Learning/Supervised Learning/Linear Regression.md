A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the *bias* term (also called the *intercept*). $$\begin{eqnarray}\hat y &=& \theta_0 + \theta_1x_1 + \ldots + \theta_nx_n \\ &=& h_\mathbf{\theta}(\mathbf{x}) = \mathbf{\theta}\cdot\mathbf{x}\end{eqnarray}$$
1. $\theta$ is the model's *parameter vector*, containing the bias term $\theta_0$ and the feature weights $\theta_1$ to $\theta_n$
2. $\mathbf{x}$ is the instance's *feature vector*, containing $x_0$ to $x_n$, with $x_0$ always equal to 1
3. $\theta\cdot\mathbf{x}$ is the dot product of the above vectors
4. $h_\theta$ is the *hypothesis function*, using the model parameters $\theta$

The most common performance measure of a regression model is the [[Root Mean Squared Error]]. In practice, it is simpler to minimize the [[Mean Squared Error]] than the RMSE, and it leads to the same result.

> It is often the case that a learning algorithm will try to optimize a different function than the performance measure used to evaluate the final model. This is generally because that function is easier to compute, because it has useful differentiation properties that the performance measure lacks, or because we want to constrain the model during training, as is the case in [[Regularization]].

The MSE of a linear regression hypothesis $h_\theta$ on a training set $\mathbf{X}$ is calculated using: $$\text{MSE}(\mathbf{X}, h_\theta) = \frac1m\sum_{i=1}^m\left(\theta^T\mathbf{x}^{(i)}-y^{(i)}\right)^2.$$
## The Normal Equation

To find the value of $\theta$ that minimizes the cost function, there is a closed-form solution $$\hat\theta=(X^TX)^{-1}X^Ty.$$