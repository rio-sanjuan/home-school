ADAptive Moment Estimation (Adam) combines the benefits of [[RMSprop]] and [[Momentum Gradient Descent]], maintaining per-parameter adaptive learning rates. The previous algorithms share the idea of saving a list of squared gradients with each weight. They then create a scaling factor by adding up the values in this list, perhaps after scaling them. The gradient at each update step is divided by this total. [[Adagrad]] gives all the elements in the list equal weight when it builds its scaling factor, while [[Adadelta]] and RMSprop treat older elements as less important, and thus they contribute less to the overall total.

Squaring the gradient before putting it into the list is useful mathematically, but when we square a number, the result is always positive. This means that we lose track of whether that gradient in our list was positive or negative, which is useful information. So, to avoid losing this information, we can keep a second list of the gradients without squaring them. Then we can use both lists to derive our scaling factor.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
## Parameters

1. $\beta_1$ and $\beta_2$: Exponential decay rates for moment estimates
2. $\epsilon$: Added for numerical stability