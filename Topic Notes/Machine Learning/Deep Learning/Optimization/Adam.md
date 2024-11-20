ADAptive Moment Estimation (Adam) combines the benefits of [[RMSprop]] and [[Momentum Gradient Descent]], maintaining per-parameter adaptive learning rates.

```python
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
```
## Parameters

1. **beta1** and **beta2**: Exponential decay rates for moment estimates
2. **eps**: Added for numerical stability