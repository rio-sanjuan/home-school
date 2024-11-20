RMSprop adjusts the learning rate based on a moving average of the squared gradients, helping with the problem of vanishing or exploding gradients.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```