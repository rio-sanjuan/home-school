Similar to [[Adadelta]], RMSprop adjusts the learning rate based on a moving average of the squared gradients, helping with the problem of vanishing or exploding gradients. The names from the fact that it uses a root-mean-squared operation (RMS) to determine the adjustment that is added to the gradients. RMSprop also uses a parameter to control how much it "remembers," and this parameter $\gamma$ is also typically set to 0.9.

```python
optimizer = torch.optim.RMSprop(model.parameters(), lr=0.01)
```