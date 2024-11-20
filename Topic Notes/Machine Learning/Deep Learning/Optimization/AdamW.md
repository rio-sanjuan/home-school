A variant of [[Adam]] with decoupled weight decay for better regularization

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
```
