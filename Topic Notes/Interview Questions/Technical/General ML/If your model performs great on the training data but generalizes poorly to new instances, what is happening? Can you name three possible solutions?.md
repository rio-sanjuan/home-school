This is an example of [[Overfitting]], or a model with high variance. To correct this, you can:

1. Reduce model complexity (e.g., [[Regularization]])
2. Add more training data (if possible)
3. Remove outliers from the dataset (check for more data cleaning to be done)
4. Add a validation set (evaluate your model on dataset that is distinct from the training set)
5. Add [[Dropout]] in the case of deep learning