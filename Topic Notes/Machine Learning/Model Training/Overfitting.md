Overfitting is the situation where a model fits very well to the current data but fails when predicting new samples. It typically occurs when the model has relied too heavily on patterns and trends in the current data set that do not occur otherwise. Since the model only has access to the current data set, it has no ability to understand that such patterns are anomalous, in this way the model fails to *generalize* to new data.

Often, models that are very flexible (called "low-bias models") have a higher likelihood of overfitting the data. Overfitting is one of the primary risks in modeling and should be a concern for practitioners.

While models can overfit to the *data points*, feature selection techniques can overfit to the *predictors*. This occurs when a variable appears relevant in the current data set but shows no real relationship with the outcome once new data are collected. The risk of this type of overfitting is especially dangerous when the number of data points $n$ is small and the number of potential predictors $p$ is very large. As with overfitting to the data points, this problem can be mitigated using a methodology that will show a warning when this is occurring.

## Solutions
1. Simplify the model by selecting one with fewer parameters ([[Regularization]])
2. Gather more training data
3. Reduce noise in the training data (e.g., fix errors/typos and remove outliers)