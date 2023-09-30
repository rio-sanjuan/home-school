# What is the difference between bagging and boosting?

Both Bagging and Boosting and ensemble learning methods, which combine the decisions from multiple models to improve the model's performance, with each working in different ways.

## Bagging

Bagging aims to reduce overfitting in high-variance, low-bias models (like Decision Trees) by averaging over multiple models to smooth out the noise and reduce variance. It involves building multiple models (often of the same type) from different subsamples of the training set, typically drawn with replacement (bootstrap samples). The final model's output is an average (for regression problems) or a majority vote (for classification problems) of the outputs of individual models. Because models are built and trained individually, bagging algorithms are parallelizable and train efficiently.

### Example Algorithm:

Random Forest is a classic example where multiple Decision Trees are created, and their predictions are aggregated as described above.

## Boosting

Boosting aims to increase the accuracy of models by focusing on the training instances that are harder to predict. It is used to convert weak models (high bias, low variance) into strong models. Models are built sequentially, with each new model trying to correct the errors of the combined ensemble of existing models. The final model's output is typically a weighted sum or a weighted vote of the individual models, where the weights depend on the performance of the individual models. Because models are built sequentially, boosting algorithsm are generally more time-consuming than bagging algorithms.

### Example Algorithm:

Gradient Boosted Trees (like XGBoost and LightGBM) and AdaBoost are popular exmaples of boosting algorithms.
