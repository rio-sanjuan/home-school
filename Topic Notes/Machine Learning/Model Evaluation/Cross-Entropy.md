Cross-Entropy Loss is a widely used loss function in classification problems, especially in [[Logistic Regression]] and Neural Networks. It measures the difference between the predicted probability distribution and the true distribution (actual labels). Cross-entropy quantifies how well the predicted probabilities align with the actual labels.

Cross-entropy penalizes predictions that are far from the true label. The penalty increases logarithmically as the predicted probability diverges from the actual label. It is particularly suitable for models that output probabilities (e.g., through a [[Softmax]] or [[Sigmoid]] activation function). The cross-entropy measures the "distance" (in terms of information) between two probability distributions: the true distribution $y$ and the predicted distribution $\hat y$.

Cross-entropy loss has the following properties:
1. **Penalizes incorrect predictions**: predictions that are far from the true label result in a high loss
2. **Well suited for Probabilistic Outputs**: works well when the model outputs are probabilities
3. **Logarithmic sensitivity**: the logarithmic nature amplifies the penalty for low confidence in correct predictions
4. **Well-defined gradient**: Cross-entropy works smoothly with gradient-based optimization methods (e.g., [[Gradient Descent]], [[Stochastic Gradient Descent]])
## Binary Cross-Entropy

For binary classification problems, where the true label $y$ is either 0 or 1, the cross-entropy loss is defined as:$$\mathcal L = -\frac 1n\sum_{i=1}^n\left[y_i\log(\hat y_i) + (1-y_i)\log(1-\hat y_i)\right].$$If $y_i$ = 1, the loss focuses on $-\log(\hat y_i)$, ensuring the predicted probability $\hat y_i$ is close to 1.
## Categorical Cross-Entropy

For multiclass classification problems, where $y$ is one-hot encoded, and the model outputs a probability distribution over $C$ classes, the loss is:$$\mathcal L = -\frac 1n\sum_{i=1}^n\sum_{c=1}^Cy_{i,c}\log(\hat y_{i,c}).$$The loss only considers the probability for the true class $y_{i,c}=1$, ignoring probabilities for other classes.