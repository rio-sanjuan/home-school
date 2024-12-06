A Decision Tree is a supervised machine learning model used for both classification and regression tasks. It predicts the target variable by learning simple decision rules inferred from the data features. Decision trees are intuitive, interpretable, and form the basis of more advanced ensemble methods like [[Random Forests]] and gradient boosting algorithms like [[XGBoost]] and [[LightGBM]]. A decision tree is an intuitive and powerful tool for classification and regression tasks. Its flexibility and interpretability make it a popular choice in machine learning, though it often benefits from regularization or ensemble methods to mitigate [[Overfitting]] and instability.
## Key Components

The **root node** is the topmost node of the tree. It represents the entire dataset and is split into branches based on a feature and condition. The **decision nodes** are intermediate nodes where the dataset is split further based on a feature condition. Each decision node leads to one or more branches. The **leaf nodes** are the terminal nodes of the tree. They contain the predicted output (a class label for classification or a continuous value for regression). **Splitting** refers to the process of dividing a node into two or more branches based on a feature condition (e.g., $x > 5$) where a **condition** is a rule or threshold for a feature that determines how the data is split at a decision node.
## Classification

At each node, the model selects a feature and a threshold to split the data into groups that are as homogeneous as possible (i.e., belong to the same class). The splitting process continues recursively until:
1. All the data points in a node belong to the same class
2. A maximum depth is reached
3. A minimum number of samples in a node is reached
The splitting criteria uses a combination of [[Gini Impurity]] and information gained based on [[Cross-Entropy]].
## Regression

The model selects a feature and a threshold to split the data such that the variance of the target variable in each resulting subset is minimized. At the leaf nodes, the prediction is typically the mean value of the target variable for the data points in that node. The splitting criteria here is based on how much the variance of the target variable is reduced after splitting.
## Advantages
1. **Interpretability**: easy to understand and visualize and provides a clear decision-making process
2. **Flexibility**: can handle both categorical and numerical data, and supports multi-class classification and regression type problems
3. **Non-parametric**: decision trees make no assumptions about the distribution of the data
4. **Feature importance**: highlights which features are most important for predictions
## Disadvantages

1. **Overfitting**: decision trees can grow very deep, creating overly complex models that do not generalize well to unseen data
2. **Instability**: small changes in the data can lead to significantly different tree structures
3. **Bias Towards Features with More Levels**: features with more distinct values can dominate the splitting process
4. **Suboptimal splits**: greedy splitting may not always result in the best global solution
## Regularization

To prevent overfitting, decision trees are often regularized. The regularization parameters are:
1. **Maximum depth**: limits the depth of the tree
2. **Minimum samples per leaf**: sets the minimum number of samples required to form a leaf node
3. **Maximum number of features**: restricts the number of features considered for splitting