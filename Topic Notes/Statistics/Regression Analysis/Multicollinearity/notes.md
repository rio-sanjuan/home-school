# Multicollinearity

Multicollinearity refers to a situation in which two or more explanatory variables in a multiple regression model are highly linearly related. In other words, one predictor variable can be used to predict another. This condition creates several problems and challenges in estimating the model parameters and it undermines the statistical significance of an independent variable.

## Problems

1.  _Unreliable Estimates:_ The model become unstable and sensitive to small changes in the coefficients.
2.  _Impaired Interpretability:_ It becomes difficult to asses the individual impact of indepedent variables on the dependent variables due to the high intercorrelations among them.
3.  _Overfitting:_ The model might fit the training data too closely and may perform poorly on new, unseen data.

## Detection

1. _Variance Inflation Factor (VIF):_ VIF quantifies how much the variance of a regression coefficient is increased because of multicollinearity. A VIF above 5-10 indiciates a problematic amount of collinearity.
2. _Correlation Matrix:_ Examining pairwise correlation between indepedent variables. High correlation may indiciate multicollinearity.
3. _Condition Index:_ Calculated using the eigenvalues of the scaled and centered design matrix. A condition index above 30 indicates potentially harmful multicollinearity.

## Solutions

1. _Removing Variables:_ Removing one of the highly correlated variables.
2. _Combining Variables:_ Creating a new variable that represents the collinear variables.
3. _Regularization Methods:_ Using Ridge or Lasso regression that can penalize large coefficients or a large number of features.
4. _Principal Component Analysis (PCA):_ Reducing the dimensionality of the dataset.

## Example

Consider a real estate pricing model with the independent variables being the number of rooms and the house size (in square feet). These two variables could be highly correlated as a house with more rooms is likely to be larger in size, leading to multicollinearity.
