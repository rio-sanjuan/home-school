A linear model makes a prediction by computing a weighted sum of the input features, plus a constant called the *bias* term (also called the *intercept*). $$\begin{eqnarray}\hat y &=& \theta_0 + \theta_1x_1 + \ldots + \theta_nx_n \\ &=& h_\mathbf{\theta}(\mathbf{x}) = \mathbf{\theta}\cdot\mathbf{x}\end{eqnarray}$$
1. $\theta$ is the model's *parameter vector*, containing the bias term $\theta_0$ and the feature weights $\theta_1$ to $\theta_n$
2. $\mathbf{x}$ is the instance's *feature vector*, containing $x_0$ to $x_n$, with $x_0$ always equal to 1
3. $\theta\cdot\mathbf{x}$ is the dot product of the above vectors
4. $h_\theta$ is the *hypothesis function*, using the model parameters $\theta$

The most common performance measure of a regression model is the [[Root Mean Squared Error]]. In practice, it is simpler to minimize the [[Mean Squared Error]] than the RMSE, and it leads to the same result.

> It is often the case that a learning algorithm will try to optimize a different function than the performance measure used to evaluate the final model. This is generally because that function is easier to compute, because it has useful differentiation properties that the performance measure lacks, or because we want to constrain the model during training, as is the case in [[Regularization]].

The MSE of a linear regression hypothesis $h_\theta$ on a training set $\mathbf{X}$ is calculated using: $$\text{MSE}(\mathbf{X}, h_\theta) = \frac1m\sum_{i=1}^m\left(\theta^T\mathbf{x}^{(i)}-y^{(i)}\right)^2.$$
## The Normal Equation

To find the value of $\theta$ that minimizes the cost function, there is a closed-form solution $$\hat\theta=(X^TX)^{-1}X^Ty.$$
## Potential Problems

### Non-linearity of the response-predictor relationships

The linear regression model assumes that there is a straight-line relationship between the predictors and the response. If the true relationship is far from linear, then virtually all of the conclusions that we draw from the fit are suspect. In addition, the prediction accuracy of the model can be significantly reduced.

[[Residual Plots]] are a useful graphical tool for identifying non-linearity. Given a simple linear regression model, we can plot the residuals $e_i = y_i - \hat y_i$, versus the predictor $x_i$. If the residual plot indicates that there are non-linear associations in the data, then a simple approach is to use non-linear transformations of the predictors, such as $\log X$, $\sqrt X$, and $X^2$, in the regression model.
### Correlation of error terms

An important assumption of the linear regression model is that the error terms $\epsilon_1, \ldots, \epsilon_n$, are uncorrelated. If there is correlation among the error terms, then the estimated standard errors will tend to underestimate the true standard errors. As a result, confidence and prediction intervals will be narrower than they should be. In addition, $p$-values associated with the model will be lower than they should be; this could cause us to erroneously conclude that a parameter is statistically significant. In short, if the error terms are correlated, we may have an unwarranted sense of confidence in the model.

As an example, if we accidentally doubled our data, then the observations and error terms would come in pairs. If we ignored this, our standard error calculations would be as if we had a sample size of $2n$, when in fact we only have $n$ samples. Our estimated parameters would be the same, but the confidence intervals would be narrower by a factor of $\sqrt 2$.
### Non-constant variance of error terms

Often it is the case that the variances of the error terms are non-constant. For instance, the variances of the error terms may increase with the value of the response. One possible solution is to transform the response $Y$ using a concave function like $\log Y$ or $\sqrt Y$. Such a transformation results in a greater amount of shrinkage of the larger responses, leading to a reduction in heteroscedasticity.  
### Outliers

An outlier is a point for which $y_i$ is far from the value predicted by the model. Outliers can arise for a variety of reasons, such as incorrect recording during data collection. Even if an outlier does not have much effect on the least squares fit, it can cause other problems, such as affecting the [[Residual Standard Error]], and thereby affecting the confidence intervals and $p$-values.

If we believe that an outlier has occurred due to an error in data collection or recording, then one solution is to simply remove the observation. However, care should be taken, since an outlier may instead indicate a deficiency with the model, such as a missing predictor.
### High-leverage points

Observations with *high leverage* are data points that have a significant impact on the model's predictions because they are located far from the mean of the predictor variables. These points do not necessarily have large residuals, but their position in the predictor space gives them the potential to strongly influence the regression line or hyperplane. The leverage of a data point is quantified using the **hat matrix**. The diagonal elements of the hat matrix $H$ measure the leverage of each observation.$$H = X(X^TX)^{-1}X^T.$$Values in this matrix range from $\frac 1n$ to $1$, where $n$ is the number of observations. The average leverage is $\frac pn$, where $p$ is the number of predictors plus the intercept. A point is typically considered high-leverage if $h_{ii} > \frac{2p}{n}$. Use diagnostic measures like [[Cook's Distance]] to assess the influence of high-leverage points on the model. Use robust regression techniques (e.g., [[Huber Regression]], [[RANSAC]]) to reduce the impact of high-leverage points.
### Collinearity

Collinearity refers to the situation in which two or more predictor variables are closely related to one another. The presence of collinearity can pose problems in the regression context, since it can be difficult to separate out the individual effects of collinear variables on the response. Since collinearity reduces the accuracy of the estimates of the regression coefficients, it causes the standard error for $\hat\beta_i$ to grow. Recall that that the [[t-statistic]] for each predictor is calculated by dividing $\hat\beta_i$ by its standard error. Consequently, collinearity results in a decline in the $t$-statistic. As a result, we may fail to reject $H_0:\beta_i=0$. This means that the power of the hypothesis test (the probability of correctly detecting a non-zero coefficient) is reduced by collinearity. To avoid such a situation, it is desirable to identify and address potential collinearity problems while fitting the model. 

A simple way to detect collinearity is to look at the correlation matrix of the predictors. Unfortunately, not all collinearity problems can be detected by the inspection of the correlation matrix. It is possible for collinearity to exist between three or more variables even if no pair of variables has a particularly high correlation. This is called *multicollinearity*. The best way to asses multicollinearity is to compute the [[Variance Inflation Factor]].