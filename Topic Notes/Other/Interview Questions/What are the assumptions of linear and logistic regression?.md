# What are the assumptions of linear and logistic regression models?

### Linear Regression:

Linear Regression models have several assumptions:

1. **Linearity:**

   - The relationship between the independent and dependent variable is linear.

2. **Independence:**

   - The residuals (the differences between observed and predicted values) are independent, meaning the residuals are not correlated.

3. **Homoscedasticity:**

   - The residuals have constant variance against fitted values. In other words, the “spread” or “width” of the residuals remains constant across levels of the independent variable.

4. **Normality of Residuals:**

   - For valid hypothesis testing, the residuals should ideally follow a normal distribution.

5. **No or Little Multicollinearity:**

   - Independent variables should not be highly correlated with each other.

6. **No Autocorrelation:**

   - The residuals should not show patterns when plotted against time or other variables.

7. **No Perfect Collinearity:**

   - No independent variable is a perfect linear function of other variables.

8. **Additivity & No Interaction:**

   - The effect of differences in an independent variable X on the dependent variable Y is constant, regardless of the values of the other independent variables.

9. **No Endogeneity:**
   - The model’s errors are not correlated with the independent variables.

### Logistic Regression:

Logistic Regression models have somewhat similar assumptions but are structured differently due to the categorical nature of the dependent variable.

1. **Linearity of Log-Odds:**

   - Logistic regression assumes a linear relationship between the log-odds of the dependent variable and the independent variables.

2. **Independence of Errors:**

   - Observations are assumed to be independent of one another.

3. **No Multicollinearity:**

   - Like linear regression, logistic regression also assumes little to no multicollinearity among the independent variables.

4. **No Outliers or High Leverage Points:**

   - Logistic regression is sensitive to outliers and high leverage points.

5. **Large Sample Size:**

   - Logistic regression requires a larger sample size compared to linear regression for sufficient power.

6. **Binary Outcome:**

   - The dependent variable must be binary (two-class) in nature. For multi-class problems, multinomial logistic regression or other techniques are used.

7. **Independence of Irrelevant Alternatives:**
   - The odds of outcomes do not depend on the presence or absence of other options (more relevant for multinomial logistic regression).

### Conclusion:

While the assumptions are a guideline to understand the conditions under which these models perform best, violating some of these assumptions to a certain extent may still yield useful results, and there are remedies and diagnostics to detect and correct violations. The critical part is to understand these assumptions and verify them during model development and evaluation.
