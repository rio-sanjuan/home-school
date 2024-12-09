1. [[Maximum Likelihood Estimate]] (MLE): maximizes the likelihood of the observed data, no prior beliefs about $\beta$
2. [[Maximum A Posteriori]] (MAP): maximizes the posterior, combining likelihood with prior beliefs about $\beta$, adds [[Regularization]] to avoid overfitting
   
## [[Linear Regression]]

> Derive the MLE and MAP estimates of linear regression.

### Step 1: Setup

Linear regression models the relationship with $y = X\beta + \epsilon$ where 
* $y\in\mathbb{R}^n$: vector of observed outputs
* $X\in\mathbb{R}^{n\times p}$: design matrix of features ($n$ rows, $p$ columns)
* $\beta\in\mathbb{R}^p$: coefficients or weights to be determined
* $\epsilon\sim\mathcal{N}(0, \sigma^2I)$: independent and identically distributed [[Gaussian Noise]] with zero mean and variance $\sigma^2$
### Step 2: Likelihood Function

The probability of observing $y$ given $X$ and $\beta$ is:$$\begin{eqnarray}p(y\,\vert\,X,\beta,\sigma^2) &=& \prod_{i=1}^n\frac1{\sqrt{2\pi\sigma^2}}\exp\left(-\frac{(y_i-X_i\beta)^2}{2\sigma^2}\right) \\ &=& \frac1{(2\pi\sigma^2)^{n/2}}\exp\left(-\frac1{2\sigma^2}(y-X\beta)^T(y-X\beta)\right)\end{eqnarray}$$
### Step 3: Log-Likelihood 

The log-likelihood simplifies the expression and is given by:$$\log p(y\,\vert\,X,\beta,\sigma^2)=-\frac n2\log(2\pi)-\frac n2\log(\sigma^2) - \frac1{2\sigma^2}(y-X\beta)^T(y-X\beta)$$

## [[Logistic Regression]]