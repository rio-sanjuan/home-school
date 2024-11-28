Formula: $$ \text{ELU}(x) = \begin{cases}
x & x > 0 \\
\alpha(e^x - 1) & x \leq 0
\end{cases} $$

1. Range of $(-𝛼, \infty)$
2. For positive inputs, ELU behaves like ReLU. For negative inputs, it has a smooth exponential curve, which avoids the zero-centered issue of ReLU and Leaky ReLU
3. ELU has a non-zero centered output for negative values, which can help with training dynamics
4. It can lead to faster learning because it avoids the vanishing gradient problem for negative inputs
5. ELU can be computationally more expensive due to the exponential function, and it may lead to slower convergence for certain tasks