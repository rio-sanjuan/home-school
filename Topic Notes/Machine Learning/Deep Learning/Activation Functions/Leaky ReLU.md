Formula: $$ \text{Leaky ReLU}(x) = \begin{cases}
x & x > 0 \\
𝛼 x & x \leq 0
\end{cases} $$
Where $\alpha$ is a small constant (typically $𝛼 = 0.01$).
1. Similar to ReLU, but allows a small, non-zero output for negative values of the input (with slope $\alpha$).
2. Helps to address the **Dying ReLU Problem**, ensuring that negative inputs do not completely cause neurons to become inactive
3. $\alpha$ is a hyperparameter that controls how much negative input contributes to the output