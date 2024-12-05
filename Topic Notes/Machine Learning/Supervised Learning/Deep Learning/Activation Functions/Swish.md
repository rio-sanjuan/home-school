Formula: $$ \text{swish}_\beta(x) = x\;\text{sigmoid}(𝛽 x) = \frac{x}{1 + e^{-𝛽 x}}$$
The swish family of functions was designed to smoothly interpolate between a linear function and the ReLU function. Often defined $𝛽 = 1$, the swish function is smooth and non-monotonic, which can help the model with gradient flow and reduce the likelihood of vanishing gradients. It tends to outperform ReLU in some deep network architectures.