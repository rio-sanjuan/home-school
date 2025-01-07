Formula: $$\text{ReLU}(x) = \text{max}(0,x)$$
The **ReLU** function $\text{ReLU} : \mathbb{R} \to [0,\infty]$ is one of the most widely used activation functions. It outputs the input directly if it's positive; otherwise, it outputs zero. **ReLU** is simple and computationally efficient, making it a go-to activation function for many deep learning architectures. Commonly used in hidden layers of deep neural networks, including convolutional neural networks (CNNs) and fully connected networks.

1. **Not bounded**: ReLU can produce very large output values
2. **Sparsity**: ReLU introduces sparsity (many neurons can be inactive at once, outputting 0), which can help with regularization
3. **Avoids vanishing gradient**: Unlike sigmoid and tanh, ReLU does not saturate for positive values, which helps with faster learning and gradient propagation
4. **Dying ReLU Problem**: If a neuron gets stuck in the negative input region, it will always output zero, and thus its gradient will be zero, which can prevent the neuron from learning