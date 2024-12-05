A Recurrent Neural Network (RNN) is a type of neural network designed to process sequential data, such as time series, speech, or text by maintaining a hidden state that carries information from previous inputs. Unlike feedforward networks, RNNs can capture temporal dependencies and patterns in sequences, making them well-suited for tasks like language modeling, sequence classification, and time-series prediction. At each time step $t$, the RNN computes the hidden state $h_t$ and output $y_t$ as follows: $$\begin{eqnarray}h_t &=& f(W_hh_{t-1} + W_xx_t + b_h) \\ y_t &=& g(W_yh_t + b_y)\end{eqnarray}$$Here, $x_t$ is the input at time $t$, $h_{t-1}$ is the previous hidden state, $f$ is a non-linear activation function (e.g. [[Hyperbolic Tangent]] or [[Rectified Linear Unit]]), and $W$ and $b$ are learnable weights and biases. The hidden state $h_t$ serves as a memory that summarizes the past sequence.

RNNs are trained using Backpropagation Through Time (BPTT), which unrolls the network across the sequence to compute the gradients. However, they face challenges like [[Vanishing-Exploding Gradient]], limiting their ability to learn long-term dependencies. To address this, variants like [[Long Short-Term Memory]] and [[Gated Recurrent Units]] introduce mechanisms to control information flow, improving memory and gradient stability. RNNs are widely used in natural language processing, speech recognition, and sequential data modeling, though they are increasingly supplanted by [[Transformer]]-based architectures for many applications.