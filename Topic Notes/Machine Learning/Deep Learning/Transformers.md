A transformer is a type of deep learning model architecture that has revolutionized the field of Natural Language Processing (NLP) and has found applications in various other domains such as computer vision, speech recognition, and more. Introduced in the paper [Attention Is All You Need](https://arxiv.org/abs/1706.03762) in 2017, the Transformer architecture departs from traditional [[Recurrent Neural Networks]] and [[Convolutional Neural Networks]] by relying entirely on a mechanism called *self-attention* to process input data.

## Structure

The transformer is divided into two main parts:
1. **Encoder**: Processes the input data and generates a set of representations
2. **Decoder**: Uses these representations to generate the desired output, such as translating a sentence from one language to another
Each encoder and decoder is composed of multiple identical layers (commonly six in the original Transformer).

### Input Embedding

```python
import math
import torch.nn as nn

class InputEmbedding(nn.Module):
    """
    Input Embedding Layer for Transformer Models.

    This module transforms input token indices into dense vector representations and scales them by the square root of the model's dimensionality. This scaling helps in stabilizing the gradients during training, as suggested in the original Transformer architecture.

    Args:
        d_model (int): Dimensionality of the embedding vectors. This should match the model's hidden size to ensure compatibility across layers.
        vocab_size (int): Size of the vocabulary, i.e., the total number of unique tokens.
    """
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model  # Dimensionality of the embeddings
        self.vocab_size = vocab_size  # Number of unique tokens in the vocabulary

        # Embedding layer that converts token indices to dense vectors
        self.embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=d_model)

    def forward(self, x):
        """
        Forward pass through the Input Embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.
                              Shape: (batch_size, sequence_length)

        Returns:
            torch.Tensor: Output tensor of scaled embeddings.
                          Shape: (batch_size, sequence_length, d_model)
        """
        # Convert token indices to embeddings
        # Shape: (batch_size, sequence_length, d_model)
        embedded = self.embedding(x)  

        # Scale embeddings by the square root of d_model
        scaled_embeddings = embedded * math.sqrt(self.d_model)

        return scaled_embeddings

```

The input embedding is responsible for converting discrete input tokens (such as words or subwords) into continuous vector representations of a fixed size (`d_model`) that the model can process. These embeddings allow the model to learn meaningful relationships between tokens based on their contextual usage within the training data.

By default, `nn.Embedding` initializes the embedding weights uniformly in the range $$ \left[-\sqrt{\frac{1}{d_{\text{model}}}}, \sqrt{\frac{1}{d_{\text{model}}}}\right], $$and the embeddings are usually scaled by the square root of their dimensionality ($\sqrt{d_{\text{model}}}$) to help in maintaining the variance of the embeddings consistent with the positional encodings that are added later. This scaling ensures that the magnitude of the input embeddings does not disproportionately affect the gradients during training, promoting more stable and efficient learning. (This was found empirically in the paper)
#### Example

```python
import torch

embedding_layer = InputEmbedding(d_model=512, vocab_size=1e4)
input_tokens = torch.tensor([[1, 234, 56, 789, 10], [345, 67, 890, 12, 345]])

embedded = embedding_layer(input_tokens)
print(embedded.shape) # Output: torch.Size([2, 5, 512])
```
### Positional Encoding

$$ 
\begin{eqnarray}
PE_{(pos,2i)} &=& \sin(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}}) \\
PE_{(pos,2i + 1)} &=& \cos(\frac{pos}{10000^{\frac{2i}{d_\text{model}}}})
\end{eqnarray}
$$
where:
* $pos$ is the position in the sequence
* $i$ is the dimension index
* $d_\text{model}$ is the dimensionality of the embeddings

```python
import math
import torch
import torch.nn as nn

class PositionalEncoding(nn.Module):
    """
    Positional Encoding Module for Transformer Models.

    This module injects information about the relative or absolute position of tokens in the sequence. Since Transformer architectures do not inherently capture the sequential order of input data, positional encodings are added to the token embeddings to provide the model with information about token positions.

    The positional encodings have the same dimension (`d_model`) as the embeddings so that the two can be summed. The encoding uses sine and cosine functions of different frequencies as described in the "Attention Is All You Need" paper.

    Args:
        d_model (int): Dimensionality of the embeddings (must match the embedding size).
        seq_len (int): Maximum length of the input sequences.
        dropout (float): Dropout rate applied after adding positional encodings.
    """
    def __init__(self, d_model: int, seq_len: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # Dimensionality of the embeddings
        self.seq_len = seq_len  # Maximum sequence length
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization

        # Initialize a positional encoding matrix with zeros
        pe = torch.zeros(seq_len, d_model)

        # Create a tensor containing position indices (0 to seq_len-1)
        # Shape: (seq_len, 1)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)  

        # Compute the div_term, which determines the frequency of the sine and cosine functions
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(1e4) / d_model))

        # Apply sine to even indices in the positional encoding matrix
        pe[:, 0::2] = torch.sin(position * div_term)

        # Apply cosine to odd indices in the positional encoding matrix
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension to the positional encoding matrix
        # Shape: (1, seq_len, d_model)
        pe = pe.unsqueeze(0)  

        # Register `pe` as a buffer to ensure it's saved with the model but not updated during training
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to add positional encodings to the input embeddings.

        Args:
            x (torch.Tensor): Input tensor containing token embeddings.
                              Shape: (batch_size, sequence_length, d_model)

        Returns:
            torch.Tensor: Output tensor with positional encodings added and dropout applied.
                          Shape: (batch_size, sequence_length, d_model)
        """
        # Add positional encoding to the input embeddings
        # Ensure that the positional encoding matches the sequence length of x
        # `.requires_grad_(False)` ensures that the positional encodings are not updated during backpropagation
        x = x + self.pe[:, :x.size(1), :].requires_grad_(False)

        # Apply dropout for regularization
        return self.dropout(x)
```

The `PositionalEncoding` class is designed to inject information about the position of tokens within a sequence into the model. Since Transformers process all tokens in parallel and lack inherent sequential information, positional encodings are essential for the model to understand the order of tokens.

The use of sine and cosine function with varying frequencies allows the model to learn to attend to relative positions. The properties of these functions enable the model to easily learn to attend by relative positions since, for any fixed offset $k$, $PE_{pos + k}$ can be represented as a linear function of $PE_{pos}$.
#### Example Usage

```python
import torch
import torch.nn as nn

# Define model parameters
d_model = 512       # Embedding dimension
seq_len = 100       # Maximum sequence length
dropout = 0.1       # Dropout rate
vocab_size = 10000  # Vocabulary size

# Instantiate the InputEmbedding and PositionalEncoding modules
input_embedding = InputEmbedding(d_model=d_model, vocab_size=vocab_size)
pos_encoding = PositionalEncoding(d_model=d_model, seq_len=seq_len, dropout=dropout)

# Example input: batch of 2 sequences, each with 50 tokens
input_tokens = torch.randint(0, vocab_size, (2, 50))  # Shape: (2, 50)

# Pass through the embedding layer
embedded = input_embedding(input_tokens)  # Shape: (2, 50, 512)

# Add positional encodings
embedded_with_pos = pos_encoding(embedded)  # Shape: (2, 50, 512)

print(embedded_with_pos.shape)  # Output: torch.Size([2, 50, 512])
```
### Layer Normalization

```python
import torch
import torch.nn as nn

class LayerNormalization(nn.Module):
    """
    Layer Normalization Module.

    This module applies Layer Normalization over the last dimension of the input tensor. Layer Normalization normalizes the inputs across the features for each data point, maintaining the mean and variance of the input. It stabilizes and accelerates the training of deep neural networks by reducing internal covariate shift.

    Args:
        normalized_shape (int or tuple): Input shape from an expected input of size `(batch_size, ..., normalized_shape)`. If a single integer is provided, it is treated as a singleton tuple.
        eps (float, optional): A value added to the denominator for numerical stability. Default: `1e-5`.
        elementwise_affine (bool, optional): If `True`, this module has learnable affine parameters (weight and bias). Default: `True`.

    Attributes:
        layer_norm (nn.LayerNorm): PyTorch's built-in LayerNorm module handling the normalization.
    """
    def __init__(self, normalized_shape, eps: float = 1e-5, elementwise_affine: bool = True):
        super().__init__()
        # Utilize PyTorch's built-in LayerNorm for efficiency and reliability
        self.layer_norm = nn.LayerNorm(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for Layer Normalization.

        Args:
            x (torch.Tensor): Input tensor to be normalized.
                              Shape: `(batch_size, ..., normalized_shape)`

        Returns:
            torch.Tensor: Normalized tensor with the same shape as input `x`.
        """
        return self.layer_norm(x)

```

Layer Normalization (LayerNorm) normalizes the inputs across the features for each individual data sample. Specifically, for a given input vector, it computes the mean and standard deviation across all its feature dimensions and uses these statistics to normalize the input. The mathematical operation can be described as: $$ \text{LayerNorm}(x) = \gamma\cdot\frac{x-\mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$where:
* $x$ is the input vector
* $\mu$ is the mean of $x$ across its feature dimensions
* $\sigma^2$ is the variance of $x$ across its feature dimensions
* $\epsilon$ is a small constant for numerical stability
* $𝛾$ and $\beta$ are learnable scaling and shifting parameters, respectively

Layer Normalization is important because it reduces internal covariate shift, leading to more stable and faster convergence during training. This stabilization allows for higher learning rates and can improve overall training efficiency. Unlike [[Batch Normalization]], which relies on batch statistics and can be sensitive to varying batch sizes, LayerNorm operates independently of the batch dimension. This makes it particularly suitable for models that process variable-length sequences or operate with small batch sizes, such as in natural language processing tasks.

LayerNorm often leads to improved performance and generalization by ensuring that the activations remain within a consistent range, preventing issues like vanishing or exploding gradients.
### Feed Forward Block

```python
import torch
import torch.nn as nn

class FeedForwardBlock(nn.Module):
    """
    Feed Forward Neural Network Block for Transformer Models.

    This module implements a position-wise feed-forward network as described in the Transformer architecture ("Attention Is All You Need" by Vaswani et al., 2017). It consists of two linear transformations with a ReLU activation and dropout applied in between. This block is applied independently to each position in the input sequence, allowing the model to learn complex transformations of the data.

    Args:
        d_model (int): Dimensionality of the input and output features.
                       This should match the model's hidden size to ensure
                       compatibility across layers.
        d_ff (int): Dimensionality of the inner (hidden) layer.
                   Typically larger than `d_model` to allow for richer transformations.
        dropout (float): Dropout rate applied after the activation function
                         for regularization to prevent overfitting.

    Attributes:
        linear_1 (nn.Linear): First linear transformation layer that projects
                              the input from `d_model` to `d_ff` dimensions.
        dropout (nn.Dropout): Dropout layer applied after the activation function
                              to regularize the model.
        linear_2 (nn.Linear): Second linear transformation layer that projects
                              the data back from `d_ff` to `d_model` dimensions.
    """
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)  # First linear transformation
        self.dropout = nn.Dropout(dropout)        # Dropout layer for regularization
        self.linear_2 = nn.Linear(d_ff, d_model)  # Second linear transformation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the FeedForwardBlock.

        This method applies two linear transformations with a ReLU activation
        and dropout in between. The block is applied independently to each position in the input sequence.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, sequence_length, d_model)
                              where:
                              - `batch_size` is the number of samples in the batch,
                              - `sequence_length` is the length of each input sequence,
                              - `d_model` is the dimensionality of the model.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, sequence_length, d_model)
                          after applying the feed-forward transformations.
        """
        # Apply first linear transformation
        # Shape: (batch_size, sequence_length, d_ff)
        hidden = self.linear_1(x)

        # Apply ReLU activation
        # Shape: (batch_size, sequence_length, d_ff)
        activated = torch.relu(hidden)

        # Apply dropout for regularization
        # Shape: (batch_size, sequence_length, d_ff)
        dropped = self.dropout(activated)

        # Apply second linear transformation to project back to original dimension
        # Shape: (batch_size, sequence_length, d_model)
        output = self.linear_2(dropped)

        return output
```

The `FeedForwardBlock` introduces non-linearity and enhances the model's capacity to learn complex patterns.
1. **Enhances Representational Capacity**: By increasing the dimensionality (`d_ff`) in the intermediate layer, the block allows the model to learn more complex and abstract representations of the input data.
### Multi-Head Attention Block

### Residual Connection

### Encoder Block

### Decoder Block

### Projection Layer