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
2. **Introduces Non-Linearity**: The ReLU activation function enables the model to capture non-linear relationships, which are essential for modeling intricate patterns in data
3. **Regularization Through Dropout**: Dropout prevents the model from becoming overly reliant on specific neurons, promoting robustness and improving generalization to new data
4. **Position-wise Independence**: Applied independently to each position in the sequence, the `FeedForwardBlock` ensures that transformations are consistent across all positions, maintaining the model's ability to handle variable-length sequences effectively
5. **Integration within Transformer Layers**: As part of each Transformer encoder and decoder layer, the `FeedForwardBlock` works alongside self-attention mechanisms to process and transform data, contributing to the overall power and flexibility of the Transformer model
### Multi-Head Attention Block

```python
import math
import torch
import torch.nn as nn

class MultiHeadAttentionBlock(nn.Module):
    """
    Multi-Head Attention Block for Transformer Models.

    This module implements the Multi-Head Attention mechanism as described
	    in the "Attention Is All You Need" paper by Vaswani et al. (2017). 
	    It allows the model to jointly attend to information from different 
	    representation subspaces at different positions. The block consists 
	    of linear projections for queries, keys, and values, followed by
	    scaled dot-product attention applied in parallel across multiple 
	    heads. The outputs from all heads are concatenated and passed through
	    a final linear transformation.

    Args:
        d_model (int): Dimensionality of the input and output features. This 
	        should match the model's hidden size to ensure compatibility 
	        across layers.
        h (int): Number of attention heads. The `d_model` must be divisible 
	        by `h`.
        dropout (float): Dropout rate applied to the attention scores for 
	        regularization to prevent overfitting.

    Attributes:
        d_model (int): Dimensionality of the input and output features.
        h (int): Number of attention heads.
        d_k (int): Dimensionality of each attention head's key/query/value
	        vectors.
        w_q (nn.Linear): Linear layer to project inputs to queries.
        w_k (nn.Linear): Linear layer to project inputs to keys.
        w_v (nn.Linear): Linear layer to project inputs to values.
        w_o (nn.Linear): Linear layer to project concatenated outputs of 
	        all heads.
        dropout (nn.Dropout): Dropout layer applied to attention scores.
        attention_scores (torch.Tensor): Stores the attention scores for 
	        potential analysis.
    """
    def __init__(self, d_model: int, h: int, dropout: float):
        super().__init__()
        self.d_model = d_model  # Dimensionality of the model
        self.h = h                # Number of attention heads

        # Ensure that d_model is divisible by the number of heads
        assert d_model % h == 0, "d_model must be divisible by the number of heads (h)."
        self.d_k = d_model // h  # Dimensionality of each attention head

        # Linear layers to project inputs to queries, keys, and values
        self.w_q = nn.Linear(d_model, d_model)  # Query projection
        self.w_k = nn.Linear(d_model, d_model)  # Key projection
        self.w_v = nn.Linear(d_model, d_model)  # Value projection

        # Linear layer to project concatenated attention outputs
        self.w_o = nn.Linear(d_model, d_model)  # Output projection

        # Dropout layer applied to attention scores
        self.dropout = nn.Dropout(dropout)

        # Placeholder for attention scores (useful for visualization or analysis)
        self.attention_scores = None

    @staticmethod
    def attention(query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                 mask: torch.Tensor, dropout: nn.Dropout) -> (torch.Tensor, torch.Tensor):
        """
        Compute the scaled dot-product attention.

        Args:
            query (torch.Tensor): Query tensor of shape 
	            (batch_size, h, seq_len, d_k).
            key (torch.Tensor): Key tensor of shape 
	            (batch_size, h, seq_len, d_k).
            value (torch.Tensor): Value tensor of shape 
	            (batch_size, h, seq_len, d_k).
            mask (torch.Tensor): Mask tensor to prevent attention to 
	            certain positions. Shape: (batch_size, 1, 1, seq_len) 
		        or similar, depending on masking strategy.
            dropout (nn.Dropout): Dropout layer to apply to the 
	            attention scores.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                - Output tensor after applying attention, shape: 
	                (batch_size, h, seq_len, d_k).
                - Attention scores tensor, shape: 
	                (batch_size, h, seq_len, seq_len).
        """
        d_k = query.size(-1)  # Dimensionality of the key/query vectors

        # Compute the scaled dot-product attention scores
        # Shape: (batch_size, h, seq_len, seq_len)
        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)

        if mask is not None:
            # Apply the mask by setting masked positions to a large 
            # negative value
            attention_scores = attention_scores.masked_fill(mask == 0, -1e9)

        # Apply softmax to obtain attention weights
        attention_weights = torch.softmax(attention_scores, dim=-1)

        if dropout is not None:
            # Apply dropout to the attention weights
            attention_weights = dropout(attention_weights)

        # Compute the final attention output
        # Shape: (batch_size, h, seq_len, d_k)
        attention_output = torch.matmul(attention_weights, value)

        return attention_output, attention_weights

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass for the Multi-Head Attention Block.

        Args:
            q (torch.Tensor): Query tensor of shape 
	            (batch_size, seq_len, d_model).
            k (torch.Tensor): Key tensor of shape 
	            (batch_size, seq_len, d_model).
            v (torch.Tensor): Value tensor of shape 
	            (batch_size, seq_len, d_model).
```

The `MultiHeadAttentionBlock` is a pivotal component of Transformer architectures, enabling the model to focus on different parts of the input sequence simultaneously. Here's a breakdown of its functionality.
1. **Linear Projections for Queries, Keys, and Values**: Queries ($w_q$), Keys ($w_k$), and Values ($w_v$): The input tensors $q$, $k$, and $v$ are each passed through separate linear layers to generate the corresponding query, key, and value vectors. This transformation allows the model to project the input data into different representation subspaces, facilitating diverse attention mechanisms.
2. **Spliiting into Multiple Heads**: The projected queries, keys, and values are reshaped and transposed to create multiple attention heads. Each head operates independently, allowing the model to capture various aspects of the input data simultaneously. The dimensionality of each head ($d_k$) is derived by dividing the model's dimensionality ($d_\text{model}$) by the number of heads ($h$).
3. **Scaled Dot-Product Attention**: For each head, the scaled dot-product attention is computed by taking the dot product of queries and keys, scaling by the square root of $d_k$, applying a mask (if provided) to prevent attention to certain positions, and then applying a softmax function to obtain attention weights. These weights are used to aggregate the values, producing attention outputs for each head.
4. **Concatenation and Final Linear Projection**: The attention outputs from all heads are concatenated and passed through a final linear layer ($w_o$). This step combines the information from different attention heads, producing the final output that integrates diverse contextual information from the input sequence.
### Residual Connection

### Encoder Block

### Decoder Block

### Projection Layer