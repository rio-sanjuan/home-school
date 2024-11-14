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

### Feed Forward Block

### Multi-Head Attention Block

### Residual Connection

### Encoder Block

### Decoder Block

### Projection Layer