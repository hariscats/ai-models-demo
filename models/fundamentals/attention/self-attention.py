import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Self-Attention Mechanism
# -----------------------------------------------------------------------------
# This class implements a basic self-attention module.
# The mechanism computes Query, Key, and Value matrices from input embeddings,
# calculates attention scores, normalizes them with softmax, and finally
# generates a weighted sum of values as the attention output.
class SelfAttention(nn.Module):
    def __init__(self, embed_size):
        super(SelfAttention, self).__init__()
        self.embed_size = embed_size
        
        # Linear layers to project input embeddings to Query, Key, and Value spaces.
        self.query = nn.Linear(embed_size, embed_size, bias=False)
        self.key = nn.Linear(embed_size, embed_size, bias=False)
        self.value = nn.Linear(embed_size, embed_size, bias=False)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_size)
        Returns:
            output: Tensor of shape (batch_size, seq_len, embed_size) after attention
            attn_weights: Attention weights of shape (batch_size, seq_len, seq_len)
        """
        # Project input embeddings into Query, Key, and Value representations.
        Q = self.query(x)  # Shape: (batch_size, seq_len, embed_size)
        K = self.key(x)    # Shape: (batch_size, seq_len, embed_size)
        V = self.value(x)  # Shape: (batch_size, seq_len, embed_size)

        # Compute raw attention scores by matrix multiplication of Q and K^T,
        # then scale the scores to stabilize gradients.
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.embed_size ** 0.5)

        # Apply softmax to obtain normalized attention weights.
        attn_weights = F.softmax(scores, dim=-1)

        # Compute the final output as the weighted sum of the value vectors.
        output = torch.matmul(attn_weights, V)

        return output, attn_weights

# -----------------------------------------------------------------------------
# Tokenization and Vocabulary Setup
# -----------------------------------------------------------------------------
# A simple tokenizer that splits a sentence into lowercase words.
def tokenize(sentence):
    return sentence.lower().split()

# Define a small vocabulary mapping for our example sentence.
# Note: In a real application, the vocabulary would be built from a large corpus.
vocab = {
    "the": 0,
    "cat": 1,
    "sat": 2,
    "on": 3,
    "mat": 4
}
vocab_size = len(vocab)  # Number of known tokens

# Convert tokens into their corresponding indices.
# Unknown tokens (not in vocab) will be assigned an index equal to vocab_size.
def tokens_to_indices(tokens, vocab):
    return [vocab.get(token, vocab_size) for token in tokens]

# -----------------------------------------------------------------------------
# Example: "The cat sat on the mat"
# -----------------------------------------------------------------------------
# Tokenize the sentence.
sentence = "The cat sat on the mat"
tokens = tokenize(sentence)
print("Tokens:", tokens)  # Expected: ['the', 'cat', 'sat', 'on', 'the', 'mat']

# Map tokens to their indices.
indices = tokens_to_indices(tokens, vocab)
print("Token Indices:", indices)

# Convert the list of indices into a PyTorch tensor.
# Tensor shape will be (batch_size, sequence_length).
input_tensor = torch.tensor([indices])  # Here, batch_size = 1
print("Input Tensor:\n", input_tensor)

# -----------------------------------------------------------------------------
# Embedding Layer
# -----------------------------------------------------------------------------
# Each token is converted to a dense vector using an embedding layer.
# We add one extra index to handle unknown tokens.
embedding_dim = 8  # Dimensionality for each token's embedding
embedding_layer = nn.Embedding(num_embeddings=vocab_size + 1, embedding_dim=embedding_dim)

# Obtain embeddings for our input tokens.
embedded_tokens = embedding_layer(input_tensor)
print("\nToken Embeddings:\n", embedded_tokens)

# -----------------------------------------------------------------------------
# Apply Self-Attention
# -----------------------------------------------------------------------------
# Initialize the self-attention module with the embedding dimension.
self_attention = SelfAttention(embed_size=embedding_dim)

# Pass the token embeddings through the self-attention layer.
attn_output, attn_weights = self_attention(embedded_tokens)

print("\nSelf-Attention Output:\n", attn_output)
print("\nAttention Weights:\n", attn_weights)
