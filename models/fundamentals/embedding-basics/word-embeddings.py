import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Step 1: Tokenization
# -----------------------------------------------------------------------------
# Tokenization is the process of breaking up a text string into smaller units called "tokens."
# In NLP, these tokens are often words (or sometimes subwords/characters).
# Here we implement a simple tokenizer that splits text on spaces.

def simple_tokenize(text):
    """
    Splits the input text into a list of words.
    
    Args:
        text (str): The input sentence.
    
    Returns:
        List[str]: A list of tokens (words).
    """
    return text.split()

# -----------------------------------------------------------------------------
# Step 2: Building a Vocabulary
# -----------------------------------------------------------------------------
# The vocabulary maps each unique token to a unique integer (an index).
# In a real-world application, the vocabulary is built from a large corpus.
# For simplicity, we define a small vocabulary here.

vocabulary = {
    'hello': 0,
    'world': 1,
    'this': 2,
    'is': 3,
    'a': 4,
    'test': 5
}

# Define an index for out-of-vocabulary (unknown) tokens.
unknown_token_index = len(vocabulary)

def tokens_to_indices(tokens, vocab):
    """
    Converts a list of tokens to a list of corresponding indices based on the vocabulary.
    
    Args:
        tokens (List[str]): The list of tokens.
        vocab (dict): The vocabulary mapping tokens to indices.
    
    Returns:
        List[int]: A list of token indices.
    """
    return [vocab.get(token, unknown_token_index) for token in tokens]

# -----------------------------------------------------------------------------
# Example Text Input
# -----------------------------------------------------------------------------
# Let's define an example sentence.
input_text = "hello world this is a simple test"

# Tokenize the input text.
tokens = simple_tokenize(input_text)
print("Tokens:", tokens)  # e.g., ['hello', 'world', 'this', 'is', 'a', 'simple', 'test']

# Convert tokens to their corresponding indices.
token_indices = tokens_to_indices(tokens, vocabulary)
print("Token indices:", token_indices)  # 'simple' is not in vocab, so it gets the unknown index.

# Convert the list of indices into a PyTorch tensor.
# The tensor shape should be (batch_size, sequence_length).
input_tensor = torch.tensor([token_indices])
print("Input tensor:", input_tensor)

# -----------------------------------------------------------------------------
# Step 3: Creating the Embedding Layer
# -----------------------------------------------------------------------------
# An embedding layer converts integer token indices into dense vector representations.
# These dense vectors (or embeddings) capture semantic relationships between words.
# For example, words with similar meanings may have similar vector representations.
#
# Here we define:
# - vocab_size: total number of tokens (including an extra slot for unknown tokens)
# - embedding_dim: size of each embedding vector

vocab_size = len(vocabulary) + 1  # +1 to include the unknown token.
embedding_dim = 16  # Each word will be represented by a 16-dimensional vector.

# Create an instance of the Embedding layer.
embedding_layer = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)

# -----------------------------------------------------------------------------
# Step 4: Generating Embeddings for the Input Tokens
# -----------------------------------------------------------------------------
# Pass the input tensor (containing token indices) through the embedding layer.
# The output is a tensor of shape (batch_size, sequence_length, embedding_dim),
# which means each word is now represented by a 16-dimensional vector.

embedded_output = embedding_layer(input_tensor)

print("Embedded output shape:", embedded_output.shape)
print("Embedded output:", embedded_output)
