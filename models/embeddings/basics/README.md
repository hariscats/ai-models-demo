# Tokenization and Embeddings Demo

This contains a simple demo of tokenization and embeddings using PyTorch. The script illustrates how to:
- Tokenize raw text into individual words.
- Build a basic vocabulary mapping words to unique indices.
- Convert tokens into indices.
- Create and apply an embedding layer to generate dense vector representations.

## Key Features
- **Simple Tokenizer**: Splits text based on whitespace.
- **Vocabulary Mapping**: Manual dictionary for token-to-index conversion.
- **Embedding Layer**: Uses PyTorch's `nn.Embedding` to generate embeddings.
- **Minimal Dependencies**: Relies only on Python and PyTorch.

## How `nn.Embedding` Works
`nn.Embedding` serves as a learnable lookup table that maps discrete indices (representing words or tokens) to continuous vector representations. Here are some key points:
- **Learnable Parameter Matrix**:  
  When instantiated, `nn.Embedding` creates a matrix of size `(num_embeddings, embedding_dim)`. Each row corresponds to the embedding vector of a token.
- **Index Lookup**:  
  Given a tensor of indices, the embedding layer returns the corresponding rows from the matrix, effectively converting each index into a dense vector.
- **Efficiency**:  
  This approach is more efficient than one-hot encoding or older sparse methods followed by a dense layer since it directly retrieves low-dimensional vectors.
- **Training and Adaptability**:  
  The embedding vectors are initialized randomly and are refined during model training via **backpropagation**. This process helps capture semantic relationships between tokens (e.g., similar words have similar vectors).

## How to Run
1. **Install Dependencies**:  
   Ensure you have Python and PyTorch installed.
2. **Execute the Script**:
   ```bash
   python word-embeddings.py
