# Tokenization and Embeddings Demo

This script contains a basic demo of tokenization and embeddings using PyTorch. The script illustrates how to:
- Tokenize raw text into individual words.
- Build a basic vocabulary mapping words to unique indices.
- Convert tokens into indices.
- Create and apply an embedding layer to generate dense vector representations.

## Key Features
- **Simple Tokenizer**: Splits text based on whitespace.
- **Vocabulary Mapping**: Manual dictionary for token-to-index conversion.
- **Embedding Layer**: Uses PyTorch's `nn.Embedding` to generate embeddings.
- **Minimal Dependencies**: Relies only on Python and PyTorch.

## How to Run
1. Ensure you have Python and PyTorch installed.
2. Run the script:
   ```bash
   python word-embeddings.py
