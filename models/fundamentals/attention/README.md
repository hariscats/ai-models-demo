# Intuitive Self-Attention Demo

This repository demonstrates an intuitive implementation of a self-attention mechanism using PyTorch. The script guides you through:
- **Tokenization**: Splitting a sentence into words.
- **Vocabulary Mapping & Embeddings**: Converting words to indices and mapping them to dense vectors.
- **Self-Attention**: Computing attention scores and obtaining context-aware token representations.

## Overview

The demo uses the sentence "The cat sat on the mat" and performs the following steps:

1. **Tokenization & Vocabulary**:
   - A simple tokenizer splits the sentence into lowercase tokens.
   - A small vocabulary maps each known token to a unique index.
   - Tokens not found in the vocabulary are assigned a default "unknown" index.

2. **Embedding**:
   - An embedding layer converts token indices into dense 8-dimensional vectors.
   - This step translates discrete tokens into continuous vector representations that are easier for neural networks to process.

3. **Self-Attention Mechanism**:
   - The self-attention module projects the token embeddings into Query, Key, and Value spaces using linear layers.
   - It computes attention scores using a scaled dot-product, normalizes them with softmax, and uses these weights to generate a weighted sum of the value vectors.
   - The output is a context-aware representation of the input tokens, and the attention weights indicate how much each token attends to every other token.

## How to Run

1. **Prerequisites**:
   - Ensure Python and PyTorch are installed on your system.

2. **Execute the Script**:
   ```bash
   python self-attention.py
