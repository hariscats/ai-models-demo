# NLP Tokenizer Demo

This repository contains a Python script that demonstrates how to use the GPT tokenizer from Hugging Face's `transformers` library. The script processes sample texts, tokenizes them, and provides insights into how the tokenizer breaks text into tokens.

## Features
- Tokenizes text into subwords or words using the GPT tokenizer (`gpt2` by default).
- Displays tokens, their corresponding token IDs, and token types (Word/Subword) in a tabular format.
- Demonstrates encoding (text to token IDs) and decoding (token IDs to text).
- Saves tokenization results to a sample output file (`tokenization_results.json`).

## Usage
### Prerequisites
- Install Python 3.8 or later.
- Install required dependencies:
  ```bash
  pip install transformers tabulate
  ```

### Running the Script
To execute the tokenizer demo:
```bash
python tokenizer-demo.py
```

### Output
The script will:
1. Display tokenization results in the terminal.
2. Save results to a file named `tokenization_results.json` in the current directory.

## Example Output
### Terminal Output:
```plaintext
=== Tokenization Demo ===

Input Text 1:
The quick brown fox jumps over the lazy dog.

Tokenization Breakdown:
╒═════════════╤════════════╤════════╕
│ Token       │   Token ID │ Type   │
╞═════════════╪════════════╪════════╡
│ The         │        464 │ Word   │
│ Ġquick      │       1438 │ Subword│
│ Ġbrown      │       1825 │ Subword│
...

=== Decoding Demo ===

Original Text:
The quick brown fox jumps over the lazy dog.
Encoded Tokens:
[464, 1438, 1825, 1778, 3269, 588, 464, 1682, 3290, 13]
Decoded Text:
The quick brown fox jumps over the lazy dog.
```

### File Output (`tokenization_results.json`):
```plaintext
Input Text: The quick brown fox jumps over the lazy dog.
Tokens: ['The', 'Ġquick', 'Ġbrown', 'Ġfox', 'Ġjumps', 'Ġover', 'Ġthe', 'Ġlazy', 'Ġdog', '.']
Token IDs: [464, 1438, 1825, 1778, 3269, 588, 464, 1682, 3290, 13]
Number of Tokens: 10
...
```

## Learning Goals
- Understand how tokenization works for NLP models.
- Explore the relationship between raw text, tokens, and numerical representations.