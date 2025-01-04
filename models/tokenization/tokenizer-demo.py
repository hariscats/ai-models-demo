from transformers import AutoTokenizer
from tabulate import tabulate
import json

# Load the GPT tokenizer
# Replace "gpt2" with other model names if needed
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Sample input texts
texts = [
    "The quick brown fox jumps over the lazy dog.",
    "A philosopher once said, 'The limits of my language mean the limits of my world.'",
    "🤗 Transformers make working with language models fun and easy!"
]

# Display header
print("\n=== Tokenization Demo ===\n")

# Data structure to store results for saving to a file
output_data = []

# Process each text
for idx, text in enumerate(texts):
    print(f"\nInput Text {idx + 1}:")
    print(text)

    # Tokenize the text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    
    # Prepare table for visualization
    table = []
    token_details = []
    for token, token_id in zip(tokens, token_ids):
        token_type = (
            "Subword" if not token.startswith("Ġ") and token != " " else "Word"
        )
        table.append([token, token_id, token_type])
        token_details.append({"token": token, "token_id": token_id, "type": token_type})

    # Display tokenization results
    print("\nTokenization Breakdown:")
    print(
        tabulate(
            table,
            headers=["Token", "Token ID", "Type"],
            tablefmt="fancy_grid",
            colalign=("left", "right", "center"),
        )
    )

    # Summary
    print(f"\nNumber of Tokens: {len(tokens)}")

    # Add to output data
    output_data.append({
        "input_text": text,
        "tokens": token_details,
        "num_tokens": len(tokens),
    })

# Decoding demonstration
print("\n=== Decoding Demo ===\n")
sample_text = texts[0]
encoded = tokenizer.encode(sample_text)
decoded_text = tokenizer.decode(encoded)

# Display encoded and decoded results
print("Original Text:")
print(sample_text)
print("\nEncoded Tokens:")
print(encoded)
print("\nDecoded Text:")
print(decoded_text)

# Save results to a JSON file
output_file = "tokenization_results.json"
try:
    with open(output_file, "w") as f:
        json.dump(output_data, f, indent=4)
    print(f"\nTokenization results saved to '{output_file}'")
except Exception as e:
    print(f"Failed to save results: {e}")
