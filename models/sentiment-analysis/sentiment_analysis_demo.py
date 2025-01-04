import os
import json
import torch
from transformers import pipeline

def check_pytorch():
    """Check if PyTorch is installed and configured properly."""
    try:
        print(f"PyTorch version: {torch.__version__}")
        if torch.backends.mps.is_available():
            print("Apple Metal MPS acceleration available.")
        else:
            print("MPS acceleration not available; using CPU, which may be slow.")
    except Exception as e:
        print(f"PyTorch is not installed or configured properly: {e}")
        raise

def check_transformers():
    """Check if Hugging Face Transformers is installed properly."""
    try:
        from transformers import __version__ as hf_version
        print(f"Transformers version: {hf_version}")
    except Exception as e:
        print(f"Hugging Face Transformers is not installed or configured properly: {e}")
        raise

def initialize_pipeline(model_name="distilbert-base-uncased-finetuned-sst-2-english"):
    """Initialize the sentiment analysis pipeline."""
    try:
        print(f"Loading model: {model_name}")
        return pipeline("sentiment-analysis", model=model_name, framework="pt")
    except Exception as e:
        print(f"Failed to initialize sentiment analysis pipeline: {e}")
        raise

def perform_analysis(nlp, sentences):
    """Perform sentiment analysis on a list of sentences."""
    results = []
    for sentence in sentences:
        try:
            result = nlp(sentence)
            results.append({
                "sentence": sentence,
                "analysis": {
                    "label": result[0]["label"],
                    "confidence": result[0]["score"]
                }
            })
            print(f"Input: {sentence}")
            print(f"Sentiment: {result[0]['label']}, Confidence: {result[0]['score']:.2f}\n")
        except Exception as e:
            print(f"Error analyzing sentence: {sentence}. Error: {e}")
    return results

def save_results_to_file(results, output_file="output_examples.json"):
    """Save analysis results to a file."""
    try:
        with open(output_file, "w") as f:
            json.dump(results, f, indent=4)
        print(f"Sentiment analysis results saved to {output_file}")
    except Exception as e:
        print(f"Failed to save results to file: {e}")

def main():
    """Main function to run the sentiment analysis demo."""
    print("🚀 Starting Sentiment Analysis Demo")

    # Check installations
    check_pytorch()
    check_transformers()

    # Initialize pipeline
    nlp = initialize_pipeline()

    # Sample Sentences
    sentences = [
        "We are very happy to show you the 🤗 Transformers library.",
        "The software deployment process was cumbersome and frustrating.",
        "A philosopher once said, 'The limits of my language mean the limits of my world.'",
        "Cloud computing is revolutionizing industries by enabling scalable solutions."
    ]

    # Perform Sentiment Analysis
    results = perform_analysis(nlp, sentences)

    # Save Results to File
    save_results_to_file(results)

if __name__ == "__main__":
    main()
