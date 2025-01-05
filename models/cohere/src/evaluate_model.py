import os
import json
import cohere
from dotenv import load_dotenv

# Load environment variables from the .env file
load_dotenv()

# Initialize Cohere client
API_KEY = os.getenv("COHERE_API_KEY")
if not API_KEY:
    raise ValueError("COHERE_API_KEY not found! Please set it in .env file.")
co = cohere.ClientV2(api_key=API_KEY)

# Load labeled dataset from data/labeled_dataset.json
def load_labeled_data(file_path):
    try:
        with open(file_path, "r") as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file - {file_path}")
        return None

# Calculate Mean Reciprocal Rank (MRR)
def calculate_mrr(results, ground_truth):
    for idx, result in enumerate(results.results):
        if ground_truth[result.index] > 0:  # Relevant document
            return 1 / (idx + 1)
    return 0

# Calculate Precision@k
def calculate_precision_at_k(results, ground_truth, k):
    relevant_count = 0
    for i, result in enumerate(results.results[:k]):
        if ground_truth[result.index] > 0:  # Relevant document
            relevant_count += 1
    return relevant_count / k

# Evaluate the model
def evaluate_model(dataset, co_client, top_k=3):
    mrr_scores = []
    precision_at_k_scores = []
    
    for item in dataset:
        query = item["query"]
        documents = [{"text": doc["text"]} for doc in item["documents"]]
        ground_truth = [doc["label"] for doc in item["documents"]]

        # Perform reranking
        try:
            results = co_client.rerank(query=query, documents=documents, top_n=top_k, model="rerank-english-v3.0")
        except Exception as e:
            print(f"Error during reranking for query '{query}': {e}")
            continue

        # Calculate metrics
        mrr = calculate_mrr(results, ground_truth)
        precision_at_k = calculate_precision_at_k(results, ground_truth, k=top_k)

        mrr_scores.append(mrr)
        precision_at_k_scores.append(precision_at_k)

    # Print overall results
    print(f"Mean Reciprocal Rank (MRR): {sum(mrr_scores) / len(mrr_scores):.4f}")
    print(f"Precision@{top_k}: {sum(precision_at_k_scores) / len(precision_at_k_scores):.4f}")

def main():
    # Define file path for labeled dataset
    dataset_path = "data/labeled_dataset.json"
    dataset = load_labeled_data(dataset_path)
    if not dataset:
        return

    # Evaluate the model
    evaluate_model(dataset, co)

if __name__ == "__main__":
    main()
