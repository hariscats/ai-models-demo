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

# Load documents from data/documents.json
def load_documents(file_path):
    try:
        with open(file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Error: File not found - {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in file - {file_path}")
        return None

# Display results in a readable format
def display_results(results, documents):
    if not results:
        print("No results to display.")
        return
    print("\n--- Reranked Documents ---\n")
    for idx, result in enumerate(results.results):
        print(f"Rank: {idx + 1}")
        print(f"Score: {result.relevance_score:.2f}")
        print(f"Document: {documents[result.index]['text']}\n")

def main():
    # Define file path for documents
    file_path = "data/documents.json"
    documents = load_documents(file_path)
    if not documents:
        return

    # Define query
    query = "Are there fitness-related perks?"

    # Perform reranking
    try:
        results = co.rerank(query=query, documents=documents, top_n=2, model="rerank-english-v3.0")
        display_results(results, documents)
    except Exception as e:
        print(f"Error during reranking: {e}")

if __name__ == "__main__":
    main()
