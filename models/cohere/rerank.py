import os
import json
import cohere
from dotenv import load_dotenv

# Load environment variables from the .env file, if it exists
load_dotenv()

# Function to initialize the Cohere client
def initialize_cohere(api_key):
    try:
        return cohere.ClientV2(api_key=api_key)
    except Exception as e:
        print(f"Error initializing Cohere client: {e}")
        return None

# Function to load documents from a JSON file
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

# Function to rerank documents based on a query
def rerank_documents(co_client, query, documents, top_n=2, model="rerank-english-v3.0"):
    try:
        return co_client.rerank(
            query=query,
            documents=documents,
            top_n=top_n,
            model=model
        )
    except Exception as e:
        print(f"Error during reranking: {e}")
        return None

# Function to display the results in a readable format
def display_results(results, documents):
    if not results:
        print("No results to display.")
        return
    print("\n--- Reranked Documents ---\n")
    for idx, result in enumerate(results.results):
        print(f"Rank: {idx + 1}")
        print(f"Score: {result.relevance_score:.2f}")
        print(f"Document: {documents[result.index]['text']}\n")

# Main function to execute the program
def main():
    # Retrieve API key from environment variable
    API_KEY = os.getenv("COHERE_API_KEY")
    
    # Check if API key is available
    if not API_KEY:
        print("Error: COHERE_API_KEY not found! Please set it as an environment variable or in a .env file.")
        return

    # Initialize the Cohere client
    co = initialize_cohere(api_key=API_KEY)
    if not co:
        return

    # Load documents from a JSON file
    file_path = "documents.json"  # Update with the path to your JSON file
    documents = load_documents(file_path)
    if not documents:
        return

    # Define the query
    query = "Are there fitness-related perks?"

    # Call the rerank function
    results = rerank_documents(co_client=co, query=query, documents=documents)

    # Display the results
    display_results(results, documents)

# Run the main function
if __name__ == "__main__":
    main()
