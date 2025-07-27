# Importing the necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions
import json

# Define the embedding function using SentenceTransformers
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
  model_name="all-MiniLM-L6-v2"
)

# Create a new instance of ChromaClient to interact with the Chroma DB
client = chromadb.Client()

# Define the name for the collection to be created or retrieved
collection_name = "my_grocery_collection"

# Function to load grocery data from JSON file
def load_grocery_data(file_path="grocery_data.json"):
  try:
    with open(file_path, 'r') as file:
      data = json.load(file)
    return data['grocery_items']
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return []
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")
    return []

# Function to perform a similarity search in the collection
def perform_similarity_search(collection):
  try:
    # Define the query term you want to search for in the collection
    query_term = ["red", "fresh"]

    if isinstance(query_term, str):
      query_term = [query_term]

    # Perform a query to search for the most similar documents to the 'query_term'
    results = collection.query(
      query_texts=query_term,
      n_results=3  # Retrieve top 3 results
    )
    print(f"Query results for '{query_term}':")
    print(results)

    # Check if no results are returned or if the results array is empty
    if not results or not results['ids'] or len(results['ids'][0]) == 0:
      # Log a message indicating that no similar documents were found for the query term
      print(f'No documents found similar to "{query_term}"')
      return

    print(f'Top 3 similar documents to "{query_term}":')
    # Access the nested arrays in 'results["ids"]' and 'results["distances"]'
    for q in range(len(query_term)):
      print(f'Top 3 similar documents to "{query_term[q]}":')

      # Access the nested arrays in 'results["ids"]' and 'results["distances"]'
      for i in range(min(3, len(results['ids'][q]))):
        doc_id = results['ids'][q][i]  # Get ID from 'ids' array
        score = results['distances'][q][i]  # Get score from 'distances' array

        # Retrieve text data from the results
        text = results['documents'][q][i]
        if not text:
            print(f' - ID: {doc_id}, Text: "Text not available", Score: {score:.4f}')
        else:
            print(f' - ID: {doc_id}, Text: "{text}", Score: {score:.4f}')

  except Exception as error:
    print(f"Error in similarity search: {error}")

# Define the main function to interact with the Chroma DB
def main():
  try:
    # Create a collection in the Chroma database with a specified name,
    # distance metric, and embedding function. In this case, we are using
    # cosine distance
    collection = client.create_collection(
      name=collection_name,
      metadata={"description": "A collection for storing grocery data"},
      configuration={
        "hnsw": {"space": "cosine"},
        "embedding_function": ef
      }
    )
    print(f"Collection created: {collection.name}")

    # Load grocery data from JSON file
    grocery_items = load_grocery_data()

    if not grocery_items:
      print("No grocery data loaded. Exiting.")
      return

    # Extract texts, IDs, and metadata from the loaded data
    texts = [item['text'] for item in grocery_items]
    ids = [item['id'] for item in grocery_items]
    metadatas = [item['metadata'] for item in grocery_items]

    # Add documents and their corresponding IDs to the collection
    # The `add` method inserts the data into the collection
    # The documents are the actual text items, and the IDs are unique identifiers
    # ChromaDB will automatically generate embeddings using the configured embedding function
    collection.add(
      documents=texts,
      metadatas=metadatas,
      ids=ids
    )

    # Retrieve all the items (documents) stored in the collection
    # The `get` method fetches all data from the collection
    all_items = collection.get()
    # Log the retrieved items to the console for inspection
    # This will print out all the documents, IDs, and metadata stored in the collection
    print("Collection contents:")
    print(f"Number of documents: {len(all_items['documents'])}")

    perform_similarity_search(collection)

  except Exception as error:  # Catch any errors and log them to the console
    print(f"Error: {error}")

if __name__ == "__main__":
  main()