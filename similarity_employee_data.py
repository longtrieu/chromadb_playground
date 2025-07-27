# Importing necessary modules from the chromadb package:
# chromadb is used to interact with the Chroma DB database,
# embedding_functions is used to define the embedding model
import chromadb
from chromadb.utils import embedding_functions
import json

# Define the embedding function using SentenceTransformers
# This function will be used to generate embeddings (vector representations) for the data
ef = embedding_functions.SentenceTransformerEmbeddingFunction(
  model_name="all-MiniLM-L6-v2"
)

# Creating an instance of ChromaClient to establish a connection with the Chroma database
client = chromadb.Client()

# Defining a name for the collection where data will be stored or accessed
# This collection is likely used to group related records, such as employee data
collection_name = "employee_collection"

# Function to load employee data from JSON file
def load_employee_data(file_path="employee_data.json"):
  try:
    with open(file_path, 'r') as file:
      data = json.load(file)
    return data['employees']
  except FileNotFoundError:
    print(f"Error: File '{file_path}' not found.")
    return []
  except json.JSONDecodeError:
    print(f"Error: Invalid JSON format in '{file_path}'.")
    return []

# Helper function to display similarity search results
def display_similarity_results(results, query_text, result_type=""):
  """Display similarity search results in a formatted way"""
  if not results or not results['ids'] or len(results['ids'][0]) == 0:
    print(f'No documents found similar to "{query_text}"')
    return

  print(f"Query: '{query_text}'")
  for i, (doc_id, document, distance) in enumerate(zip(
      results['ids'][0], results['documents'][0], results['distances'][0]
  )):
    metadata = results['metadatas'][0][i]
    print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
    print(f"     Role: {metadata['role']}, Department: {metadata['department']}")
    if result_type == "leadership":
      print(f"     Experience: {metadata['experience']} years")
    else:
      print(f"     Document: {document[:100]}...")

# Helper function to display metadata filtering results
def display_filter_results(results, filter_description):
  """Display metadata filtering results in a formatted way"""
  print(f"Found {len(results['ids'])} {filter_description}:")
  for i, doc_id in enumerate(results['ids']):
    metadata = results['metadatas'][i]
    print(f"  - {metadata['name']}: {metadata['role']} ({metadata['experience']} years)")

# Function to perform similarity searches
def perform_similarity_searches(collection):
  """Perform various similarity search examples"""
  print("=== Similarity Search Examples ===")

  # Search for Python developers
  print("\n1. Searching for Python developers:")
  query_text = "Python developer with web development experience"
  results = collection.query(
    query_texts=[query_text],
    n_results=3
  )
  display_similarity_results(results, query_text)

  # Search for leadership roles
  print("\n2. Searching for leadership and management roles:")
  query_text = "team leader manager with experience"
  results = collection.query(
    query_texts=[query_text],
    n_results=3
  )
  display_similarity_results(results, query_text, "leadership")

# Function to perform metadata filtering searches
def perform_metadata_filtering(collection):
  """Perform various metadata filtering examples"""
  print("\n=== Metadata Filtering Examples ===")

  # Filter by department
  print("\n3. Finding all Engineering employees:")
  results = collection.get(where={"department": "Engineering"})
  display_filter_results(results, "Engineering employees")

  # Filter by experience range
  print("\n4. Finding employees with 10+ years experience:")
  results = collection.get(where={"experience": {"$gte": 10}})
  display_filter_results(results, "senior employees")

  # Filter by location
  print("\n5. Finding employees in California:")
  results = collection.get(where={"location": {"$in": ["San Francisco", "Los Angeles"]}})
  print(f"Found {len(results['ids'])} employees in California:")
  for i, doc_id in enumerate(results['ids']):
    metadata = results['metadatas'][i]
    print(f"  - {metadata['name']}: {metadata['location']}")

# Function to perform combined similarity and metadata filtering
def perform_combined_search(collection):
  """Perform combined similarity search with metadata filtering"""
  print("\n=== Combined Search: Similarity + Metadata Filtering ===")

  print("\n6. Finding senior Python developers in major tech cities:")
  query_text = "senior Python developer full-stack"
  results = collection.query(
    query_texts=[query_text],
    n_results=5,
    where={
      "$and": [
        {"experience": {"$gte": 8}},
        {"location": {"$in": ["San Francisco", "New York", "Seattle"]}}
      ]
    }
  )

  if not results or not results['ids'] or len(results['ids'][0]) == 0:
    print(f'No documents found similar to "{query_text}"')
    return

  print(f"Query: '{query_text}' with filters (8+ years, major tech cities)")
  print(f"Found {len(results['ids'][0])} matching employees:")
  for i, (doc_id, document, distance) in enumerate(zip(
    results['ids'][0], results['documents'][0], results['distances'][0]
  )):
    metadata = results['metadatas'][0][i]
    print(f"  {i+1}. {metadata['name']} ({doc_id}) - Distance: {distance:.4f}")
    print(f"     {metadata['role']} in {metadata['location']} ({metadata['experience']} years)")
    print(f"     Document snippet: {document[:80]}...")

# Main function to orchestrate all search types
def perform_advanced_search(collection):
  """Orchestrate all types of searches"""
  try:
    perform_similarity_searches(collection)
    perform_metadata_filtering(collection)
    perform_combined_search(collection)
  except Exception as error:
    print(f"Error in advanced search: {error}")

# Defining a function named 'main'
# This function is used to encapsulate the main operations for creating collections,
# generating embeddings, and performing similarity search
def main():
  try:
    # Creating a collection using the ChromaClient instance
    # The 'create_collection' method creates a new collection with the specified configuration
    collection = client.create_collection(
      # Specifying the name of the collection to be created
      name=collection_name,
      # Adding metadata to describe the collection
      metadata={"description": "A collection for storing employee data"},
      # Configuring the collection with cosine distance and embedding function
      configuration={
        "hnsw": {"space": "cosine"},
        "embedding_function": ef
      }
    )
    print(f"Collection created: {collection.name}")

            # Load employee data from JSON file
    employees = load_employee_data()

    if not employees:
      print("No employee data loaded. Exiting.")
      return

    print(f"Loaded {len(employees)} employees from JSON file")
    print("Employees array ready for use:")
    for employee in employees:
      print(f"  - {employee['name']} ({employee['role']})")

    # Create comprehensive text documents for each employee
    # These documents will be used for similarity search based on skills, roles, and experience
    employee_documents = []
    for employee in employees:
      document = f"{employee['role']} with {employee['experience']} years of experience in {employee['department']}. "
      document += f"Skills: {employee['skills']}. Located in {employee['location']}. "
      document += f"Employment type: {employee['employment_type']}."
      employee_documents.append(document)

    # Adding data to the collection in the Chroma database
    # The 'add' method inserts or updates data into the specified collection
    collection.add(
      # Extracting employee IDs to be used as unique identifiers for each record
      ids=[employee["id"] for employee in employees],
      # Using the comprehensive text documents we created
      documents=employee_documents,
      # Adding comprehensive metadata for filtering and search
      metadatas=[{
        "name": employee["name"],
        "department": employee["department"],
        "role": employee["role"],
        "experience": employee["experience"],
        "location": employee["location"],
        "employment_type": employee["employment_type"]
      } for employee in employees]
    )

    # Retrieving all items from the specified collection
    # The 'get' method fetches all records stored in the collection
    all_items = collection.get()
    # Logging the retrieved items to the console for inspection or debugging
    print("Collection contents:")
    print(f"Number of documents: {len(all_items['documents'])}")

    # Call the perform_advanced_search function with the collection and all_items as arguments
    perform_advanced_search(collection)

  except Exception as error:
    # Catching and handling any errors that occur within the 'try' block
    # Logs the error message to the console for debugging purposes
    print(f"Error: {error}")

if __name__ == "__main__":
  main()