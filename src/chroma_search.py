#Chroma Vector DB
#Installation
# pip install chromadb

import chromadb
from chromadb.config import Settings
import ollama
from sentence_transformers import SentenceTransformer
import time
import logging

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')  # Model from sentence-transformers library

# db_path = '/Users/clairemahon/DS4300/DS4300-LLM/src/chroma_ingest.py'
# client = chromadb.Client(Settings(chroma_db_impl="sqlite", chroma_db_path=db_path))

#client = chromadb.Client()
client = chromadb.HttpClient(host='localhost', port=8002)
client.heartbeat()

print(f"Available collections: {client.list_collections()}")


# Chroma collection setup
VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "cosine"


# Get the collection
def get_chroma_collection(collection):
    try:
        collection = client.get_collection(collection)
        print(f"Collection '{collection}' retrieved successfully.")
        return collection
    except Exception as e:
        print(f"Error getting collection: {e}")
        return None

# Generate an embedding using sentence-transformers
def get_embedding(text: str) -> list:
    # Generate embedding using the sentence-transformers model
    embedding = model.encode(text).tolist()
    #print(f"Document Text: {embedding}")
    return embedding

def search_embeddings(query, collection, top_k=3):
    """
    Search for the top-k most similar embeddings to the query in the Chroma collection.
    """
    query_embedding = get_embedding(query)

    # Perform the search in the Chroma collection
    results = collection.query(
        query_embeddings=[query_embedding],  # Query with the embedding
        n_results=top_k  # Number of results to return
    )
    print(f"Raw results: {results}")


    top_results = []
    for result in results['documents'][0]:
        # Ensure 'result' is not a string before trying to access its metadata
        if isinstance(result, dict):
            metadata = result.get('metadata', {})
            top_results.append({
                "file": metadata.get('file', 'Unknown file'),
                "page": metadata.get('page', 'Unknown page'),
                "chunk": metadata.get('chunk', 'Unknown chunk'),
                "similarity": result.get('score', 'Unknown score')
            })

    # Print results for debugging
    # for result in top_results:
    #     print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")

    return top_results

def generate_rag_response(query, context_results):
    """
    Generate a response using a Retrieval Augmented Generation (RAG) model based on the context results.
    """

    # # Prepare context string
    # context_str = "\n".join(
    #     [
    #         f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
    #         f"with similarity {float(result.get('similarity', 0)):.2f}"
    #         for result in context_results
    #     ]
    # )

    # Prepare context string by focusing on the 'metadata' field of the top results
    context_str = ""
    for result in context_results:
        metadata = result.get('metadata', {})
        
        # Include author information if available
        authors = metadata.get('authors', 'Unknown authors')
        context_str += f"Author(s): {authors}\n"  # Add author to the context string

        # Add document details like file, page, chunk
        context_str += f"From {metadata.get('file', 'Unknown file')} (page {metadata.get('page', 'Unknown page')}, chunk {metadata.get('chunk', 'Unknown chunk')})\n"
        context_str += f"Similarity: {result.get('similarity', 'Unknown score')}\n\n"
    
    if not context_str.strip():
        context_str = "No relevant context found in the collection."


    print(f"context_str: {context_str}")

    # Construct prompt with context
    prompt = f"""You are a helpful AI assistant. 
    Use the following context to answer the query as accurately as possible. If the context is 
    not relevant to the query, say 'I don't know'.

Context:
{context_str}

Query: {query}

Answer:"""

    # Generate response using Ollama or any other model
    response = ollama.chat(
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]

def interactive_search(collection):
    """
    Interactive search interface.
    """
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        # Search for relevant embeddings
        context_results = search_embeddings(query, collection)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)

def check_collection_data(collection):
    # Fetch and print a few documents from the collection to ensure it's populated
    documents = collection.get(include=["documents", "embeddings"])
    print(f"Documents in collection: {documents}")


def main():
        collection = get_chroma_collection(COLLECTION_NAME)
        if collection:
            #check_collection_data(collection)
            interactive_search(collection)
        else:
            print("Error: Chroma collection not found.")
        # interactive_search(COLLECTION_NAME)

if __name__ == "__main__":
    main()

