#Chroma Vector DB
#Installation
# pip install chromadb

# import chromadb
# import fitz
# import numpy as np
# import os

from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import numpy as np
import os
import fitz
import ollama
from sentence_transformers import SentenceTransformer

# Initialize SentenceTransformer model
model = SentenceTransformer('all-MiniLM-L6-v2')  # Model from sentence-transformers library

client = chromadb.Client()

# Chroma collection setup
VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "cosine"


# Create a Chroma collection
def create_chroma_collection():
    try:
        collection = client.get_or_create_collection(COLLECTION_NAME)
    except Exception as e:
        print(f"Error creating collection: {e}")
    else:
        clear_collection(collection)
        print(f"Collection '{COLLECTION_NAME}' created or already exists.")
    return client.get_collection(COLLECTION_NAME)

# Clear the collection before adding data
def clear_collection(collection):
    try:
        collection.delete(where={"file": "*"})  # Clears the entire collection
        print(f"Collection {COLLECTION_NAME} cleared.")
    except Exception as e:
        print(f"Error clearing collection: {e}")


# Generate an embedding using sentence-transformers
def get_embedding(text: str) -> list:
    # Generate embedding using the sentence-transformers model
    embedding = model.encode(text).tolist()
    #print(f"Document Text: {embedding}")
    return embedding


# Store the embedding in Chroma
def store_embedding(file: str, page: str, chunk_text: str, embedding: list, collection):
    metadata = {"file": file, "page": page, "chunk": chunk_text}
    collection.add(
        documents=[chunk_text],  # This stores the actual text chunk
        metadatas=[metadata],
        embeddings=[embedding],
        ids=[f"{file}_page_{page}_chunk_{chunk_text}"]  # Use chunk text or an ID based on your needs
    )
    #print(f"Stored embedding for: {chunk_text}")  # Print the actual text


# Extract text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir, collection):
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk_index, chunk in enumerate(chunks):  # 'chunk' is the actual text
                    embedding = get_embedding(chunk)  # Get embedding for the actual text chunk
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        chunk_text=chunk,  # Pass the actual text chunk here, not its index
                        embedding=embedding,
                        collection=collection
                    )
            print(f" -----> Processed {file_name}")



# Query Chroma collection
def query_chroma(query: str, collection):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )
    #print("Raw Query Results:", results)  # Debugging the raw query result

    # Loop through results and access documents, metadata, and distances
    for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
        #print(f"Raw Document Result: {doc}")  # Check the raw document result (which should be chunk indices or text)
        
        # If the result is an index, fetch the text
        if isinstance(doc, list):
            print(f"Document Text: {' '.join(doc)}")  # Join the list if it contains indices and print the corresponding text
        else:
            print(f"Document Text: {doc}")  # In case it's not a list, print it directly

        print(f"Metadata: {metadata}")  # The associated metadata (file, page, chunk)
        print(f"Distance: {distance}")  # The cosine similarity score)

    # # Need to update the following for the above to append the metadata to the top results
    # top_results = []
    # for result in results['documents'][0]:
    #     # Ensure 'result' is not a string before trying to access its metadata
    #     if isinstance(result, dict):
    #         metadata = result.get('metadata', {})
    #         top_results.append({
    #             "file": metadata.get('file', 'Unknown file'),
    #             "page": metadata.get('page', 'Unknown page'),
    #             "chunk": metadata.get('chunk', 'Unknown chunk'),
    #             "similarity": result.get('score', 'Unknown score')
    #         })

    # # Print results for debugging
    # # for result in top_results:
    # #     print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")

    # return top_results

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
    # Prepare context string
    context_str = "\n".join(
        [
            f"From {result.get('file', 'Unknown file')} (page {result.get('page', 'Unknown page')}, chunk {result.get('chunk', 'Unknown chunk')}) "
            f"with similarity {float(result.get('similarity', 0)):.2f}"
            for result in context_results
        ]
    )

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
        context_results = query_chroma(query, collection)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)


def main():
    collection = create_chroma_collection()

    process_pdfs("/Users/clairemahon/DS4300/DS4300-LLM/data", collection)
    print("\n---Done processing PDFs---\n")
    #query_chroma("What are the data structures used?", collection)
    
    # Start the interactive search
    interactive_search(collection)


if __name__ == "__main__":
    main()
