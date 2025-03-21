#Chroma Vector DB
#Installation
# pip install chromadb

from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import numpy as np
import os
import fitz
import ollama
from sentence_transformers import SentenceTransformer
import time
import psutil

# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')  # Model from sentence-transformers library

client = chromadb.Client()

# Chroma collection setup
VECTOR_DIM = 768
COLLECTION_NAME = "embedding_collection"
DISTANCE_METRIC = "cosine"

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

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

def store_embedding(file: str, page: str, chunk: str, embedding: list, collection):
    metadata = {
        "file": file,
        "page": page,
        "chunk": chunk,
    }
    
    # Add the document, embedding, and metadata to Chroma
    collection.add(
        documents=[chunk],  # Text chunk
        metadatas=[metadata],  # Metadata associated with the chunk
        embeddings=[embedding],  # Embedding vector
        ids=[f"{file}_page_{page}_chunk_{chunk}"]  # Unique ID for the chunk (using file, page, chunk as unique identifier)
    )
    
    #print(f"Stored embedding for: {chunk}")

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
    # End memory tracking
    mem_start = get_memory_usage()
    print(f"Memory usage: {mem_end - mem_start:.2f} MB")
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
                        chunk=str(chunk),
                        embedding=embedding,
                        collection=collection
                    )
            print(f" -----> Processed {file_name}")
    mem_end = get_memory_usage()
    print(f"Memory usage: {mem_end - mem_start:.2f} MB")


# Query Chroma collection
def query_chroma(query: str, collection):
    embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[embedding],
        n_results=5
    )
    #print("Raw Query Results:", results)  # Debugging the raw query result
    top_results = []

    # Loop through results and access documents, metadata, and distances
    for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
        # Here, we store the document text (doc), metadata (file, page, chunk), and similarity score (distance)
        
        # Check if distance is a list, and if so, take the first element
        if isinstance(distance, list):
            distance = distance[0]  # Taking the first element of the distance list (if it's a list)


        if isinstance(metadata, dict): 
            top_results.append({
                "file": metadata.get('file', 'Unknown file'),  # Get file name from metadata, default to 'Unknown file'
                "page": metadata.get('page', 'Unknown page'),  # Get page number from metadata, default to 'Unknown page'
                "chunk": doc,  # The chunk of text (document) returned from the search
                "similarity": distance  # The similarity score (cosine distance)
            })

        elif isinstance(metadata, list):
            # Handle case where metadata is a list. For simplicity, take the first item (adjust based on your needs)
            metadata_item = metadata[0] if metadata else {}
            top_results.append({
                "file": metadata_item.get('file', 'Unknown file'),
                "page": metadata_item.get('page', 'Unknown page'),
                "chunk": doc,  # The chunk of text (document) returned from the search
                "similarity": distance  # The similarity score (cosine distance)
            })
        else:
            # If metadata is neither a dictionary nor a list, you might want to handle it differently
            print(f"Unexpected metadata type: {type(metadata)}")
            top_results.append({
                "file": 'Unknown file',
                "page": 'Unknown page',
                "chunk": doc,  # The chunk of text (document) returned from the search
                "similarity": distance  # The similarity score (cosine distance)
            })
    return top_results

# def query_chroma(query: str, collection):
#     embedding = get_embedding(query)
#     results = collection.query(
#         query_embeddings=[embedding],
#         n_results=5
#     )
#     #print("Raw Query Results:", results)  # Debugging the raw query result

#     # Loop through results and access documents, metadata, and distances
#     for doc, metadata, distance in zip(results['documents'], results['metadatas'], results['distances']):
#         # Here, we store the document text (doc), metadata (file, page, chunk), and similarity score (distance)
#         # If the result is an index, fetch the text
#         if isinstance(doc, list):
#             print(f"Document Text: {' '.join(doc)}")  # Join the list if it contains indices and print the corresponding text
#         else:
#             print(f"Document Text: {doc}")  # In case it's not a list, print it directly

#         print(f"Metadata: {metadata}")  # The associated metadata (file, page, chunk)
#         print(f"Distance: {distance}")  # The cosine similarity score)

#     # Print results for debugging
#     # for result in top_results:
#     #     print(f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}, Similarity: {result['similarity']}")



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

        start_time= time.time()

        # Search for relevant embeddings
        context_results = query_chroma(query, collection)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"⏱️ Search took {time_taken:.2f} seconds.")

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
