#Chroma Vector DB
#Installation
# pip install chromadb

from chromadb.config import Settings
from chromadb.utils import embedding_functions
import chromadb
import numpy as np
import os
import fitz
from sentence_transformers import SentenceTransformer


# Initialize SentenceTransformer model
model = SentenceTransformer('all-mpnet-base-v2')  # Model from sentence-transformers library

# db_path = '/Users/clairemahon/DS4300/DS4300-LLM/src/chroma_ingest.py'
# client = chromadb.Client(Settings(chroma_db_impl="sqlite", chroma_db_path=db_path))
client = chromadb.HttpClient(host='localhost', port=8002)
client.heartbeat()

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
        print(f"Collection '{COLLECTION_NAME}' created or already exists.")
    return collection

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
    
    print(f"Stored embedding for: {chunk}")

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
                        chunk=str(chunk),
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


def main():
    #clear_collection(collection)
    collection = create_chroma_collection()

    process_pdfs("/Users/clairemahon/DS4300/DS4300-LLM/data", collection)
    print("\n---Done processing PDFs---\n")
    query_chroma("What is the capital of France?", collection)

    print(f"Available collections: {client.list_collections()}")

if __name__ == "__main__":
    main()