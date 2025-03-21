import weaviate
import ollama
import numpy as np
import os
import fitz
import time
import psutil

# Initialize Weaviate client
client = weaviate.Client("http://localhost:8090")

VECTOR_DIM = 768
COLLECTION_NAME = "PDFEmbeddings"

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

# Clear Weaviate database
def clear_weaviate_store():
    if client.schema.exists(COLLECTION_NAME):
        client.schema.delete_class(COLLECTION_NAME)
    print("Weaviate store cleared.")


# Create a collection in Weaviate
def create_weaviate_schema():
    class_obj = {
        "class": COLLECTION_NAME,
        "vectorIndexType": "hnsw",
        "vectorizer": "none",  # Using our own embeddings
        "properties": [
            {"name": "file", "dataType": ["string"]},
            {"name": "page", "dataType": ["int"]},
            {"name": "chunk", "dataType": ["string"]},
        ],
    }
    client.schema.create_class(class_obj)
    print("Schema created successfully.")


# Generate an embedding using nomic-embed-text
def get_embedding(text: str, model: str = "nomic-embed-text:latest") -> list:
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the embedding in Weaviate
def store_embedding(file: str, page: int, chunk: str, embedding: list):
    data_object = {
        "file": file,
        "page": page,
        "chunk": chunk,
    }
    client.data_object.create(
        data_object, class_name=COLLECTION_NAME, vector=np.array(embedding, dtype=np.float32)
    )
    print(f"Stored embedding for: {chunk[:50]}...")


# Extract text from a PDF file
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# Split text into chunks
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files
def process_pdfs(data_dir):

    # Track memory usage
    mem_start = get_memory_usage()
    
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                for chunk in chunks:
                    embedding = get_embedding(chunk)
                    store_embedding(file_name, page_num, chunk, embedding)
            print(f" -----> Processed {file_name}")

    # End memory tracking
    mem_end = get_memory_usage()
    print(f"Memory usage: {mem_end - mem_start:.2f} MB")


# Query Weaviate
def query_weaviate(query_text: str, top_k=5):
    embedding = get_embedding(query_text)
    response = (
        client.query.get(COLLECTION_NAME, ["file", "page", "chunk"])
        .with_near_vector({"vector": embedding})
        .with_limit(top_k)
        .do()
    )

    for item in response["data"]["Get"][COLLECTION_NAME]:
        print(f"File: {item['file']}, Page: {item['page']}\nChunk: {item['chunk'][:200]}...\n")


def main():
    clear_weaviate_store()
    create_weaviate_schema()

    # Start time it takes to read in pdf
    start_time = time.time()
    
    process_pdfs("../Data/")
    
    # End time it takes to read in pdf
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print("\n---Done processing PDFs---\n")

    query_weaviate("What is the capital of France?")

    pass


if __name__ == "__main__":
    try:
        main()
    finally:
        try:
            if client._connection._session is not None:
                client._connection._session.close()  # Properly close session
        except Exception as e:
            print(f"Error closing session: {e}")
