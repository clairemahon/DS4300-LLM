import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import os
import fitz
import ollama
import psutil
import time


# initialize redis connection
redis_client = redis.Redis(host = 'localhost', port = 6380, db = 0)

# defines dimensions of embedding vectors
VECTOR_DIM = 768
# defining name of redis index
INDEX_NAME = "embedding_index"
# prefix used to create unique key for each document stored in redis
DOC_PREFIX = "doc:"
# defines distance metric to use when comparing vectors
DISTANCE_METRIC = "COSINE"

def get_memory_usage():
    return psutil.Process().memory_info().rss / (1024 * 1024)

# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# used to clear the redis vector store
def clear_redis_store():
    print("Clearing existing Redis store...")
    redis_client.flushdb()
    print("Redis store cleared.")


# create an hnsw index in redis
def create_hnsw_index():
    try:
        redis_client.execute_command(f"FT.DROPINDEX {INDEX_NAME} DD")
    except redis.exceptions.ResponseError:
        pass

    redis_client.execute_command(
        f"""
        FT.CREATE {INDEX_NAME} ON HASH PREFIX 1 {DOC_PREFIX}
        SCHEMA text TEXT
        embedding VECTOR HNSW 6 DIM {VECTOR_DIM} TYPE FLOAT32 DISTANCE_METRIC {DISTANCE_METRIC}
        """
    )
    print("Index created successfully.")

# # Generate an embedding using sentence-transformers
# def get_embedding(text: str, model) -> list:

#     response = ollama.embeddings(model=model, prompt=text)
#     return response["embedding"]

# Generate an embedding using sentence-transformers
def get_embedding(text: str) -> list:
    embedding = model.encode(text)  # Use the model.encode() method
    return embedding

# Store the calculated embedding in Redis
def store_embedding(file: str, page: str, chunk: str, embedding: list):
    key = f"{DOC_PREFIX}:{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "file": file,
            "page": page,
            "chunk": chunk,
            "embedding": np.array(
                embedding, dtype=np.float32
            ).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {chunk}")

# extract the text from a PDF by page
def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    doc = fitz.open(pdf_path)
    text_by_page = []
    for page_num, page in enumerate(doc):
        text_by_page.append((page_num, page.get_text()))
    return text_by_page


# split the text into chunks with overlap
def split_text_into_chunks(text, chunk_size=300, overlap=50):
    """Split text into chunks of approximately chunk_size words with overlap."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunk = " ".join(words[i : i + chunk_size])
        chunks.append(chunk)
    return chunks


# Process all PDF files in a given directory
def process_pdfs(data_dir):

    # Track memory usage
    mem_start = get_memory_usage()
  
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".pdf"):
            pdf_path = os.path.join(data_dir, file_name)
            text_by_page = extract_text_from_pdf(pdf_path)
            for page_num, text in text_by_page:
                chunks = split_text_into_chunks(text)
                # print(f"  Chunks: {chunks}")
                for chunk_index, chunk in enumerate(chunks):
                    # embedding = calculate_embedding(chunk)
                    embedding = get_embedding(chunk)
                    store_embedding(
                        file=file_name,
                        page=str(page_num),
                        # chunk=str(chunk_index),
                        chunk=str(chunk),
                        embedding=embedding,
                    )
            print(f" -----> Processed {file_name}")

    # End memory tracking
    mem_end = get_memory_usage()
    print(f"Memory usage: {mem_end - mem_start:.2f} MB")


def query_redis(query_text: str):
    q = (
        Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
        .sort_by("vector_distance")
        .return_fields("id", "vector_distance")
        .dialect(2)
    )
    query_text = "Efficient search in vector databases"
    embedding = get_embedding(query_text)
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )
    # print(res.docs)

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")

def main():
    clear_redis_store()
    create_hnsw_index()

    # Start time it takes to read in pdf
    start_time = time.time()
    

    process_pdfs("../Data/")

    # End time it takes to read in pdf
    end_time = time.time()
    print(f"Processing time: {end_time - start_time:.2f} seconds")
    
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()


