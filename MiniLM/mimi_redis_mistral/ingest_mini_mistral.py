import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from redis.commands.search.query import Query
import os
import fitz
import ollama


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

def get_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> list:

    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


# Store the calculated embedding in Redis
def store_embedding(doc_id: str, text: str, embedding: list):
    key = f"{DOC_PREFIX}{doc_id}"
    redis_client.hset(
        key,
        mapping={
            "text": text,
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),  # Store as byte array
        },
    )
    print(f"Stored embedding for: {text}")

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

    process_pdfs("../data/")
    print("\n---Done processing PDFs---\n")
    query_redis("What is the capital of France?")


if __name__ == "__main__":
    main()


