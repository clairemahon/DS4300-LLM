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
VECTOR_DIM = 384
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

# Generate an embedding using sentence-transformers
def get_embedding(text: str) -> list:
    embedding = model.encode(text)  # Use the model.encode() method
    # print(f'Document Text:  {embedding}')
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
    print(f' redis name: {redis_client.ft(INDEX_NAME).info()}')
    res = redis_client.ft(INDEX_NAME).search(
        q, query_params={"vec": np.array(embedding, dtype=np.float32).tobytes()}
    )

    for doc in res.docs:
        print(f"{doc.id} \n ----> {doc.vector_distance}\n")


# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
# redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

# VECTOR_DIM = 768
# INDEX_NAME = "embedding_index"
# DOC_PREFIX = "doc:"
# DISTANCE_METRIC = "COSINE"

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# def get_embedding(text: str, model: str = "all-MiniLM-L6-v2") -> list:

#     response = ollama.embeddings(model=model, prompt=text)
#     return response["embedding"]

# load model
# model = SentenceTransformer('all-MiniLM-L6-v2')

# def get_embedding(text: str) -> list:
#     embedding = model.encode(text)  # Use the model.encode() method
#     print("EMBEDDING: ", embedding)
#     return embedding


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)
    # print("Query embedding: ", query_embedding)

    # Convert embedding to bytes for Redis search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()

    try:
        # Construct the vector similarity search query
        # Use a more standard RediSearch vector search syntax
        # q = Query("*").sort_by("embedding", query_vector)

        q = (
            Query("*=>[KNN 5 @embedding $vec AS vector_distance]")
            .sort_by("vector_distance")
            .return_fields("id", "file", "page", "chunk", "vector_distance")
            .dialect(2)
        )

        # Perform the search
        results = redis_client.ft(INDEX_NAME).search(
            q, query_params={"vec": query_vector}
        )

        # Transform results into the expected format
        top_results = [
            {
                "file": result.file,
                "page": result.page,
                "chunk": result.chunk,
                "similarity": result.vector_distance,
            }
            for result in results.docs
        ][:top_k]

        # Print results for debugging
        for result in top_results:
            print(
                f"---> File: {result['file']}, Page: {result['page']}, Chunk: {result['chunk']}"
            )

        return top_results

    except Exception as e:
        print(f"Search error: {e}")
        return []


def generate_rag_response(query, context_results):

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

    # Generate response using Ollama
    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )

    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 RAG Search Interface")
    print("Type 'exit' to quit")

    while True:
        query = input("\nEnter your search query: ")

        if query.lower() == "exit":
            break

        start_time= time.time()

        # Search for relevant embeddings
        context_results = search_embeddings(query)

        # Generate RAG response
        response = generate_rag_response(query, context_results)

        end_time = time.time()
        time_taken = end_time - start_time
        print(f"⏱️ Search took {time_taken:.2f} seconds.")

        print("\n--- Response ---")
        print(response)




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
    interactive_search()


if __name__ == "__main__":
    main()


