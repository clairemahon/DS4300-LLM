import redis
import json
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama
from redis.commands.search.query import Query
from redis.commands.search.field import VectorField, TextField


# Initialize models
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
redis_client = redis.StrictRedis(host="localhost", port=6380, decode_responses=True)

VECTOR_DIM = 384
INDEX_NAME = "embedding_index"
DOC_PREFIX = "doc:"
DISTANCE_METRIC = "COSINE"


# load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# def cosine_similarity(vec1, vec2):
#     """Calculate cosine similarity between two vectors."""
#     return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


# Generate an embedding using sentence-transformers
def get_embedding(text: str) -> list:
    embedding = model.encode(text)  # Use the model.encode() method
    # print(f'Document Text:  {embedding}')
    return embedding


def search_embeddings(query, top_k=3):

    query_embedding = get_embedding(query)

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
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
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


def store_embedding(file, page, chunk, embedding):
    """
    Store an embedding in Redis using a hash with vector field.

    Args:
        file (str): Source file name
        page (str): Page number
        chunk (str): Chunk index
        embedding (list): Embedding vector
    """
    key = f"{file}_page_{page}_chunk_{chunk}"
    redis_client.hset(
        key,
        mapping={
            "embedding": np.array(embedding, dtype=np.float32).tobytes(),
            "file": file,
            "page": page,
            "chunk": chunk,
        },
    )


if __name__ == "__main__":
    interactive_search()