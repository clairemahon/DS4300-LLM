import weaviate
import ollama
import time

# Connect to Weaviate
client = weaviate.Client("http://localhost:8090")
class_name = "PDFEmbeddings"


def get_embedding(text: str, model: str = "nomic-embed-text") -> list:
    """Generate query embedding using Ollama."""
    try:
        response = ollama.embeddings(model=model, prompt=text)
        if "embedding" not in response or response["embedding"] is None:
            raise ValueError(f"⚠️ Embedding generation failed for model '{model}'")
        return response["embedding"]
    except Exception as e:
        print(f"⚠️ Embedding error: {e}")
        return []  


def search_weaviate(query, top_k=3):
    """Search Weaviate for the most relevant document chunks."""
    query_embedding = get_embedding(query, "nomic-embed-text")

    if not query_embedding:  
        print("⚠️ No embedding was generated. Skipping search.")
        return []  


    # Perform vector similarity search
    try:
        result = client.query.get(
            class_name,
            ["file", "page", "chunk"]
        ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()


        if "data" not in result or "Get" not in result["data"] or class_name not in result["data"]["Get"]:
            print("\n⚠️ Weaviate query returned no results.")
            return [] 

        return result["data"]["Get"][class_name]

    except Exception as e:
        print(f"⚠️ Weaviate search error: {e}")
        return []



def generate_rag_response(query, context_results):
    """Generate a response using RAG (Retrieval-Augmented Generation)."""
    context_str = "\n".join([
        f"From {res['file']} (page {res['page']}): {res['chunk']}"
        for res in context_results
    ])

    prompt = f"""
    You are a helpful AI assistant.
    Use the following context to answer the query as accurately as possible. 
    If the context is not relevant, say 'I don't know'.

    Context:
    {context_str}

    Query: {query}
    Answer:
    """

    response = ollama.chat(
        model="llama3.2:latest", messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]


def interactive_search():
    """Interactive search interface."""
    print("🔍 Weaviate Search Interface (Type 'exit' to quit)")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        start_time= time.time()

        context_results = search_weaviate(query)
        response = generate_rag_response(query, context_results)
        
        end_time = time.time()
        time_taken = end_time - start_time
        print(f"⏱️ Search took {time_taken:.2f} seconds.")

        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
