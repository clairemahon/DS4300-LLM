import weaviate
import ollama

# Connect to Weaviate
client = weaviate.Client("http://localhost:8090")
# class_name = "DocumentChunk"
class_name = "PDFEmbeddings"

print(client.schema.get())

def get_embedding(text: str, model: "mistral") -> list:
    """Generate query embedding using Ollama."""
    response = ollama.embeddings(model=model, prompt=text)
    return response["embedding"]


def search_weaviate(query, top_k=3):
    """Search Weaviate for the most relevant document chunks."""
    query_embedding = get_embedding(query, "mistral")

    # Perform vector similarity search
    result = client.query.get(
        class_name,
        ["file", "page", "chunk"]  # Ensure these properties exist in Weaviate
    ).with_near_vector({"vector": query_embedding}).with_limit(top_k).do()

    # Check if the response contains valid data
    if "data" not in result or "Get" not in result["data"] or class_name not in result["data"]["Get"]:
        print("\n⚠️ Weaviate query failed or returned no results.")
        return []

    results = result["data"]["Get"][class_name]
    return [
        {
            "file": doc["file"],
            "page": doc["page"],
            "chunk": doc["chunk"]
        }
        for doc in results
    ]

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
        model="mistral:latest", messages=[{"role": "user", "content": prompt}]
    )
    
    return response["message"]["content"]

def interactive_search():
    """Interactive search interface."""
    print("🔍 Weaviate Search Interface (Type 'exit' to quit)")

    while True:
        query = input("\nEnter your search query: ")
        if query.lower() == "exit":
            break

        context_results = search_weaviate(query)
        response = generate_rag_response(query, context_results)

        print("\n--- Response ---")
        print(response)

if __name__ == "__main__":
    interactive_search()
