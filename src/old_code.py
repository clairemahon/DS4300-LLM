

# # Store the embedding in Chroma
# def store_embedding(file: str, page: str, chunk_text: str, embedding: list, collection):
#     metadata = {"file": file, "page": page, "chunk": chunk_text}
#     collection.add(
#         documents=[chunk_text],  # This stores the actual text chunk
#         metadatas=[metadata],
#         embeddings=[embedding],
#         ids=[f"{file}_page_{page}_chunk_{chunk_text}"]  # Use chunk text or an ID based on your needs
#     )
#     #print(f"Stored embedding for: {chunk_text}")  # Print the actual text



#Old version of query_chroma
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
