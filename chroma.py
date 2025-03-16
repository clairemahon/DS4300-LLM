#Chroma Vector DB
#Installation
# pip install chromadb

import chromadb

#Create a chroma client
chroma_client = chromadb.Client() 

#Create a collection where you'll store your documents
collection = chroma_client.create_collection(name="chroma_collection")

#Add some documents to the collection
collection.add(
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ],
    ids=["id1", "id2"]
)

#Query the collection to return the n most similar results
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)

#It starts a Chroma server in-memory, so any data you ingest will be lost when your program terminates. 
# You can use the persistent client or run Chroma in client-server mode if you need data persistence.

