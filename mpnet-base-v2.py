# About the model: The project aims to train sentence embedding models on very large sentence level datasets using a self-supervised contrastive learning objective. 
# We used the pretrained microsoft/mpnet-base model and fine-tuned in on a 1B sentence pairs dataset. 
# We use a contrastive learning objective: given a sentence from the pair, the model should predict which out of a set of randomly sampled other sentences, was actually paired with it in our dataset.

#Our model is intented to be used as a sentence and short paragraph encoder. Given an input text, it ouptuts a vector which captures the semantic information. The sentence vector may be used for information retrieval, clustering or sentence similarity tasks.
#By default, input text longer than 384 word pieces is truncated

#Setup
# pip install -U sentence-transformers

from sentence_transformers import SentenceTransformer
sentences = ["This is an example sentence", "Each sentence is converted"]

model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
embeddings = model.encode(sentences)
print(embeddings)

embedding1 = embeddings[0]
embedding2 = embeddings[1]

from sklearn.metrics.pairwise import cosine_similarity; 
similarity_score = cosine_similarity(embedding1, embedding2)
print(similarity_score)