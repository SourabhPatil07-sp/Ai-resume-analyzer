import faiss
import numpy as np

dimension = 1536
index = faiss.IndexFlatL2(dimension)

documents = []

def store_embedding(embedding, text):
    vector = np.array([embedding]).astype("float32")
    index.add(vector)
    documents.append(text)

def search(query_embedding):
    vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(vector, 3)
    return [documents[i] for i in indices[0]]
