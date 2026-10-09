from sentence_transformers import SentenceTransformer

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

def create_embeddings(texts):
    embeddings = model.encode(texts)
    
    return embeddings


