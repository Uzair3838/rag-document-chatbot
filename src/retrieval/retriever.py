import chromadb
from sentence_transformers import SentenceTransformer

client = chromadb.PersistentClient(path = 'vector_db')

collection = client.get_collection(
    name = "documents"
)

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_documents(query, top_k = 5):
    query_embedding = model.encode([query])
    
    results = collection.query(
        query_embeddings = query_embedding.tolist(),
        n_results=top_k
        
    )
    retrieved_documents = []
    
    for i , document in enumerate(results['documents'][0]):
        metadata = results['metadatas'][0][i]
        
        retrieved_documents.append({
            "text": document,
            "source": metadata['source'],
            "page": metadata['page']
        })
    
    return retrieved_documents


