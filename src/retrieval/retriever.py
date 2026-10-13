
from sentence_transformers import SentenceTransformer
from ingestion.vector_store import get_collection

model = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2"
)


def retrieve_documents(query, session_id, top_k=5):
    collection = get_collection(session_id)

    if collection.count() == 0:
        return []

    query_embedding = model.encode([query])

    results = collection.query(
        query_embeddings=query_embedding.tolist(),
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"]
    )

    retrieved_documents = []

    for i, document in enumerate(results["documents"][0]):
        metadata = results["metadatas"][0][i]

        retrieved_documents.append({
            "text": document,
            "source": metadata["source"],
            "page": metadata["page"],
            "distance": results["distances"][0][i]
        })

    return retrieved_documents
