
from sentence_transformers import CrossEncoder

model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def rerank_documents(query, documents, top_n=5):
    if not documents:
        return []

    pairs = [
        (query, document["text"])
        for document in documents
    ]

    scores = model.predict(pairs)

    for document, score in zip(documents, scores):
        document["rerank_score"] = float(score)

    ranked_documents = sorted(
        documents,
        key=lambda document: document["rerank_score"],
        reverse=True
    )

    return ranked_documents[:top_n]
