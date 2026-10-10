
from src.retrieval.retriever import retrieve_documents
from src.retrieval.reranker import rerank_documents

question = "What products does NexaTech Solutions offer, and what does each product do?"

# Retrieve 15 candidate chunks from ChromaDB
documents = retrieve_documents(query=question, top_k=15)

print(f"\nRetrieved {len(documents)} candidate chunks.")

# Rerank candidates and keep the 5 most relevant
ranked_documents = rerank_documents(
    query=question,
    documents=documents,
    top_n=5
)

print("\n" + "=" * 60)
print("TOP 5 RERANKED CHUNKS")
print("=" * 60)

for i, document in enumerate(ranked_documents, start=1):
    print(f"\nRank: {i}")
    print(f"Source: {document['source']}")
    print(f"Page: {document['page']}")
    print(f"Reranker score: {document['rerank_score']:.4f}")
    print(f"Text:\n{document['text'][:500]}")
    print("-" * 60)
