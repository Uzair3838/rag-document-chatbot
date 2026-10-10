from retrieval.retriever import retrieve_documents

question = "What are the monthly and annual prices of the Starter, Professional, Business, and Enterprise plans?"

results = retrieve_documents(
    query=question,
    top_k=5
)

for i, doc in enumerate(results, start=1):
    print(f"\n--- Result {i} ---")
    print(f"Source: {doc['source']}")
    print(f"Page: {doc['page']}")
    print(f"Text:\n{doc['text']}")
    print("-" * 50)