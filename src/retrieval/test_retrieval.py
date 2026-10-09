from retriever import retrieve_documents

queries = [
    "How many annual leave days do employees get?",
    "What are the available pricing plans?",
    "What is Project Alpha?"
]

for query in queries:
    print("\n" + "=" * 60)
    print(f"Question: {query}")
    print("=" *60)
    
    results = retrieve_documents(query= query, top_k= 3)
    
    for i, result in enumerate(results, start = 1):
        print(f"\nResult {i}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(result["text"])
    
    
    