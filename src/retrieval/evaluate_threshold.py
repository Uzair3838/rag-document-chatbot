import uuid
import numpy as np
from sentence_transformers import SentenceTransformer
from ingestion.vector_store import get_collection, store_chunks, clear_collection
from retrieval.retriever import retrieve_documents
from retrieval.reranker import rerank_documents

def run_threshold_test():
    # 1. Setup unique session
    session_id = str(uuid.uuid4())
    print(f"Testing with session_id: {session_id}\n")

    # Use the same embedding model as production
    embed_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # 2. Curated Test Data
    query = "What is the capital of France?"

    test_cases = [
        {
            "text": "The capital of France is Paris.",
            "metadata": {"source": "geo_facts.pdf", "page": 1},
            "label": "POSITIVE (Direct Answer)",
            "expected": True
        },
        {
            "text": "I love eating apples in the autumn and visiting parks.",
            "metadata": {"source": "random_blog.pdf", "page": 1},
            "label": "NEGATIVE (Irrelevant)",
            "expected": False
        },
        {
            "text": "France is a country located in Western Europe with a rich history and culture.",
            "metadata": {"source": "europe_intro.pdf", "page": 1},
            "label": "NEGATIVE (Near-miss/No Answer)",
            "expected": False
        }
    ]

    # Manually prepare chunks and embeddings for storage
    chunks = [{"text": c["text"], "metadata": c["metadata"]} for c in test_cases]
    embeddings = embed_model.encode([c["text"] for c in test_cases])

    print("Inserting test documents...")
    store_chunks(chunks, embeddings, session_id)

    try:
        # 3. Retrieval & Reranking
        print(f"Query: {query}")
        print("-" * 60)

        # Stage 1: Retrieve (top_k=15 to ensure we get all our test cases)
        candidates = retrieve_documents(query, session_id, top_k=15)

        # Stage 2: Rerank
        ranked_results = rerank_documents(query, candidates)

        # 4. Analysis
        print(f"{'Label':<30} | {'Score':<10} | {'Current Logic (>0)':<20} | {'Result'}")
        print("-" * 60)

        # Map results back to our test cases for verification
        results_map = {doc["text"]: doc["rerank_score"] for doc in ranked_results}

        all_passed = True
        for case in test_cases:
            text = case["text"]
            score = results_map.get(text, -999) # -999 if not even retrieved

            is_retained = score > 0
            passed = is_retained == case["expected"]
            if not passed:
                all_passed = False

            status = "PASS" if passed else "FAIL"
            print(f"{case['label']:<30} | {score:>10.4f} | {'Retained' if is_retained else 'Rejected':<20} | {status}")

        print("-" * 60)
        if all_passed:
            print("\nSUCCESS: Current threshold (score > 0) works for these cases.")
        else:
            print("\nFAILURE: Threshold (score > 0) is either too lenient or too strict.")

    finally:
        # 6. Cleanup
        print("\nCleaning up collection...")
        clear_collection(session_id)

if __name__ == "__main__":
    run_threshold_test()
