import numpy as np
from retrieval.retriever import retrieve_documents
from retrieval.reranker import rerank_documents
from ingestion.vector_store import get_collection

# Test queries categorized by expected outcome
TEST_QUERIES = {
    "answerable": [
        "What products does NexaTech Solutions offer?",
        "What are the monthly prices of the Starter plan?",
        "What is the company's IT security policy regarding passwords?",
        "What is the purpose of Project Alpha?",
        "How should employees handle attendance and leave?"
    ],
    "unanswerable": [
        "What is the CEO's favorite color?",
        "When was the company founded in 1920?",
        "What is the specific revenue for Q3 2023?",
        "Who is the lead architect of Project Gamma?",
        "What are the specific benefits for part-time employees in the handbook?"
    ],
    "irrelevant": [
        "How do I bake a chocolate cake?",
        "Who won the World Cup in 2022?",
        "What is the distance from Earth to Mars?",
        "How does photosynthesis work?",
        "What is the capital of France?"
    ]
}

def analyze_scores(session_id="threshold_analysis"):
    results = []

    print(f"Analyzing scores for session: {session_id}...")

    for category, queries in TEST_QUERIES.items():
        for query in queries:
            # Stage 1: Retrieve
            candidates = retrieve_documents(query, session_id, top_k=15)

            if not candidates:
                results.append({"category": category, "query": query, "top_score": None, "scores": []})
                continue

            # Stage 2: Rerank
            ranked = rerank_documents(query, candidates, top_n=5)

            scores = [doc["rerank_score"] for doc in ranked]
            results.append({
                "category": category,
                "query": query,
                "top_score": scores[0] if scores else None,
                "scores": scores
            })

    return results

def print_analysis(results):
    print("\n" + "="*80)
    print(f"{'Category':<15} | {'Query':<40} | {'Top Score':<10}")
    print("-" * 80)

    category_scores = {"answerable": [], "unanswerable": [], "irrelevant": []}

    for res in results:
        cat = res["category"]
        query = res["query"][:38] + "..." if len(res["query"]) > 38 else res["query"]
        top_score = f"{res['top_score']:.4f}" if res["top_score"] is not None else "N/A"

        print(f"{cat:<15} | {query:<40} | {top_score:<10}")

        if res["top_score"] is not None:
            category_scores[cat].append(res["top_score"])

    print("\n" + "="*80)
    print("Score Distributions (Top-1)")
    print("-" * 80)
    for cat, scores in category_scores.items():
        if scores:
            avg = np.mean(scores)
            min_s = np.min(scores)
            max_s = np.max(scores)
            print(f"{cat:<15}: Mean={avg:.4f}, Min={min_s:.4f}, Max={max_s:.4f}")
        else:
            print(f"{cat:<15}: No scores collected")
    print("="*80)

if __name__ == "__main__":
    # Ensure documents are indexed for the analysis session
    # In a real scenario, we'd run ingestion first.
    # For this test, we assume 'ingestion_test' session is used as a baseline.
    # We'll map the analysis session to the ingestion_test session.

    # Note: Since get_collection creates a new collection if not exists,
    # we must use the session_id where data was actually ingested.

    # Using a dummy session_id for the function but we will
    # hardcode the one we used in the previous ingestion step.

    # To avoid modifying the function signature, we'll just call it
    # with the session_id we know has data.

    import sys
    import os

    # Setup PYTHONPATH if not already set by the runner
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

    analysis_results = analyze_scores(session_id="ingestion_test")
    print_analysis(analysis_results)
