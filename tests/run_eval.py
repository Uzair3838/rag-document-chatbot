import json
import sys
import uuid
from pathlib import Path

# Add src to path to allow imports
sys.path.append("src")

from ingestion.vector_store import store_chunks, clear_collection
from ingestion.pdf_loader import process_pdf
from ingestion.embedder import create_embeddings
from rag_pipeline import answer_question
from retrieval.retriever import retrieve_documents

EVAL_SESSION_ID = f"eval_{uuid.uuid4().hex}"
EVAL_DOCS_DIR = Path("tests/eval_docs")
EVAL_DATASET_FILE = Path("tests/eval_dataset.json")
FALLBACK_ANSWER = "I don't know based on the provided documents."

def setup_eval_environment():
    print(f"Setting up evaluation environment (Session: {EVAL_SESSION_ID})...")
    clear_collection(EVAL_SESSION_ID)

    pdf_files = sorted(EVAL_DOCS_DIR.glob("*.pdf"))
    if not pdf_files:
        print("Error: No evaluation PDFs found in tests/eval_docs/")
        sys.exit(1)

    for pdf_file in pdf_files:
        print(f"  Indexing {pdf_file.name}...")
        chunks = process_pdf(str(pdf_file))
        texts = [chunk['text'] for chunk in chunks]
        embeddings = create_embeddings(texts)
        store_chunks(chunks, embeddings, EVAL_SESSION_ID)

    # Diagnostic: Verify stored chunk count
    from ingestion.vector_store import get_collection
    collection = get_collection(EVAL_SESSION_ID)
    print(f"  Verification: {collection.count()} chunks stored in collection '{collection.name}'")

    print("Ingestion complete.\n")

def verify_retrieval(question, evidence):
    if not evidence:
        return True, "N/A"

    chunks = retrieve_documents(question, EVAL_SESSION_ID, top_k=15)
    all_texts = " ".join([c['text'] for c in chunks]).lower()

    missing = [e for e in evidence if e.lower() not in all_texts]
    if not missing:
        return True, "Relevant"
    else:
        return False, f"Missing evidence: {', '.join(missing)}"

def verify_answer(answer, expected_answer, expected_fallback):
    if expected_fallback:
        if answer == FALLBACK_ANSWER:
            return True, "Correct fallback"
        else:
            return False, f"Expected fallback, got: {answer}"

    if not expected_answer:
        return False, "No expected answer provided for non-fallback case"

    # Keyword-based correctness check
    # We consider it a pass if a significant portion of the expected answer's key terms are present
    # To keep it simple and transparent, we check if key substrings are in the answer.
    # We'll split the expected answer into phrases or use a simple overlap.

    expected_lower = expected_answer.lower()
    answer_lower = answer.lower()

    # Check if the expected answer (as a whole or in parts) is reflected
    # We'll use a simple "at least one major phrase" or "keyword overlap"
    # For this small set, we'll check if the core factual part of the expected answer is in the answer.

    # Heuristic: if the answer is too short and not the fallback, it's likely wrong.
    if len(answer_lower) < 10 and answer_lower != FALLBACK_ANSWER.lower():
        return False, "Answer too short"

    # Check for keyword overlap using a simple split
    expected_keywords = [word for word in expected_lower.split() if len(word) > 3]
    answer_keywords = set(answer_lower.split())

    matches = [k for k in expected_keywords if k in answer_keywords]
    match_rate = len(matches) / len(expected_keywords) if expected_keywords else 0

    if match_rate >= 0.5: # 50% keyword overlap threshold
        return True, f"Correct ({match_rate:.1%})"
    else:
        return False, f"Low keyword overlap ({match_rate:.1%})"

def run_evaluation():
    try:
        setup_eval_environment()

        with open(EVAL_DATASET_FILE, 'r') as f:
            dataset = json.load(f)

        results = []
        total_passed = 0

        print(f"{'ID':<8} | {'Category':<15} | {'Retrieval':<12} | {'Answer':<12} | {'Status'}")
        print("-" * 70)

        for case in dataset:
            q = case['question']

            # 1. Verify Retrieval (BEFORE answer_question to avoid reranker resource interference)
            ret_passed, ret_msg = verify_retrieval(q, case['evidence'])

            # 2. RAG Pipeline
            try:
                answer, references = answer_question(q, EVAL_SESSION_ID)
            except Exception as e:
                print(f"{case['id']:<8} | {case['category']:<15} | {ret_msg:<12} | ERROR        | EXEC_ERR")
                continue

            # 3. Verify Answer
            ans_passed, ans_msg = verify_answer(answer, case['expected_answer'], case['expected_fallback'])

            status = "PASS" if (ret_passed and ans_passed) else "FAIL"
            if status == "PASS":
                total_passed += 1

            results.append({
                "id": case['id'],
                "category": case['category'],
                "retrieval": ret_msg,
                "answer": ans_msg,
                "status": status
            })

            print(f"{case['id']:<8} | {case['category']:<15} | {ret_msg:<12} | {ans_msg:<12} | {status}")

        print("-" * 70)
        print(f"Total Cases: {len(dataset)} | Passed: {total_passed} | Failed: {len(dataset) - total_passed}")

    finally:
        print(f"\nCleaning up collection for session: {EVAL_SESSION_ID}...")
        clear_collection(EVAL_SESSION_ID)

    if total_passed < len(dataset):
        print("\nEvaluation failed: some cases did not pass.")
        sys.exit(1)
    else:
        print("\nEvaluation successful: all cases passed!")
        sys.exit(0)

if __name__ == "__main__":
    run_evaluation()
