import sys
import uuid
from pathlib import Path

# Add src to path to allow imports
sys.path.append("src")

from ingestion.vector_store import store_chunks, clear_collection, get_collection
from ingestion.pdf_loader import process_pdf
from ingestion.embedder import create_embeddings, model as embedder_model
from retrieval.retriever import retrieve_documents
from retrieval.reranker import rerank_documents

# Use a fixed session ID for this diagnostic run
DIAG_SESSION_ID = "diag_retrieval_123"
EVAL_DOCS_DIR = Path("tests/eval_docs")

def setup():
    clear_collection(DIAG_SESSION_ID)
    pdf_files = sorted(EVAL_DOCS_DIR.glob("*.pdf"))
    for pdf_file in pdf_files:
        chunks = process_pdf(str(pdf_file))
        texts = [chunk['text'] for chunk in chunks]
        embeddings = create_embeddings(texts)
        store_chunks(chunks, embeddings, DIAG_SESSION_ID)
    print(f"Setup complete. Collection: {get_collection(DIAG_SESSION_ID).name}")

def diagnose():
    query = "What is the company's remote work policy?"
    print(f"\nQuery: {query}")
    
    # 1. Query Embedding
    q_emb = embedder_model.encode([query])
    print(f"\n[1] Query Embedding: Shape={q_emb.shape}, Type={type(q_emb)}")

    # 2. Raw ChromaDB Query
    collection = get_collection(DIAG_SESSION_ID)
    raw_results = collection.query(
        query_embeddings=q_emb.tolist(),
        n_results=10,
        include=["documents", "metadatas", "distances"]
    )
    
    print("\n[2] Raw ChromaDB Results:")
    docs = raw_results["documents"][0]
    metas = raw_results["metadatas"][0]
    dists = raw_results["distances"][0]
    
    if not docs:
        print("  No documents returned by ChromaDB.")
    for i in range(len(docs)):
        print(f"  Result {i+1}: Dist={dists[i]:.4f} | Source={metas[i]['source']} | Text={docs[i][:100]}...")

    # 3. Retriever Output
    ret_docs = retrieve_documents(query, DIAG_SESSION_ID, top_k=15)
    print(f"\n[3] Retriever Output: Returned {len(ret_docs)} documents.")
    for i, d in enumerate(ret_docs):
        print(f"  Doc {i+1}: Dist={d['distance']:.4f} | Text={d['text'][:100]}...")

    # 4. Reranker Output
    if ret_docs:
        ranked = rerank_documents(query, ret_docs)
        print("\n[4] Reranker Output:")
        for i, d in enumerate(ranked):
            print(f"  Rank {i+1}: Score={d['rerank_score']:.4f} | Text={d['text'][:100]}...")
    else:
        print("\n[4] Reranker Output: Skipped (no candidate documents).")

    # 5. Final RAG Pipeline Simulation
    if 'ranked' in locals():
        final_docs = [d for d in ranked if d["rerank_score"] > 0]
        print(f"\n[5] Final Pipeline Filter (score > 0): {len(final_docs)} documents remaining.")
    else:
        print("\n[5] Final Pipeline Filter: 0 documents remaining.")

if __name__ == "__main__":
    try:
        setup()
        diagnose()
    finally:
        clear_collection(DIAG_SESSION_ID)
