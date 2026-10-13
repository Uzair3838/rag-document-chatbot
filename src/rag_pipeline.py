
from pathlib import Path

from retrieval.retriever import retrieve_documents
from retrieval.context_builder import build_context
from retrieval.reranker import rerank_documents
from llm import generate_answer


def answer_question(question, session_id):
    candidate_documents = retrieve_documents(
        query=question,
        session_id=session_id,
        top_k=15
    )

    if not candidate_documents:
        return "I don't know based on the provided documents.", []

    retrieved_documents = rerank_documents(
        query=question,
        documents=candidate_documents,
        top_n=5
    )

    # Keep only documents with positive reranking scores.
    retrieved_documents = [
        document
        for document in retrieved_documents
        if document["rerank_score"] > 0
    ]

    if not retrieved_documents:
        return "I don't know based on the provided documents.", []

    context = build_context(
        retrieved_documents=retrieved_documents
    )

    prompt = f"""
    You are a document question-answering assistant.

    Answer the user's question using only the provided context.

    If the answer cannot be found in the context, say:
    "I don't know based on the provided documents."

    Do not use outside knowledge.

    Context:
    {context}

    Question:
    {question}

    Answer:
    """

    answer = generate_answer(prompt=prompt)

    references = []

    for document in retrieved_documents:
        reference = (
            f"{Path(document['source']).name}, "
            f"Page {document['page']}"
        )

        if reference not in references:
            references.append(reference)

    if "I don't know based on the provided documents." in answer:
        references = []

    return answer, references
