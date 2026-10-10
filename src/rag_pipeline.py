from retrieval.retriever import retrieve_documents
from retrieval.context_builder import build_context
from llm import generate_answer
from retrieval.reranker import rerank_documents
from pathlib import Path

def answer_question(question):
    
    candidate_documents = retrieve_documents(
        query= question,
        top_k= 15
    )
    
    retrieved_documents = rerank_documents(
        query= question,
        documents= candidate_documents,
        top_n= 5
    )
    
   
    #build context from retrieved chunks
    context = build_context(retrieved_documents=retrieved_documents)
    
    #create the prompt for llm
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
    
    #generate the final answer
    answer = generate_answer(prompt= prompt)
    
    references = []
    
    for document in retrieved_documents:
        reference = f"{Path(document['source']).name}, Page {document['page']}"
        
        if reference not in references:
            references.append(reference)
    
    if "I don't know based on the provided documents." in answer:
        references = []
    
    return answer, references



if __name__ == "__main__":
    question = "What products does NexaTech Solutions offer, and what does each product do?"

    answer, references = answer_question(question=question)

    print("\n" + "=" * 60)
    print(question)
    print("\n" + "=" * 60)
    print("\nAnswer:")
    print(answer)
    print("\nReferences:")

    for reference in references:
        print(f"- {reference}")
