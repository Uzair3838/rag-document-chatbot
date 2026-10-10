from retrieval.retriever import retrieve_documents
from retrieval.context_builder import build_context
from llm import generate_answer

def answer_question(question):
    
    #retrieve relevent documents chunks
    retrieved_documents = retrieve_documents(
        query= question,
        top_k= 8
    )
    
    #temporary code to see the retrieved chunks
    print("\n--- RETRIEVED CHUNKS ---")

    for i, doc in enumerate(retrieved_documents, start=1):
        print(f"\nChunk {i}")
        print(f"Source: {doc['source']}")
        print(f"Page: {doc['page']}")
        print(f"Distance: {doc['distance']:.4f}")
        print(f"Text:\n{doc['text']}")
        print("-" * 50)
    
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
        reference = f"{document['source']}, Page {document['page']}"
        
        if reference not in references:
            references.append(reference)
    
    if "I don't know based on the provided documents." in answer:
        references = []
    
    return answer, references


question = "What products does NexaTech Solutions offer, and what does each product do?"

answer, references = answer_question(question= question)

print("\n" + "="*60)
print(question)
print("\n" + "="*60)


print("\nAnswer:")
print(answer)

if references:
    print("\nReference:")

    for reference in references:
        print(f"- {reference}")    
    
    