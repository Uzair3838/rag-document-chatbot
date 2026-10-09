from retrieval.retriever import retrieve_documents
from retrieval.context_builder import build_context
from llm import generate_answer

def answer_question(question):
    
    #retrieve relevent documents chunks
    retrieved_documents = retrieve_documents(
        query= question,
        top_k= 3
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
    
    refernces = []
    
    for document in retrieved_documents:
        reference = f"{document['source']}, Page {document['page']}"
        
        if reference not in refernces:
            refernces.append(reference)
    
    return answer, refernces


question = "What are the company's main products and what are their prices?"

answer, references = answer_question(question= question)

print("\nAnswer:")
print(answer)

print("\nReference:")

for reference in references:
    print(f"- {reference}")    
    
    