def build_context(retrieved_documents):
    context_parts = []
    
    for doc in retrieved_documents:
        context_parts.append(
            f"Source: {doc['source']},"
            f"Page: {doc['page']}\n"
            f"{doc['text']}"
        )
        
    return "\n\n".join(context_parts)


