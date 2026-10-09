import chromadb

client = chromadb.PersistentClient(path='vector_db')

def get_collection():
    return client.get_or_create_collection(
        name = 'documents'
    )
    

def store_chunks(chunks, embeddings):
    collection = get_collection()
    
    documents = [chunk['text'] for chunk in chunks]
    metadatas = [chunk['metadata'] for chunk in chunks]
    
    ids = []
    
    for index, chunk in enumerate(chunks):
          source = chunk['metadata']['source']
          page = chunk['metadata']['page']
          
          chunk_id = f"{source}_page_{page}_chunk_{index}"
          ids. append(chunk_id)
          
    collection.add(
        ids= ids,
        documents= documents,
        metadatas= metadatas,
        embeddings= embeddings.tolist()
    )
