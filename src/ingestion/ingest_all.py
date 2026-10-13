from pathlib import Path

from pdf_loader import process_pdf
from embedder import create_embeddings
from vector_store import store_chunks

docs_folder = Path('data/documents')

pdf_files = sorted(docs_folder.glob('*.pdf'))

total_chunks = 0

for pdf_file in pdf_files:
    print(f"\n processing: {pdf_file.name}")
    
    chunks = process_pdf(str(pdf_file))
    
    texts = [chunk['text'] for chunk in chunks]
    
    embeddings = create_embeddings(texts)
    
    store_chunks(
        chunks = chunks,
        embeddings= embeddings,
        session_id= 'ingestion_test'
        )
    
    total_chunks += len(chunks)
    
    print(f"Chunks created: {len(chunks)}")
    print(f"Embeddings shape: {embeddings.shape}")
    

print(f"\nTotal PDFs processed: {len(pdf_files)}")
print(f"Total chunks stored: {total_chunks}")