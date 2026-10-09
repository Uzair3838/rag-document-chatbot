# RAG Document Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers questions using information extracted from PDF documents. It retrieves relevant text chunks from a local vector database and uses an LLM to generate answers grounded in the provided documents.

## Features

- PDF text extraction using PyMuPDF
- Text cleaning and chunking with overlap
- Semantic embeddings using Sentence Transformers
- Vector storage and similarity search with ChromaDB
- LLM-powered answers using Ollama and LangChain
- Source references to help trace answers back to PDF documents
- Designed to acknowledge when the provided documents do not contain enough information

## Architecture

1. **PDF ingestion:** Extract text and page metadata from PDF files.
2. **Text preprocessing:** Clean extracted text and split it into overlapping chunks.
3. **Embedding generation:** Convert text chunks into numerical vectors.
4. **Vector storage:** Store chunks, metadata, and embeddings in ChromaDB.
5. **Retrieval:** Find relevant chunks for a user's question.
6. **Answer generation:** Pass the retrieved context to an LLM to generate a grounded answer.
7. **Source references:** Display the source document and page number.

## Tech Stack

- Python
- LangChain
- Sentence Transformers (`all-MiniLM-L6-v2`)
- ChromaDB
- PyMuPDF
- Ollama Cloud (`gpt-oss:20b-cloud`)

## Project Structure

```text
rag-document-chatbot/
├── data/
│   └── documents/          # Place PDF documents here
├── src/
│   ├── ingestion/
│   │   ├── pdf_loader.py
│   │   ├── cleaner.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── ingest_all.py
│   ├── retrieval/
│   │   ├── retriever.py
│   │   ├── context_builder.py
│   │   └── test_retrieval.py
│   ├── llm.py
│   └── rag_pipeline.py
├── requirements.txt
└── .gitignore
```

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/Uzair3838/rag-document-chatbot.git
cd rag-document-chatbot
```

### 2. Create and activate a virtual environment

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4. Add PDF documents

Create a `data/documents/` directory in the project root and place your PDF files inside it.

### 5. Start Ollama

Install and sign in to [Ollama](https://ollama.com/) and ensure its service is running. This project uses the `gpt-oss:20b-cloud` model, which requires the appropriate Ollama Cloud access.

### 6. Ingest the documents

Run the ingestion script from the project root:

```powershell
python src/ingestion/ingest_all.py
```

### 7. Ask questions

Run the RAG pipeline:

```powershell
python src/rag_pipeline.py
```

The current pipeline uses the question defined in the source code. Update that question to test other queries.

## Current Limitations

- The current interface is command-line based.
- PDF documents must be ingested before they can be searched.
- The vector database is stored locally and is not included in the repository.
- Answer quality depends on the source documents, retrieval results, and language model.
- The project is under active development.

## Future Improvements

- Build a PDF upload and chat interface with Streamlit.
- Add hybrid search and reranking.
- Support metadata filters and multiple documents.
- Add automated evaluation for retrieval accuracy and answer correctness.
- Improve conversation history and document management.

## Privacy

PDF files and the local vector database are excluded from Git tracking by default. Do not upload confidential documents or credentials to a public repository.

## Author

**Uzair Mustafa**

GitHub: [@Uzair3838](https://github.com/Uzair3838)