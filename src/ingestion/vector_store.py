
import uuid
import chromadb

client = chromadb.EphemeralClient()


def get_collection(session_id):
    collection_name = f"documents_{session_id.replace('-', '')}"

    return client.get_or_create_collection(
        name=collection_name
    )


def store_chunks(chunks, embeddings, session_id):
    if not chunks:
        return

    collection = get_collection(session_id)

    documents = [chunk["text"] for chunk in chunks]
    metadatas = [chunk["metadata"] for chunk in chunks]

    # Remove previously indexed chunks from the same PDF.
    sources = {metadata["source"] for metadata in metadatas}

    for source in sources:
        collection.delete(
            where={"source": source}
        )

    # Create unique IDs for the new chunks.
    ids = [
        uuid.uuid4().hex
        for _ in chunks
    ]

    collection.add(
        ids=ids,
        documents=documents,
        metadatas=metadatas,
        embeddings=embeddings.tolist()
    )


def clear_collection(session_id):
    collection_name = f"documents_{session_id.replace('-', '')}"

    try:
        client.delete_collection(name=collection_name)
    except Exception:
        pass
