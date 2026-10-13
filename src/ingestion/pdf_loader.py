import pymupdf
from .cleaner import clean_text
from .chunker import chunk_text
from pathlib import Path



def process_pdf(pdf_path):
    try:
        doc = pymupdf.open(pdf_path)
    except Exception as e:
        raise ValueError(f"Invalid or corrupted PDF file: {e}")

    all_chunks = []

    try:
        for page_number, page in enumerate(doc, start=1):
            raw_text = page.get_text()

            cleaned_text = clean_text(raw_text)

            metadata = {
                "source": Path(pdf_path).name,
                "page": page_number
            }

            page_chunks = chunk_text(cleaned_text, metadata)

            all_chunks.extend(page_chunks)
    finally:
        doc.close()

    return all_chunks
