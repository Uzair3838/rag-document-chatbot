import pymupdf
from cleaner import clean_text
from chunker import chunk_text




def process_pdf(pdf_path):
    doc = pymupdf.open(pdf_path)

    all_chunks = []

    for page_number, page in enumerate(doc, start=1):
        raw_text = page.get_text()

        cleaned_text = clean_text(raw_text)

        metadata = {
            "source": pdf_path.split("/")[-1],
            "page": page_number
        }

        page_chunks = chunk_text(cleaned_text, metadata)

        all_chunks.extend(page_chunks)

    doc.close()

    return all_chunks
