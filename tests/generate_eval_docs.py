from fpdf import FPDF
from pathlib import Path

def create_test_pdf(filename, content):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("helvetica", size=12)
    for line in content:
        pdf.multi_cell(w=190, h=10, text=line)

    output_path = Path("tests/eval_docs") / filename
    pdf.output(str(output_path))
    print(f"Created {output_path}")

def generate_all_docs():
    docs = {
        "doc_a.pdf": [
            "Company Policy: Remote Work",
            "The company supports a hybrid work model.",
            "Employees are allowed to work from home up to 3 days a week.",
            "Core hours are 10am to 3pm EST."
        ],
        "doc_b.pdf": [
            "Financial Report 2023",
            "The total revenue for 2023 was 10 million dollars.",
            "Growth was 5% compared to 2022."
        ],
        "doc_c.pdf": [
            "Financial Report 2024",
            "The total revenue target for 2024 is 15 million dollars.",
            "Expected growth is 10%."
        ],
        "long_doc.pdf": [
            "This is a very long document for testing retrieval.",
            "Section 1: Introduction",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Section 2: History",
            "The history of the project started in 2010.",
            "More filler text to ensure the document is long.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Section 3: The Secret Fact",
            "The secret code for the vault is 12345.",
            "This fact is hidden deep in the document.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit.",
            "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
        ]
    }

    for filename, content in docs.items():
        create_test_pdf(filename, content)

if __name__ == "__main__":
    generate_all_docs()
