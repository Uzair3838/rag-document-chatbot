
import os
import sys
import uuid
import tempfile
from pathlib import Path

import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"

sys.path.insert(0, str(SRC_DIR))

from rag_pipeline import answer_question
from ingestion.pdf_loader import process_pdf
from ingestion.embedder import create_embeddings
from ingestion.vector_store import (
    store_chunks,
    get_collection,
    clear_collection,
)


# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="DocuMind | RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)


# --------------------------------------------------
# SESSION INITIALIZATION
# --------------------------------------------------

if "session_id" not in st.session_state:
    st.session_state.session_id = uuid.uuid4().hex

if "messages" not in st.session_state:
    st.session_state.messages = []


session_id = st.session_state.session_id


# --------------------------------------------------
# COLOR THEME: OLIVE + WINE RED + YELLOW
# --------------------------------------------------

st.markdown(
    """
    <style>
    :root {
        --olive: #343B24;
        --olive-light: #454D32;
        --olive-border: #626B47;
        --wine: #58181F;
        --wine-light: #74242D;
        --yellow: #E7C76A;
        --cream: #F7F0E3;
        --muted: #D6D9C7;
    }

    .stApp,
    [data-testid="stAppViewContainer"] {
        background: var(--olive);
        color: var(--cream);
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    [data-testid="stSidebar"] {
        background: #252C1B;
        border-right: 1px solid var(--olive-border);
    }

    [data-testid="stSidebar"] * {
        color: var(--cream);
    }

    h1, h2, h3 {
        color: var(--yellow) !important;
        letter-spacing: -0.4px;
    }

    p, label, li {
        color: var(--cream);
    }

    .app-hero {
        background: linear-gradient(
            135deg,
            #252C1B 0%,
            #454D32 100%
        );
        border: 1px solid var(--olive-border);
        border-left: 5px solid var(--yellow);
        border-radius: 18px;
        padding: 28px 30px;
        margin: 8px 0 22px 0;
        box-shadow: 0 8px 24px rgba(0, 0, 0, 0.14);
    }

    .app-hero h1 {
        color: var(--yellow) !important;
        margin: 0 0 8px 0;
        font-size: 2.15rem;
    }

    .app-hero p {
        color: var(--cream);
        margin: 0;
        font-size: 1rem;
        line-height: 1.6;
    }

    .info-card {
        background: #252C1B;
        border: 1px solid var(--olive-border);
        border-radius: 14px;
        padding: 17px 20px;
        margin: 10px 0 20px 0;
        line-height: 1.7;
    }

    .info-card strong {
        color: var(--yellow);
    }

    [data-testid="stChatMessage"] {
        background: var(--olive-light);
        border: 1px solid var(--olive-border);
        border-radius: 16px;
        padding: 16px;
        margin-bottom: 12px;
    }

    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] span {
        color: var(--cream) !important;
    }

    [data-testid="stChatInput"] {
        background: var(--cream);
        border: 2px solid var(--yellow);
        border-radius: 14px;
    }

    [data-testid="stChatInput"] textarea {
        color: #252C1B !important;
        background: var(--cream) !important;
        -webkit-text-fill-color: #252C1B !important;
        caret-color: var(--wine);
    }

    [data-testid="stChatInput"] textarea::placeholder {
        color: #666B59 !important;
        -webkit-text-fill-color: #666B59 !important;
    }

    .stButton > button {
        background: var(--wine);
        color: #FFFFFF !important;
        border: 1px solid #8A454B;
        border-radius: 11px;
        padding: 0.55rem 1rem;
        font-weight: 600;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        background: var(--wine-light);
        border-color: var(--yellow);
        color: #FFFFFF !important;
    }

    [data-testid="stFileUploader"] {
        background: #343B24;
        border: 1px dashed var(--yellow);
        border-radius: 12px;
        padding: 8px;
    }

    [data-testid="stFileUploader"] section {
        background: transparent;
    }

    [data-testid="stFileUploader"] button {
        background: var(--wine);
        color: white !important;
        border-radius: 8px;
    }

    [data-testid="stAlert"] {
        border-radius: 12px;
    }

    a {
        color: var(--yellow) !important;
    }

    hr {
        border-color: var(--olive-border);
    }

    small, .stCaption {
        color: var(--muted) !important;
    }

    .source-label {
        color: var(--yellow);
        font-weight: 600;
        margin-top: 10px;
    }

    [data-testid="stStatusWidget"] {
        background: var(--olive-light);
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:
    st.markdown(
        "<h2>📚 DocuMind</h2>",
        unsafe_allow_html=True,
    )

    st.caption("Your private document assistant")

    st.markdown("---")

    st.markdown("### Upload your documents")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload the documents you want to ask questions about.",
    )

    if st.button("Process Uploaded PDFs", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            total_chunks = 0
            failed_files = []

            with st.spinner("Processing your PDF documents..."):
                for uploaded_file in uploaded_files:
                    temp_path = None

                    try:
                        with tempfile.NamedTemporaryFile(
                            suffix=".pdf",
                            delete=False,
                        ) as temp_file:
                            temp_file.write(uploaded_file.getvalue())
                            temp_path = temp_file.name

                        chunks = process_pdf(temp_path)

                        for chunk in chunks:
                            chunk["metadata"]["source"] = (
                                uploaded_file.name
                            )

                        if not chunks:
                            failed_files.append(uploaded_file.name)
                            st.warning(
                                f"No extractable text found in "
                                f"{uploaded_file.name}."
                            )
                            continue

                        texts = [chunk["text"] for chunk in chunks]
                        embeddings = create_embeddings(texts)

                        store_chunks(
                            chunks=chunks,
                            embeddings=embeddings,
                            session_id=session_id,
                        )

                        total_chunks += len(chunks)

                    except Exception as error:
                        failed_files.append(uploaded_file.name)
                        st.error(
                            f"Could not process {uploaded_file.name}: "
                            f"{error}"
                        )

                    finally:
                        if temp_path and os.path.exists(temp_path):
                            os.remove(temp_path)

            if total_chunks > 0:
                st.success(
                    f"Indexed {total_chunks} chunks from your PDFs."
                )

            if failed_files:
                st.warning(
                    "Some files could not be processed: "
                    + ", ".join(failed_files)
                )

    # Display the current session's document status.
    document_count = get_collection(session_id).count()

    st.markdown("---")
    st.markdown("### Current session")

    if document_count > 0:
        st.success(f"{document_count} document chunks indexed")
    else:
        st.info("No documents uploaded yet")

    st.markdown("---")
    st.markdown("### About")
    st.write(
        "Upload PDFs and ask questions answered using their contents."
    )

    st.markdown("---")

    if st.button(
        "Clear Documents & Chat",
        use_container_width=True,
    ):
        clear_collection(session_id)
        st.session_state.messages = []
        st.rerun()


# --------------------------------------------------
# MAIN HEADER
# --------------------------------------------------

st.markdown(
    """
    <div class="app-hero">
        <h1>Document Intelligence</h1>
        <p>
            Your documents. Your questions. Answers grounded in
            your uploaded PDFs, with page references.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="info-card">
        <strong>How it works</strong><br>
        Upload PDFs → Extract and index text → Ask a question →
        Retrieve relevant passages → Get an answer with sources.
    </div>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------
# CHAT HISTORY
# --------------------------------------------------

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("references"):
            st.markdown("**Sources**")

            for reference in message["references"]:
                st.caption(f"📄 {reference}")


# --------------------------------------------------
# QUESTION INPUT
# --------------------------------------------------

question = st.chat_input("Ask about your uploaded documents...")

if question:
    st.session_state.messages.append(
        {
            "role": "user",
            "content": question,
        }
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        # Do not search or call the LLM before documents are uploaded.
        if get_collection(session_id).count() == 0:
            answer = (
                "Please upload and process at least one PDF before "
                "asking questions."
            )
            references = []

            st.info(answer)

        else:
            with st.spinner(
                "Searching your documents and generating an answer..."
            ):
                try:
                    answer, references = answer_question(
                        question=question,
                        session_id=session_id,
                    )

                    st.markdown(answer)

                    if references:
                        st.markdown("**Sources**")

                        for reference in references:
                            st.caption(f"📄 {reference}")

                except Exception:
                    answer = (
                        "Something went wrong while answering your "
                        "question. Please check the application "
                        "configuration and try again."
                    )
                    references = []

                    st.error(answer)

        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer,
                "references": references,
            }
        )
