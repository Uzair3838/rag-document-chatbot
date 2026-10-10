
import streamlit as st

from src.rag_pipeline import answer_question


st.set_page_config(
    page_title="DocuMind | RAG Chatbot",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    
    <style>
    /* Main application background */
    .stApp {
        background: #343B24;
        color: #F7F0E3;
    }

    /* Main content area */
    [data-testid="stAppViewContainer"] {
        background: #343B24;
    }

    [data-testid="stHeader"] {
        background: transparent;
    }

    /* Headings and normal text */
    h1, h2, h3, p, label, span, li {
        color: #F7F0E3;
    }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: #252C1B;
    }

    [data-testid="stSidebar"] * {
        color: #F7F0E3;
    }

    /* Chat messages */
    [data-testid="stChatMessage"] {
        background: #454D32;
        border: 1px solid #626B47;
        border-radius: 16px;
        padding: 18px;
        color: #F7F0E3;
    }

    /* Make all text inside chat messages readable */
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] li {
        color: #F7F0E3 !important;
    }

    /* Question input container */
    [data-testid="stChatInput"] {
        background: #F7F0E3;
        border: 2px solid #58181F;
        border-radius: 16px;
    }

    /* Question input text */
    [data-testid="stChatInput"] textarea {
        color: #252C1B !important;
        background: #F7F0E3 !important;
        caret-color: #58181F;
        -webkit-text-fill-color: #252C1B !important;
    }

    /* Input placeholder */
    [data-testid="stChatInput"] textarea::placeholder {
        color: #666B59 !important;
        -webkit-text-fill-color: #666B59 !important;
    }

    /* Buttons */
    .stButton > button {
        background: #58181F;
        color: #FFFFFF !important;
        border: 1px solid #7A3038;
        border-radius: 12px;
    }

    .stButton > button:hover {
        background: #74242D;
        border-color: #F0D9C9;
        color: #FFFFFF !important;
    }

    /* Links */
    a {
        color: #E7C38A !important;
    }

    /* Dividers */
    hr {
        border-color: #626B47;
    }

    /* Code blocks */
    [data-testid="stCode"] {
        background: #252C1B;
    }

    /* Sources and smaller supporting text */
    small, .stCaption {
        color: #D6D9C7 !important;
    }

    /* Text input and other form fields */
    .stTextInput input,
    .stTextArea textarea {
        background: #F7F0E3 !important;
        color: #252C1B !important;
        -webkit-text-fill-color: #252C1B !important;
        border: 1px solid #8B916E;
        border-radius: 10px;
    }
    </style>

    """,
    unsafe_allow_html=True,
)


if "messages" not in st.session_state:
    st.session_state.messages = []


with st.sidebar:
    st.markdown("## 📚 DocuMind")
    st.caption("Your personal document assistant")

    st.markdown("---")

    st.markdown("### About")
    st.write(
        "Ask questions about your indexed PDF documents. "
        "Answers are generated using retrieved document context."
    )

    st.markdown("---")

    st.markdown("### Current capabilities")
    st.write("• Semantic search")
    st.write("• Reranked retrieval")
    st.write("• Source references")
    st.write("• Context-based answers")

    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


st.markdown(
    """
    <div class="neo-card">
        <h1>Document Intelligence</h1>
        <p class="app-subtitle">
            Ask questions, explore your documents, and get answers
            supported by PDF references.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


with st.container():
    st.markdown(
        """
        <div class="neo-inset">
            <strong>How it works</strong><br>
            Your question → Relevant document chunks → Reranking →
            AI-generated answer with sources
        </div>
        """,
        unsafe_allow_html=True,
    )


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("references"):
            st.markdown("**Sources**")
            for reference in message["references"]:
                st.caption(f"📄 {reference}")


question = st.chat_input("Ask a question about your documents...")

if question:
    st.session_state.messages.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating an answer..."):
            try:
                answer, references = answer_question(question=question)

                st.markdown(answer)

                if references:
                    st.markdown("**Sources**")
                    for reference in references:
                        st.caption(f"📄 {reference}")

                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "references": references,
                    }
                )

            except Exception as error:
                st.error(
                    "Something went wrong while generating the answer. "
                    "Check that the required models and LLM server are available."
                )
                st.exception(error)
