import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

# Custom CSS for styling
def local_css():
    st.markdown("""
    <style>
    .main {
        background-color: #f7f9fc;
        font-family: 'Segoe UI', sans-serif;
    }
    .title {
        text-align: center;
        color: #1f77b4;
        padding-top: 10px;
        font-size: 2.5em;
        font-weight: 700;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1.1em;
        margin-top: -10px;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        border-radius: 5px;
        padding: 0.5em 1.5em;
        font-size: 1em;
    }
    .stTextInput>div>input {
        padding: 10px;
        font-size: 1em;
    }
    .stFileUploader {
        margin-top: 20px;
    }
    .response-box {
        background-color: #ffffff;
        border-left: 4px solid #1f77b4;
        padding: 15px;
        margin-top: 20px;
        border-radius: 5px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="QA with Documents", page_icon="📄", layout="wide")
    local_css()

    st.markdown('<div class="title">📄 QA with Documents</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Upload a document and ask questions to retrieve information</div>', unsafe_allow_html=True)
    st.markdown("---")

    col1, col2 = st.columns([1, 2])
    with col1:
        doc = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])
    with col2:
        user_question = st.text_input("💬 Ask your question")

    if st.button("Submit & Process"):
        if doc is None or user_question.strip() == "":
            st.warning("Please upload a document and enter a question.")
        else:
            with st.spinner("🔍 Processing your document and querying..."):
                document = load_data(doc)
                model = load_model()
                query_engine = download_gemini_embedding(model, document)
                response = query_engine.query(user_question)
                st.markdown(f'<div class="response-box">{response.response}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
