import streamlit as st
from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

def main():
    st.set_page_config("QA with Documents", page_icon="📄")

    # Create two columns: one for the image and one for the title
    col1, col2 = st.columns([1, 3])

    with col1:
        st.image("logo.png")  # Display the logo in its original size

    with col2:
        st.markdown("<h1 style='color: #4A90E2;'>DocQuest</h1>", unsafe_allow_html=True)

    if "history" not in st.session_state:
        st.session_state.history = []

    doc = st.file_uploader("Upload your document", type=["pdf", "txt", "docx"])

    st.header("QA with Documents (Information Retrieval)")

    user_question = st.text_input("Ask your question")

    if st.button("submit & process"):
        if doc is None:
            st.warning("Please upload a document before submitting.")
        elif not user_question.strip():
            st.warning("Please enter a question.")
        else:
            with st.spinner("Processing..."):
                document = load_data(doc)
                model = load_model()
                query_engine = download_gemini_embedding(model, document)
                response = query_engine.query(user_question)

                st.session_state.history.append((user_question, response.response))
                st.write(response.response)

    if st.session_state.history:
        st.markdown("### Previous Questions & Answers")
        for i, (q, a) in enumerate(st.session_state.history):
            st.markdown(f"**Q{i+1}:** {q}")
            st.markdown(f"**A{i+1}:** {a}")

if __name__ == "__main__":
    main()
