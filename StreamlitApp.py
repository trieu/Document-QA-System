import streamlit as st
import streamlit_authenticator as stauth

from QAWithPDF.data_ingestion import load_data
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.model_api import load_model

from user_management import init_db, get_users, list_users


def main():
    st.set_page_config("DocQuest", page_icon="📄")
    init_db()

    st.sidebar.image("logo.png", width=100)
    st.sidebar.title("🔐 Authentication")

    # Load users from DB
    users = get_users()

    # Create authenticator
    authenticator = stauth.Authenticate(
        credentials={"usernames": users},
        cookie_name="docquest_auth",
        key="abcdef",
        cookie_expiry_days=1,
    )

    authenticator.login(location="sidebar")

    auth_status = st.session_state.get("authentication_status")
    username = st.session_state.get("username")

    if auth_status:
        st.sidebar.success(f"Welcome, {username}")
        authenticator.logout("Logout", "sidebar")

        # ===== Chatbot UI =====
        st.markdown("<h1 style='color: #4A90E2;'>DocQuest</h1>", unsafe_allow_html=True)

        if "history" not in st.session_state:
            st.session_state.history = []

        doc = st.file_uploader("📄 Upload your document", type=["pdf", "txt", "docx"])
        user_question = st.text_input("💬 Ask your question")

        if st.button("Submit & Process"):
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

                    st.session_state.history.append(
                        (user_question, response.response)
                    )
                    st.write(response.response)

        if st.session_state.history:
            st.markdown("### 📜 Previous Questions & Answers")
            for i, (q, a) in enumerate(st.session_state.history):
                st.markdown(f"**Q{i+1}:** {q}")
                st.markdown(f"**A{i+1}:** {a}")

        # ===== User Management (Admin only) =====
        if users.get(username, {}).get("role") == "admin":
            st.subheader("👥 User Management (Admin)")
            rows = list_users()
            for u, r, created in rows:
                st.write(f"**{u}** | Role: {r} | Created: {created}")

    elif auth_status is False:
        st.error("❌ Username/password incorrect")
    else:
        st.warning("Please login to continue")


if __name__ == "__main__":
    main()
