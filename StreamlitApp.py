
# core libs 
import bleach
import streamlit_authenticator as stauth
import streamlit as st
import logging
import sys
from dotenv import load_dotenv

# force reload .env file
load_dotenv(override=True) 

# set logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)

# app libs 
from user_management import UserDao
from user_chat_history import UserChatDao
from QAWithPDF.model_api import load_model
from QAWithPDF.embedding import download_gemini_embedding
from QAWithPDF.data_ingestion import load_data

# header 
HEADER_STR = "<h1 style='color: #4A90E2;'>DocQuest</h1>"

# Inject global CSS
HIDE_MENU_STYLE = """ 
    <style> 
    /* Hide the default file size/type instruction */
    #MainMenu, .stAppDeployButton {
        display: none !important;
    } 
    </style> """


def main():
    st.set_page_config("DocQuest", page_icon="📄")
    user_dao = UserDao()
    chat_dao = UserChatDao()

    st.sidebar.image("logo.png", width=100)
    st.sidebar.title("🔐 Authentication")

    # Load users from DB
    users = user_dao.get_users()

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
        st.markdown(HIDE_MENU_STYLE, unsafe_allow_html=True)
        st.markdown(HEADER_STR, unsafe_allow_html=True)

        # Load QA history for this user
        if "history" not in st.session_state:
            st.session_state.history = chat_dao.load_history(username)


        # Add your own message above or below
        st.markdown("**Upload your document and ask questions**")

        # File uploader
        doc = st.file_uploader(
            "📄 Upload your document",
            type=["pdf", "txt", "docx"], help=None
        )


        # File size check (10 MB max)
        if doc is not None and doc.size > 10 * 1024 * 1024:
            st.error("❌ File size exceeds 10 MB. Please upload a smaller file.")
            doc = None  # reset

        user_question = st.text_area(label="💬 Ask your question", height=120)
        user_question_str = user_question.strip()

        if st.button("Submit & Process"):
            if doc is None:
                st.warning("Please upload a document before submitting.")
            elif len(user_question_str) == 0:
                st.warning("Please enter a question.")
            else:
                with st.spinner("Processing..."):
                    document = load_data(doc)
                    model = load_model()
                    query_engine = download_gemini_embedding(model, document)
                    response = query_engine.query(user_question_str)

                    # ✅ Save only the latest Q&A to SQLite
                    chat_dao.save_qa(
                        username, user_question_str, response.response)

                    # ✅ Update session memory
                    st.session_state.history.append(
                        (user_question_str, response.response))

                    st.write(bleach.clean(response.response))

        if st.session_state.history:
            st.markdown("### 📜 Previous Questions & Answers")
            for i, (q, a) in enumerate(st.session_state.history):
                st.markdown(f"**Q{i+1}:** {bleach.clean(q)}")
                st.markdown(f"**A{i+1}:** {bleach.clean(a)}")

        # ===== User Management (Admin only) =====
        if users.get(username, {}).get("role") == "admin":
            st.subheader("👥 User Management (Admin)")
            rows = user_dao.list_users()
            for u, r, created in rows:
                st.write(f"**{u}** | Role: {r} | Created: {created}")

    elif auth_status is False:
        st.error("❌ Username or password incorrect")
    else:
        st.warning("Please login to continue")


if __name__ == "__main__":
    main()
