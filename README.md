# 📄 DocQuest – QA with Documents

DocQuest is a simple and interactive Streamlit web app that allows users to ask questions from uploaded documents and receive relevant answers using information retrieval techniques.

## 🚀 Features

- Upload documents in **PDF, TXT, or DOCX** format
- Ask natural language questions related to the uploaded document
- Real-time **question answering** powered by embeddings and a language model
- Displays chat history of previously asked questions and answers
- Intuitive and lightweight UI with branding support

## 🖼️ Preview

![image](https://github.com/user-attachments/assets/e2f0b81f-0c42-4910-8361-be8e623e13d3)


## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain / Gemini (embedding & LLM API)
- Document ingestion & text extraction
- Session state for chat history

## 📁 Directory Structure

```bash
QAWithPDF/
├── data_ingestion.py      # Loads and parses uploaded documents
├── embedding.py           # Generates document embeddings using Gemini
├── model_api.py           # Loads the LLM for answering questions
StreamlitApp.py            # Main Streamlit app script
logo.png                   # App logo
README.md
requirements.txt
```


## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/vishal220703/Document-QA-System.git
cd LLM Project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```
### 3. Run the App

```bash
streamlit run main.py
```

### 📌 Notes

1. You must configure your embedding and LLM API keys in the respective modules (embedding.py, model_api.py).
2. All uploaded documents are processed in memory and are not stored permanently.
3. Logo can be replaced by adding your own logo.png to the root directory.

🧑‍💻 Author- Vishal M
📫 LinkedIn
💻 GitHub
