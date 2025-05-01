# 📄 DocQuest – QA with Documents

DocQuest is a simple and interactive Streamlit web app that allows users to ask questions from uploaded documents and receive relevant answers using information retrieval techniques.

## 🚀 Features

- Upload documents in **PDF, TXT, or DOCX** format
- Ask natural language questions related to the uploaded document
- Real-time **question answering** powered by embeddings and a language model
- Displays chat history of previously asked questions and answers
- Intuitive and lightweight UI with branding support

## 🖼️ Preview



## 🛠️ Technologies Used

- Python
- Streamlit
- LangChain / Gemini (embedding & LLM API)
- Document ingestion & text extraction
- Session state for chat history

## 📁 Directory Structure

QAWithPDF/ ├── data_ingestion.py # Loads and parses uploaded documents ├── embedding.py # Generates document embeddings using Gemini ├── model_api.py # Loads the LLM for answering questions main.py # Main Streamlit app script logo.png # App logo README.md requirements.txt


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

📌 Notes
You must configure your embedding and LLM API keys in the respective modules (embedding.py, model_api.py).

All uploaded documents are processed in memory and are not stored permanently.

Logo can be replaced by adding your own logo.png to the root directory.

🧑‍💻 Author
Vishal M
📫 LinkedIn
💻 GitHub
