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

QAWithPDF/ ├── data_ingestion.py # Loads and parses uploaded documents 
           ├── embedding.py # Generates document embeddings using Gemini
           ├── model_api.py # Loads the LLM for answering questions
main.py # Main Streamlit app script
logo.png # App logo README.md requirements.txt


## ▶️ Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/DocQuest.git
cd DocQuest

pip install -r requirements.txt

streamlit run main.py

📄 Sample Usage
Upload a .pdf, .txt, or .docx file.

Ask a question like "What is the main topic of the document?"

Get an instant response powered by your document content.

📌 Notes
Ensure you have access to the required embedding model API (e.g., Gemini or OpenAI).

For privacy, documents are processed in memory and not stored.

🧑‍💻 Author
Vishal – LinkedIn | GitHub

📝 License
This project is licensed under the MIT License.

yaml
Copy
Edit
