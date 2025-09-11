# Document-QA-System 📄🤖

![GitHub release](https://img.shields.io/github/release/trieu/Document-QA-System.svg) [![Download Latest Release](https://img.shields.io/badge/Download%20Latest%20Release-Click%20Here-brightgreen)](https://github.com/trieu/Document-QA-System/releases)

**Document-QA-System** is a Streamlit-powered application that lets you ask questions directly about the content of uploaded documents. It combines **Gemini embeddings** with a **language model** to deliver fast, context-aware answers. The app supports multiple document formats, including **PDF, TXT, and DOCX**.

![screenshot](screenshot.png)

---

## 📑 Table of Contents

1. [Features](#features)
2. [Technologies Used](#technologies-used)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Contributing](#contributing)
6. [License](#license)
7. [Contact](#contact)

---

## 🚀 Features

* **Multi-Format Support** → Upload PDF, TXT, or DOCX files.
* **Fast & Accurate** → Answers powered by embeddings + LLM.
* **Streamlit UI** → Clean, interactive, and intuitive interface.
* **Interactive QA** → Ask natural language questions, get relevant responses.
* **Scalable Performance** → Handles documents of varying sizes and complexity.

---

## 🛠️ Technologies Used

* **[Streamlit](https://streamlit.io/)** → Web UI framework in Python.
* **[Gemini API](https://aistudio.google.com/)** → Document embeddings & retrieval.
* **[LangChain](https://www.langchain.com/)** → Orchestrates LLM workflows.
* **NLP** → Interprets and processes user queries.
* **Python** → Core development language.
* **PyPDF, python-docx, etc.** → For document parsing and preprocessing.

---

## ⚙️ Installation

Follow these steps to set up locally:

1. **Clone the Repository**

   ```bash
   git clone https://github.com/trieu/Document-QA-System
   cd Document-QA-System
   ```

2. **Set Environment Variable**
   Create a `.env` file and add your Google API key:

   ```bash
   nano .env
   ```

   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

   Get your key here: [Google AI Studio](https://aistudio.google.com/apikey).

3. **Create Virtual Environment**

   ```bash
   python -m venv venv
   source venv/bin/activate
   ```

4. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Run the Application**

   ```bash
   streamlit run StreamlitApp.py
   ```

   Then open [http://localhost:8501](http://localhost:8501) in your browser.

---

## 💡 Usage

1. **Upload a Document** → Choose PDF, TXT, or DOCX.
2. **Ask Questions** → Type your query in the text box.
3. **Get Answers** → The system retrieves and summarizes content in real time.

**Example Queries**:

* *“What is the main topic of this document?”*
* *“Summarize the key points.”*
* *“What conclusions are presented?”*

---

## 🤝 Contributing

We welcome contributions!

1. Fork the repository.
2. Create a feature branch.
3. Implement your changes.
4. Push to your fork.
5. Open a pull request.

---

## 📜 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

