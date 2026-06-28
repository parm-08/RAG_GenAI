# 📄 PDF-Based RAG Assistant

A Retrieval-Augmented Generation (RAG) application that allows users to upload PDF documents and interact with them using natural language. Instead of manually searching through lengthy documents, users can simply ask questions and receive accurate, context-aware answers powered by Large Language Models (LLMs).

---

## 🚀 Features

* 📂 Upload one or multiple PDF documents
* ✂️ Automatically split documents into manageable chunks
* 🧠 Generate semantic embeddings using Hugging Face models
* 🗄️ Store embeddings persistently in ChromaDB
* 🔍 Retrieve the most relevant document chunks based on user queries
* 💬 Generate context-aware responses using Mistral AI
* ♻️ Persistent vector storage allows previously uploaded PDFs to be reused without reprocessing
* 🌐 Interactive and user-friendly interface built with Streamlit

---

## 🛠️ Tech Stack

| Technology              | Purpose                    |
| ----------------------- | -------------------------- |
| Python                  | Core programming language  |
| Streamlit               | Web application interface  |
| LangChain               | RAG pipeline orchestration |
| Hugging Face Embeddings | Text embedding generation  |
| ChromaDB                | Vector database            |
| Mistral AI              | Large Language Model       |
| PyPDF                   | PDF document loading       |

---

## 🏗️ System Architecture

```text
User Uploads PDF
        │
        ▼
Load PDF Documents
        │
        ▼
Split Text into Chunks
        │
        ▼
Generate Embeddings
        │
        ▼
Store in ChromaDB
        │
        ▼
User Query
        │
        ▼
Similarity Search
        │
        ▼
Retrieve Relevant Chunks
        │
        ▼
Pass Context + Query to Mistral LLM
        │
        ▼
Generate Final Response
```

---

## 📁 Project Structure

```bash
├── streamlit_app.py      # Main Streamlit application
├── db/                   # Persistent Chroma vector database
├── requirements.txt      # Project dependencies
├── .env                  # Environment variables
├── README.md
```

---

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/pdf-rag-assistant.git
cd pdf-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

Activate the environment:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the root directory and add:

```env
MISTRAL_API_KEY=your_api_key_here
```

---

## ▶️ Run the Application

```bash
streamlit run streamlit_app.py
```

The application will start on:

```bash
http://localhost:8501
```

---

## 💡 How It Works

1. User uploads PDF documents.
2. PDFs are loaded and split into smaller text chunks.
3. Embeddings are generated using Hugging Face models.
4. Embeddings are stored in ChromaDB.
5. When a user asks a question, relevant chunks are retrieved.
6. Retrieved context is sent to Mistral AI.
7. The LLM generates an accurate, context-aware answer.

---

## 📸 Demo

Add screenshots or GIFs here.

```markdown
![Application Screenshot](images/demo.png)
```

---

## 🔮 Future Improvements

* Chat history support
* Multi-document conversation
* Source citation for answers
* Voice-based interaction
* Support for DOCX and TXT files
* Authentication and user management

---

## 🤝 Contributing

Contributions are welcome. Feel free to fork the repository and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Parmpreet Kaur**

If you found this project helpful, feel free to give it a ⭐ on GitHub!
