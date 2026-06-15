# RAG_GenAI

## Overview
RAG_GenAI is an end-to-end Retrieval-Augmented Generation (RAG) application built using Large Language Models (LLMs), vector databases, and document retrieval techniques. The system retrieves relevant information from documents and generates accurate, context-aware responses.

This approach improves response quality while reducing hallucinations in AI-generated outputs.

---

## Features

- Document ingestion and processing
- Vector embedding generation
- Efficient semantic search using vector databases
- Retrieval-Augmented Generation (RAG)
- Context-aware response generation using LLMs
- Modular and scalable architecture

---

## Project Structure

```
RAG_project/
│
├── Document_loaders/      # Document loading and preprocessing
├── Retriever/             # Retrieval logic
├── Vector_stores/         # Vector database operations
├── app.py                 # Application interface
├── main.py                # Main execution file
├── create_database.py     # Creates vector database
├── requirements.txt       # Dependencies
└── README.md
```

---

## Technologies Used

- Python
- LangChain
- Vector Databases (ChromaDB)
- Embeddings
- Large Language Models (LLMs)
- Retrieval-Augmented Generation (RAG)

---

## Installation

### Clone the repository

```bash
git clone <repository_url>
cd RAG_GenAI
```

### Create virtual environment

```bash
python -m venv .venv
```

Activate environment:

**Windows**
```bash
.venv\Scripts\activate
```

**Linux/Mac**
```bash
source .venv/bin/activate
```

### Install dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file and add your API keys:

```env
OPENAI_API_KEY=your_api_key
GROQ_API_KEY=your_api_key
```

---

## Usage

### Create Vector Database

```bash
python create_database.py
```

### Run the Application

```bash
python app.py
```

or

```bash
python main.py
```

---

## How It Works

1. Load documents using document loaders.
2. Generate embeddings for document chunks.
3. Store embeddings in a vector database.
4. Retrieve relevant context based on user queries.
5. Pass retrieved context to the LLM.
6. Generate accurate and context-aware responses.

---

## Future Enhancements

- Multi-document support
- PDF and web scraping integration
- Conversational memory
- Advanced reranking techniques
- Deployment using Docker and Cloud platforms

---

## Author

**Parmpreet Kaur**

Open to collaborations in AI, Machine Learning, Deep Learning, and Generative AI.
