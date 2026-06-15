# MULTIQUERY RETRIEVER
# MultiQueryRetriever creates multiple different versions
# of the user's question using an LLM.
# Then it searches for documents using all those queries.
# This helps retrieve better and broader context.

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_classic.retrievers.multi_query import MultiQueryRetriever

load_dotenv()

# =========================
# STEP 1: CREATE DOCUMENTS
# =========================

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="There are different types of gradient descent."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

# =========================
# STEP 2: LOAD EMBEDDINGS
# =========================

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# STEP 3: CREATE VECTOR DB
# =========================

vectorstore = Chroma.from_documents(docs, embeddings)

# =========================
# STEP 4: CREATE RETRIEVER
# =========================

retriever = vectorstore.as_retriever()

# =========================
# STEP 5: LOAD LLM
# =========================

llm = ChatMistralAI(
    model="mistral-small-latest"
)

# =========================
# STEP 6: MULTIQUERY RETRIEVER
# =========================

multiquery_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm
)

# =========================
# STEP 7: ASK QUESTION
# =========================

query = "What is gradient descent?"

docs = multiquery_retriever.invoke(query)

# =========================
# STEP 8: PRINT RESULTS
# =========================

print("\n===== MultiQuery Retriever Results =====\n")

for doc in docs:
    print(doc.page_content)