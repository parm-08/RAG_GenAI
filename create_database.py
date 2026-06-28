# WHAT WE WILL DO?

# 1 Load the PDF
# 2 Split the PDF into chunks
# 3 Create embeddings for each chunk
# 4 Store the embeddings in a vector database
# 5 Use a retriever to fetch relevant chunks
# 6 Pass retrieved chunks to the LLM for answering

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_mistralai import ChatMistralAI
from langchain_mistralai import MistralAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv
load_dotenv()


#Loading the pdf file
loader=PyPDFLoader("Document_loaders/deeplearning.pdf")
docs=loader.load() 

#Splitting the pdf into chunks
splitter=RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200)

chunks=splitter.split_documents(docs)

#Creating Embeddings and storing the chunks in vector database
embeddings=MistralAIEmbeddings(model="mistral-embed")
vector_store=Chroma.from_documents(documents=chunks,embedding=embeddings,persist_directory="chroma_db")

#-----------------------------------
# LLM can read long text, but it has context limits and is inefficient at searching.
# So we use a retriever to fetch only the most relevant chunks, then give those to the LLM for answering.
#-----------------------------------

# A RETRIEVER
# A retriever takes a user query and returns the most relevant documents or chunks from a database.

# Now there are 2 types of retriever
# 1. By data source (Wikipedia, Arxiv, PubMed, etc.)
# 2. By retrieval strategy (Similarity, MMR, MultiQuery, etc.) Used in RAG.


# Check the Retriever folder for more details on different types of retrievers and their code examples.

