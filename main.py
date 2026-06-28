# #Here we are performing tasks like summarizing the text file or pdf.

# from dotenv import load_dotenv
# from langchain_mistralai import ChatMistralAI
# # from langchain_community.document_loaders import TextLoader
# from langchain_core.prompts import ChatPromptTemplate
# from langchain_community.document_loaders import PyPDFLoader
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# load_dotenv()

# # data = TextLoader("Document_loaders/notes.txt")
# # data=PyPDFLoader("Document_loaders/deeplearning.pdf")
# # docs=data.load()

# # splitter =RecursiveCharacterTextSplitter(
# #     chunk_size=1000,
# #     chunk_overlap=200
# # )

# # chunks=splitter.split_documents(docs)


# template = ChatPromptTemplate.from_messages([("system","You are an AI that summarizies text"),("human","{data}")])

# model=ChatMistralAI(model="mistral-small-2506")

# # prompt=template.format_prompt(data=docs)

# # result=model.invoke(prompt)

# # print(result.content)


# # Most of the embedding models and language model have a context
# # window and they cannot take infinite tokens at the same time so for
# # that we have to use the text splitting.

#--------- FINAL MAIN.PY FILE ---------

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore=Chroma(persist_directory="Vector_stores/chroma_db",embedding_function=embedding_model)

retriever=vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k":4,"fetch_k":10,"lambda_mult":0.5}

)
llm=ChatMistralAI(model="mistral-small-2506")

#prompt template
prompt=ChatPromptTemplate.from_messages(
    [
        ("system", """You are a helpful assistant.
        Use ONLY the provided context to answer the question.
        If the answer is not present in the context, say: " could not find the answer in the document."
         """),
        ("human","Context: {context}\n\n Question: {question}")
        
    ]
)

print("RAG SYSTEM CREATED !")
print("Press 0 to EXIT !")
while True:
    query=input("YOU:")
    if query=="0":
        print("Exiting...")
        break

    docs=retriever.invoke(query)
    context="\n\n".join([doc.page_content for doc in docs])
    final_prompt=prompt.format_prompt(context=context,question=query)
    answer=llm.invoke(final_prompt)
    print("\nAnswer:",answer.content)


