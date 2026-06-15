from langchain_community.document_loaders import PyPDFLoader

data=PyPDFLoader("gen_ai.pdf")

docs=data.load()
# print(docs)
print(len(docs))
print(docs[11]) #for a page
