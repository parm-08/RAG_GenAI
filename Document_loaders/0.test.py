from langchain_community.document_loaders import TextLoader

data =TextLoader("notes.txt")

# print(data)

docs=data.load()

# print(docs)
# print(docs[0])
print(len(docs))

