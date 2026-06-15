#An arXiv retriever is a tool that helps your AI (like an LLM app) search and fetch research papers from arXiv automatically.

from langchain_community.retrievers import ArxivRetriever

# create the retriever
retriever = ArxivRetriever(
    load_max_docs=5,          # number of papers to fetch
    load_all_available_meta=True
)

# query arxiv
docs = retriever.invoke("large language models")

# print results
for i, doc in enumerate(docs):
    print(f"\nResult {i+1}")
    print("Title:", doc.metadata.get("Title"))
    print("Authors:", doc.metadata.get("Authors"))
    print("Published:", doc.metadata.get("Published"))
    print("Summary:", doc.page_content[:500])