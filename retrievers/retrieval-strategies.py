from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_mistralai import ChatMistralAI
from langchain_core.documents import Document
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from dotenv import load_dotenv
load_dotenv()

embeddings=HuggingFaceEmbeddings(model="Qwen/Qwen3-Embedding-0.6B")
llm=ChatMistralAI(model="mistral-medium-latest")

docs = [
    Document(
        page_content="Python is a popular programming language used for web development, data science, and AI.",
        metadata={"source": "doc1", "topic": "programming"}
    ),
    Document(
        page_content="FastAPI is a modern web framework for building APIs with Python. It is fast and easy to use.",
        metadata={"source": "doc2", "topic": "backend"}
    ),
    Document(
        page_content="LangChain helps developers build applications powered by large language models using chains, agents, and retrievers.",
        metadata={"source": "doc3", "topic": "llm"}
    ),
    Document(
        page_content="PostgreSQL is an open-source relational database known for its reliability and advanced features.",
        metadata={"source": "doc4", "topic": "database"}
    ),
    Document(
        page_content="Machine learning enables computers to learn from data and make predictions without being explicitly programmed.",
        metadata={"source": "doc5", "topic": "ai"}
    )
]

database=Chroma.from_documents(docs,embeddings)

query="what is python and fastapi?"


#this is not retriever just a database function:
# result=database.similarity_search(query,k=3)
# for r in result:
#      print(r.page_content)

#similarity search retriever: (based on cosine similarity)
# result=database.as_retriever(
#      search_type="similarity",
#      search_kwargs={"k":3}
# )
# result=result.invoke(query)
# for r in result:
#      print(r.page_content)

#max marginal relevence:
retriever=database.as_retriever(
     search_type="mmr",
     search_kwargs={"k":3}
)

#multi query retriever:
result=MultiQueryRetriever.from_llm(
     llm=llm,
     retriever=retriever
)

result=result.invoke(query)
for r in result:
     print(r.page_content)