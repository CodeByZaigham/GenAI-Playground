from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter,RecursiveCharacterTextSplitter,TokenTextSplitter

data=PyPDFLoader("text-splitting/neural_networks_and_transformers.pdf")
docs=data.load()

#charater based splitting -> separator you gives
# split=CharacterTextSplitter(
#      separator="\n",
#      chunk_size=500,
#      chunk_overlap=5
# )
# chunks=split.split_documents(docs)
# print(chunks[0])

#token based splitting -> tiktoken(openai)
# split=TokenTextSplitter(
#      chunk_size=500,
#      chunk_overlap=5
# )
# chunks=split.split_documents(docs)
# print(chunks[6])

#recursive text splitter -> ["\n\n","\n"," ",""]
split=RecursiveCharacterTextSplitter(
     chunk_size=500,
     chunk_overlap=50
)
chunks=split.split_documents(docs)



