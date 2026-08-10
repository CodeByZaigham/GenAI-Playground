from langchain_huggingface import HuggingFaceEmbeddings

model =HuggingFaceEmbeddings(
     model="Qwen/Qwen3-Embedding-0.6B"
)

sen="Lorem Ipsum is simply dummy text of the printing"


vector=model.embed_query(sen)

print(vector) 