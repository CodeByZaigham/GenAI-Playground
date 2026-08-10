from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
load_dotenv()

llm=HuggingFaceEndpoint(
     repo_id="deepseek-ai/DeepSeek-V4-Flash",
     temperature=0
)

model=ChatHuggingFace(llm=llm)
response=model.invoke("what in genAI?")
print(response.content)