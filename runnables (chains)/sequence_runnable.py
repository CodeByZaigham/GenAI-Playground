from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

prompt=ChatPromptTemplate([
     "give me {topic} explanation in simple words"
])

model = ChatMistralAI(model="mistral-medium-latest") 

output=StrOutputParser()

#sequence runable
chain=RunnableSequence(prompt | model | output)

response=chain.invoke({"topic":"deep learning"})

print(response)