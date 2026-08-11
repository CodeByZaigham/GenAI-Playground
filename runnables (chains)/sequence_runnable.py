from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
load_dotenv()

prompt=ChatPromptTemplate([
     "give me {topic} explanation in simple words"
])

model = ChatMistralAI(model="mistral-medium-latest",temperature=0,max_tokens=20) #good model

output=StrOutputParser()

#sequence runable
chain=prompt | model | output

response=chain.invoke({"topic":"deep learning"})

print(response)