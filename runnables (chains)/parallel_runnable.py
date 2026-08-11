from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel,RunnableLambda
from dotenv import load_dotenv
load_dotenv()

prompt1=ChatPromptTemplate([
     "give me {topic1} explanation in simple words"
])
prompt2=ChatPromptTemplate([
     "give me {topic2} explanation in simple words"
])

model = ChatMistralAI(model="mistral-medium-latest")

output=StrOutputParser()

#parallel & lambda runable (for single parameter)
chain=RunnableParallel({
     "chain1":prompt1 | model | output,
     "chain2":prompt2 | model | output
})
response=chain.invoke("computer vision")
print(response)

#parallel & lambda runable (for two different parameters)
chain=RunnableParallel({
     "chain1":RunnableLambda(lambda x: x["chain1"]) | prompt1 | model | output,
     "chain2":RunnableLambda(lambda x: x["chain2"]) | prompt2 | model | output
})

response=chain.invoke({
    "chain1":{"topic1":"computer vision"},
    "chain2":{"topic2":"natural language processing"} 
})

print(response)