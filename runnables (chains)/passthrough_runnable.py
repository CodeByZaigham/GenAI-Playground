from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence,RunnableParallel,RunnablePassthrough
from dotenv import load_dotenv
load_dotenv()

code=ChatPromptTemplate.from_messages([
     ("system","you are a senior python programmer"),
     ("human","generate code of {topic}")
])
code_explanation=ChatPromptTemplate.from_messages([
     ("system","you are a great teacher who explains code perfectly."),
     ("human","{topic} \n explain me this code")
])

model = ChatMistralAI(model="mistral-medium-latest")

output=StrOutputParser()

code_output=RunnableSequence(code | model | output)

explanation=RunnableParallel({
     "code":RunnablePassthrough(),
     "explanation":RunnableSequence(code_explanation | model | output) 
})

final_sequence=RunnableSequence(code_output | explanation)

response=final_sequence.invoke("palindrome")

print(response['code'])
print(response['explanation'])