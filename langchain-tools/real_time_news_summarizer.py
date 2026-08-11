from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableSequence
from dotenv import load_dotenv
load_dotenv()

model = ChatMistralAI(model="mistral-medium-latest")

search_engine=TavilySearchResults(max_results=5)

outputparser=StrOutputParser()

prompt=ChatPromptTemplate.from_messages([
     ("system","you are a news summarizer assistant"),
     ("human","summarize the following news into 5 bullet points\n {news}")
])

sequence= RunnableSequence(search_engine | prompt | model | outputparser)

news=input("ask news about anything to get a summary: ")


response=sequence.invoke(news)

print(response)