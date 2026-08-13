from langchain.agents import create_agent
from langchain_mistralai import ChatMistralAI
from langchain.messages import HumanMessage
from langchain_community.tools.tavily_search import TavilySearchResults
from dotenv import load_dotenv
load_dotenv()
from langchain.tools import tool
import requests
import os
from rich import print

llm=ChatMistralAI(model="mistral-medium-latest")
weatherapi=os.getenv("OPENWEATHER_API_KEY")

@tool
def get_weather(city:str)->dict:
     """this function used to fetch weather data of a city"""
     url = "https://api.openweathermap.org/data/2.5/weather"

     params = {
          "q": city,
          "appid": weatherapi,
          "units": "metric"
     }

     response=requests.get(url,params=params)

     if response.status_code != 200:
          return f"no data found for city {city}"

     data=response.json()

     return {
          "temperature":data["main"]["temp"],
          "feels like":data["main"]["feels_like"],
          "description":data["weather"][0]["description"]
     }

@tool
def get_news(city:str)->list:
     """this function gives latest news of city given"""
     search_engine=TavilySearchResults(max_results=3)
     news=search_engine.invoke(f"latest news of {city}")
     return news

agent = create_agent(
     model=llm,
     tools=[get_news,get_weather],
     system_prompt="you are a helpful city weather and news assistant"
)

print("welcome to the city agent, press 0 to exit")

while True:
     query=input("ask weather or news about a city: ")
     if query=="0": break
     result=agent.invoke({
          "messages":[{"role":"user","content":query}]
     })
     print(result["messages"][-1].content)