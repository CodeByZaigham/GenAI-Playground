from langchain.messages import HumanMessage
from dotenv import load_dotenv
load_dotenv()
import os
import requests
from langchain_core.tools import tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_mistralai import ChatMistralAI

weatherapi=os.getenv("OPENWEATHER_API_KEY")

llm=ChatMistralAI(model="mistral-medium-latest")

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

tools={
     "get_news":get_news,
     "get_weather":get_weather
}

llm_with_tools=llm.bind_tools([get_news,get_weather])

# AGENT LOGIC

print("welcome to the agent!! enter 0 to exit")

while True:

     messages=[]

     query=input("ask news and weathers about any city")

     if query=="0": break

     prompt=HumanMessage(
          content=query
     )

     messages.append(prompt)

     response=llm_with_tools.invoke([prompt])

     messages.append(response)

     if response.tool_calls:
          for i in response.tool_calls:
               name=i["name"]
               tool_response=tools[name].invoke(i)
               messages.append(tool_response)
          output=llm_with_tools.invoke(messages)
          print(output.content)

