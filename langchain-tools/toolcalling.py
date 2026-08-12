from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.messages import HumanMessage
from rich import print

model = ChatMistralAI(model="mistral-medium-latest")

@tool
def weather(city:str) -> str:
     """check weather of cities given"""
     update={
          "Karachi":"34'C",
          "Lahore":"28'C",
          "Islamabad":"24'C"
     }
     return update.get(city,"no data found")

tools={
     "weather":weather
}

messages=[]

model_with_tool=model.bind_tools([weather])
prompt=HumanMessage("what is the weather in karachi today?")
messages.append(prompt)
response=model_with_tool.invoke([prompt])
messages.append(response)
if response.tool_calls:
     for i in response.tool_calls:
          name=i["name"]
          result=tools[name].invoke(i)
messages.append(result)
output=model_with_tool.invoke(messages)
print(output.content)

          