from langchain.tools import tool
from dotenv import load_dotenv
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain_core.messages import HumanMessage,ToolMessage
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

model_with_tool=model.bind_tools([weather])

prompt1=HumanMessage(
     content="what is the weather in karachi today?"
)

response=model_with_tool.invoke([prompt1])

tool_result=[]
for i in response.tool_calls:
     name=i["name"]
     args=i["args"]
     func=tools[name]
     result=func.invoke(args)
     tool_result.append(ToolMessage(content=str(result) , tool_call_id=i["id"]))

prompt2=ChatPromptTemplate.from_messages([
     ("system","you are a helpful assistant"),
     MessagesPlaceholder("chat_history")
])

final_prompt=prompt2.invoke({
     "chat_history":[
          prompt1,
          response,
          *tool_result
     ]
})

response=model.invoke(final_prompt)
print(response.content)

          