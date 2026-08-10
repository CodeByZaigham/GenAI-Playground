from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel
from typing import Optional,List
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

model=ChatMistralAI(model="mistral-medium-latest")

class rules(BaseModel):
     title:str
     year:Optional[int]
     directors:Optional[list]
     genre:List[str]
     cast:List[str]
     summary:str

parse=PydanticOutputParser(pydantic_object=rules)

prompt=ChatPromptTemplate.from_messages([
     ("system",
     """
     You are an advanced information extraction and summarization engine.

     {information_format}

     """
     ),
     ("human","{input_text}")
])

para=input("enter paragraph to extract information")
final_prompt=prompt.invoke({"input_text":para,"information_format":parse.get_format_instructions()})
response=model.invoke(final_prompt)
# output=parse.parse(response.content)
print(response.content)