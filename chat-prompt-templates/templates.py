from langchain_core.prompts import ChatPromptTemplate
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv

load_dotenv()

model=ChatMistralAI(model="mistral-medium-latest")

prompt=ChatPromptTemplate.from_messages([
     ("system",
     """
     You are an advanced information extraction and summarization engine.

     Your task is to analyze the given input text and perform two operations:
     1. Extract all key structured information
     2. Generate a concise, high-quality summary

     """
     ),
     ("human",
      """
     INPUT TEXT:

     {input_text}

     INSTRUCTIONS:

     Step 1: Information Extraction  
     - Identify and extract all important entities, facts, and relationships.
     - Organize the output into structured JSON format.
     - Include (when available):
     - Main topic
     - Key entities (people, organizations, locations, products, etc.)
     - Important dates / time references
     - Numerical data / metrics
     - Events or actions described
     - Key relationships between entities
     - Any domain-specific insights (technical, business, etc.)

     Step 2: Summary  
     - Write a clear and concise summary (3-5 sentences).
     - Focus on the most important insights.
     - Avoid redundancy.
     - Maintain factual accuracy


     OUTPUT FORMAT:
     "extracted_information:"
     "main_topic":,
     "entities":,
     "dates": ,
     "metrics": ,
     "events": ,
     "relationships":,
     "key_insights":,
     "summary":


     RULES:
     - Do NOT hallucinate information.
     - If something is missing, return an empty list [].
     - Keep the output strictly in JSON format.
     - Ensure clarity and precision.

     """
)
])

para=input("enter paragraph to extract information")
final_prompt=prompt.invoke({"input_text":para})
response=model.invoke(final_prompt)
print(response.content)