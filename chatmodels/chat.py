from langchain.chat_models import init_chat_model
from langchain_groq import ChatGroq
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mistralai import ChatMistralAI
from dotenv import load_dotenv
load_dotenv()

# temperature=0 if doing logical work
# temperature=(0.7 - 1) if doing creative work
# max token used to save your tokens by limiting size of responses

# model1 = ChatGroq(model="openai/gpt-oss-120b") #good model
# reponse=model1.invoke("what is NLP")
# print(reponse.content)

# model2 = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite") #good model for paragraph writing
# reponse=model2.invoke("what is NLP")
# print(reponse.content)

model3 = ChatMistralAI(model="mistral-medium-latest",temperature=0,max_tokens=20) #good model
reponse=model3.invoke("build a simple restaurant management system backend using fastapi")
print(reponse.content)