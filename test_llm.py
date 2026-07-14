import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

models_to_test = [
    "gemini-3.5-flash",
    "gemini-2.5-pro",
    "gemini-2.0-flash-lite",
    "gemini-3.1-pro-preview",
    "gemini-pro-latest"
]

working_model = None

for model in models_to_test:
    print(f"\\nTesting model: {model}")
    try:
        llm = ChatGoogleGenerativeAI(model=model, temperature=0)
        res = llm.invoke([HumanMessage(content="Hello")])
        print(f"SUCCESS with {model}! Response: {res.content[:50]}")
        working_model = model
        break
    except Exception as e:
        print(f"FAILED with {model}. Error: {e}")

if working_model:
    print(f"\\n---> Use this model in app.py: {working_model}")
else:
    print("\\n---> NO MODELS WORKED!")
