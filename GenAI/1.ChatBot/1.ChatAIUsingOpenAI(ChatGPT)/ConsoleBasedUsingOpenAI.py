import os
from langchain_openai import ChatOpenAI

# Create object of ChatOpenAI to access OpenAI API
llm = ChatOpenAI(openai_api_key=os.getenv("OPENAI_API_KEY"), model="gpt-4o", max_tokens=6)

while True:
    prompt = input("User : ")
    if prompt.lower() == "exit":
        break

    response = llm.invoke(prompt)
    print("Bot : "+response.content)

print("Goodbye!")