import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

llm = ChatOpenAI(
    openai_api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",
    max_tokens=20,
    temperature=0
)

def detect_action(prompt: str):
    classification_prompt = f"""
You are a strict classifier.

Classify the following user request into ONE of these categories only:
- IMAGE
- AUDIO
- TRANSLATION
- TEXT

Return only the category name (no explanation).

User request: "{prompt}"
"""
    response = llm.invoke([HumanMessage(content=classification_prompt)])
    return response.content.strip().upper()

def main():
    user_input = input("Enter your prompt: ")
    action_type = detect_action(user_input)
    print("Detected Action:", action_type)

if __name__ == "__main__":
    main()
