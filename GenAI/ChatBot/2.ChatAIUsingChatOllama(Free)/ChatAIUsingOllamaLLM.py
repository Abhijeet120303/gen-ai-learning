from langchain_ollama import OllamaLLM

llm = OllamaLLM(
    model="gemma:2b",
    max_tokens=100
)
prompt = "Addition of 2 number in java?"

# Call the Ollama API
response = llm.invoke(prompt)
print(response)