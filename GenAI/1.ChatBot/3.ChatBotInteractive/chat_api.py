from flask import Flask, request, jsonify
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage, AIMessage

llm = ChatOllama(
    model="gemma:2b",
    max_tokens=100
)

messages = []

app = Flask(__name__)

@app.route('/api/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_message = data.get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Add user message to history
    messages.append(HumanMessage(user_message))
    # Get AI response
    response = llm.invoke(messages)
    # Add AI message to history
    messages.append(AIMessage(response.content))

    return jsonify({"response": response.content})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5400, debug=True)
