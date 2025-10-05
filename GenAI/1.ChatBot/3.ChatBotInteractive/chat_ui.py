import streamlit as st
import requests

st.set_page_config(page_title="ChatGPT-style Chatbot", layout="wide")

# Initialize session state for chat messages
if "messages" not in st.session_state:
    st.session_state.messages = []

# Function to start a new chat
def new_chat():
    st.session_state.messages = []

# Sidebar with New Chat button
with st.sidebar:
    st.button("New Chat", on_click=new_chat)

# Chat history container
chat_container = st.container()

# Bottom input box (fixed at bottom using a form)
with st.form(key="input_form", clear_on_submit=True):
    user_input = st.text_input("Type your message here...")
    submit_button = st.form_submit_button(label="Send")

if submit_button and user_input:
    
    # Append user message
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Call Flask API for AI response
    try:
        response = requests.post(
            "http://127.0.0.1:5400/api/chat", 
            json={"message": user_input}
        )
        ai_message = response.json().get("response", "Error: No response")
    except Exception as e:
        ai_message = f"Error: {e}"

    # Append AI response
    st.session_state.messages.append({"role": "bot", "content": ai_message})

# Display chat messages 
with chat_container:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(f"**User:** {msg['content']}")
        else:
            st.markdown(f"**Bot:** {msg['content']}")
