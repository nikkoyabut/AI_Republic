import streamlit as st
import openai
import warnings  # Import the warnings module
import os
# import openai
import numpy as np
import pandas as pd
import json
# from langchain.chat_models import ChatOpenAI
# from langchain.document_loaders import CSVLoader
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.prompts import ChatPromptTemplate
# from langchain.vectorstores import Chroma
# from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableLambda, RunnablePassthrough
# from openai.embeddings_utils import get_embedding
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention



def main():
    warnings.filterwarnings("ignore")
    st.set_page_config(page_title="Chatbot ng Bayan - News Summarizer", page_icon="🤖", layout="wide")

    with st.sidebar:
        openai.api_key = st.text_input("OpenAI API Key", type="password")
        if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
            st.warning("Please enter your OpenAI API key.")
        else:
            st.success("API key successfully. Proceed to enter prompt message.")
    
    with st.container():
        l,m,r = st.columns([1,3,1])
        with l : st.empty()
        with m : st.empty()
        with r : st.empty()
    
    options = option_menu(
        "Dashbaord",
        ["Home", "About Us", "Model"],
        icons = ["🏠", "🤖", "📚"],
        menu_icon = "book",
        default_index = 0
    )

    st.title("Chatbot ng Bayan")
    st.write("Welcome to the Chatbot!")

    # Initialize the conversation structure in session state if not already present
    if 'messages' not in st.session_state:
        st.session_state['messages'] = []
        st.session_state['chat_history'] = []

    # User input
    user_input = st.text_input("You: ", key="input")

    # On button click, send user input to the chatbot
    if st.button("Send"):
        if user_input:
            # Add user input to session state message history
            st.session_state['messages'].append({"user": "You", "message": user_input})
            st.session_state['chat_history'].append({"role": "user", "content": user_input})

            # Get bot response from OpenAI API
            response = get_bot_response(st.session_state['chat_history'])
            st.session_state['messages'].append({"user": "Bot", "message": response})
            st.session_state['chat_history'].append({"role": "assistant", "content": response})

    # Display chat history
    for message in st.session_state['messages']:
        st.write(f"{message['user']}: {message['message']}")

def get_bot_response(chat_history):
    try:
        # Call to OpenAI's Chat API
        chat = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",  # or "gpt-4" if available
            messages=chat_history
        )
        response = chat.choices[0].message.content
        return response
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    main()
