"""
app.py
"""
import time
import os

import streamlit as st
from groq import Groq

from self_discover import (
    REASONING_MODULES,
    select_reasoning_modules,
    adapt_reasoning_modules,
    implement_reasoning_structure,
    execute_reasoning_structure
    )

API_KEY = os.environ.get("GROQ_API_KEY")
if API_KEY is None:
    try:
        API_KEY = st.secrets["GROQ_API_KEY"]
    except KeyError:
        # Handle the case where GROQ_API_KEY is neither in the environment variables nor in Streamlit secrets
        st.error("API key not found.")
        st.stop()

client = Groq(api_key=API_KEY)

def groq_chat(model, 
              user_prompt, 
              system_prompt="You are a world-class problem solver with an IQ of 200."):
    """Generate a chat completion using the Groq API."""
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}],
    )
    return response.choices[0].message.content


st.set_page_config(page_title="Groq Demo", page_icon="⚡️", layout="wide")

st.title("Groq Demo")

tab1, tab2 = st.tabs(["Text Generation", "Self-Discover"])

with tab1:
    system_prompt = st.text_input("System Prompt", "You are a friendly chatbot.")
    user_prompt = st.text_input("User Prompt", "Tell me a joke.")

    model_list = client.models.list().data
    model_list = [model.id for model in model_list]

    model = st.radio("Select the LLM", model_list, horizontal=True)

    button = st.empty()
    time_taken = st.empty()
    response = st.empty()

    if button.button("Generate"):
        stream = client.chat.completions.create(
            model=model,
            messages=[{"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}],
            stream=True
            )

        start_time = time.time()

        streamed_text = ""

        for chunk in stream:
            chunk_content = chunk.choices[0].delta.content
            if chunk_content is not None:
                streamed_text = streamed_text + chunk_content
                response.info(streamed_text)

        time_taken.success(f"Time taken: {round(time.time() - start_time,4)} seconds")

with tab2:
    task = st.text_input("What is your task?")
    reasoning_model = st.radio("Select your LLM for this reasoning task", model_list, horizontal=True, help="mixtral is recommended for better performance, but appears to be slower")

    if st.button("Run"):
        prompt = select_reasoning_modules(REASONING_MODULES, task)
        select_reasoning_modules = groq_chat(reasoning_model, prompt)
        st.subheader("Step 1: SELECT relevant reasoning modules for the task")
        with st.expander("See the reasoning modules"):
            st.info(select_reasoning_modules)

        prompt = adapt_reasoning_modules(select_reasoning_modules, task)
        adapted_modules = groq_chat(reasoning_model, prompt)
        st.subheader("Step 2: ADAPT the selected reasoning modules to be more specific to the task.")
        with st.expander("See the adapted reasoning modules"):
            st.info(adapted_modules)

        prompt = implement_reasoning_structure(adapted_modules, task)
        reasoning_structure = groq_chat(reasoning_model, prompt)
        st.subheader("Step 3: IMPLEMENT the adapted reasoning modules into an actionable reasoning structure.")
        with st.expander("See the reasoning structure"):
            st.info(reasoning_structure)
        
        prompt = execute_reasoning_structure(reasoning_structure, task)
        result = groq_chat(reasoning_model, prompt)
        st.subheader("Step 4: Execute the reasoning structure to solve a specific task instance.")
        with st.expander("See the result"):
            st.info(result)
