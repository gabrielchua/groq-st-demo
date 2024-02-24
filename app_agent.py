"""
Groq Agent
"""
import time

import streamlit as st
from langchain import hub
from langchain_groq import ChatGroq
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_community.callbacks import StreamlitCallbackHandler
from openai import OpenAI

client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

def execute_search_agent(query):
    """
    Execute the search agent
    """
    # Define Groq LLM Model
    llm = ChatGroq(temperature=0,
                groq_api_key=st.secrets["GROQ_API_KEY"],
                model_name="mixtral-8x7b-32768")

    # Web Search Tool
    tools = [TavilySearchResults(max_results=3)]

    # Pull prompt from LangChain Hub
    react_prompt = hub.pull("hwchase17/react")

    # Construct the ReAct agent
    agent = create_react_agent(llm, tools, react_prompt)

    # Create an agent executor by passing in the agent and tools
    agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

    return agent_executor.invoke({"input": query}, 
                                 {"callbacks": [st_callback]})

def check_text(text):
    """
    check the text
    """
    response = client.moderations.create(input=text)
    return response.results[0].flagged

def is_fake_question(text):
    """Check if the given text is safe for work using gpt4 zero-shot classifer."""
    response = client.chat.completions.create(
        model="gpt-3.5-turbo-0125",
        messages=[{"role": "system", "content": "Is the given text an actual question? If yes, return `1`, else return `0`"},
                  {"role": "user", "content": text}],
        max_tokens=1,
        temperature=0,
        seed=0,
        logit_bias={"15": 100,
                    "16": 100}
    )

    result = int(response.choices[0].message.content)

    if result == 1:
        return 0
    return 1


st.title("agents go brrrr with groq")
st.subheader("ReAct Search Agent")
st.write("Powered by Groq, Mixtral, LangChain, and Tavily.")

query = st.text_input("Search Query", "Why is Groq so fast?")

if st.button("Search"):
    with st.spinner("Checking your response..."):
        is_nsfw = check_text(query)
        is_fake_qn = is_fake_question(query)
    if is_nsfw or is_fake_qn:
        st.warning("Your query was flagged. Please try again.", icon="🚫")
        st.stop()
    
    start_time = time.time()
    st_callback = StreamlitCallbackHandler(st.container())
    try:
        results = execute_search_agent(query)
    except ValueError:
        st.error("An error occurred while processing your request. Please try again.")
        st.stop()
    st.success(f"Completed in {round(time.time() - start_time, 2)} seconds.")
    st.info(f""" ### QN: {results['input']}

**ANS:** {results['output']}""")
