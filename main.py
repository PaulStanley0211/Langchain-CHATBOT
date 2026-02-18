## Integrate OpenAI API with LangChain

import os
from constants import openai_api_key
from langchain_openai import OpenAI

import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_api_key

# streamlit framework
st.title('Langchain Demo with OPENAI API')
input_text = st.text_input('Search the topic you want to know about:')

## openai llms model
llm = OpenAI(temperature=0.8)



if input_text:
    st.write(llm.invoke(input_text))






