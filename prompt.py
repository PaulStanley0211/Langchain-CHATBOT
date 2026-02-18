## Integrate OpenAI API with LangChain

import os
from constants import openai_api_key
from langchain_openai import OpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

from langchain.memory import ConversationBufferMemory

from langchain.chains import SequentialChain


import streamlit as st

os.environ['OPENAI_API_KEY'] = openai_api_key

# streamlit framework
st.title('Celeberity search Results')
input_text = st.text_input('Search the topic you want to know about:')

# prompt template
first_prompt_template = PromptTemplate(
    input_variables=["name"],
    template="Tell me about {name}"
)

# memory
person_memory = ConversationBufferMemory(input_key="name", memory_key="person")
date_of_birth_memory = ConversationBufferMemory(input_key="person", memory_key="date_of_birth")
events_memory = ConversationBufferMemory(input_key="date_of_birth", memory_key="events")


## openai llms model
llm = OpenAI(temperature=0.8)
chain = LLMChain(llm=llm, prompt=first_prompt_template, verbose=True, output_key="person", memory=person_memory)

# prompt template
second_prompt_template = PromptTemplate(
    input_variables=["person"],
    template="TWhen was {person} born?"
)

chain2 = LLMChain(llm=llm, prompt=second_prompt_template, verbose=True, output_key="date_of_birth", memory=date_of_birth_memory)

##third prompt template
third_prompt_template = PromptTemplate(
    input_variables=["date_of_birth"],
    template="Mention 5 major events happened around {date_of_birth} in the world"
)

chain3 = LLMChain(llm=llm, prompt=third_prompt_template, verbose=True, output_key="events", memory=events_memory)

parent_chain = SequentialChain(chains=[chain, chain2, chain3],input_variables=["name"],output_variables=["person", "date_of_birth", "events"], verbose=True, memory=[person_memory, date_of_birth_memory, events_memory])

if input_text:
    st.write(parent_chain.invoke({"name": input_text}))

    with st.expander("Person"):
        st.write(person_memory.load_memory_variables())
    with st.expander("Date of Birth"):
        st.write(date_of_birth_memory.load_memory_variables())
    with st.expander("Events"):
        st.write(events_memory.load_memory_variables()) 
