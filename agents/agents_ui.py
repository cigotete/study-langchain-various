import streamlit as st
from langchain.callbacks import StreamlitCallbackHandler
from agents.agents_basics import load_agent

chain = load_agent()
st_callback = StreamlitCallbackHandler(st.container())

if prompt := st.chat_input():
  st.chat_message("user").write(prompt)
  with st.chat_message("assistant"):
    st_callback = StreamlitCallbackHandler(st.container())
    response = chain.run(prompt, callbacks=[st_callback])
    st.write(response)