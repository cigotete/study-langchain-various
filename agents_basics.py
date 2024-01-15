import os
from langchain.agents import (
AgentExecutor, AgentType, initialize_agent, load_tools
)

from langchain_openai import ChatOpenAI

def load_agent() -> AgentExecutor:
  llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0.1,
    streaming=True
  )

  tools = load_tools(
    tool_names=["ddg-search", "arxiv", "wikipedia"],
    llm=llm
  )
  return initialize_agent(
  tools=tools, llm=llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
  verbose=True
  )