from langchain_openai import ChatOpenAI

llm = ChatOpenAI()
response = llm.invoke("how can langsmith help with testing?")
print(response)