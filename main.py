from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation writer."),
    ("user", "Summarize this text: {input}")
])

llm = ChatOpenAI()
output_parser = StrOutputParser() # to see the output as a string, avoiding message object received from llm

chain = prompt | llm | output_parser
text = input("Enter text to summarize: ")
response = chain.invoke({"input": text})
print(response)