import os
from langchain.chains.llm import LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain_openai import OpenAI, ChatOpenAI
from langchain_community.document_loaders import PyPDFLoader
from langchain.callbacks import get_openai_callback

from langchain_core.prompts import ChatPromptTemplate

question = input("Enter question: ")
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are world class technical documentation reader."),
    ("user", """Given the following extracted parts of a document and a question,
If you don't know the answer, just say that you don't know. Don't try to make up an answer.
QUESTION: {question}
=========
Content: {text}""")
])

pdf_file_path = "./pdf_files/pdf-demo.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()

# Define LLM chain
llm = ChatOpenAI(
  temperature=0.1,
  model="gpt-3.5-turbo"
)
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Define StuffDocumentsChain
stuff_chain = StuffDocumentsChain(
  llm_chain=llm_chain,
  document_variable_name="text"
)

with get_openai_callback() as cb:
  response = stuff_chain.invoke({"input_documents": docs, "question": question})
  print(response['output_text'])
  print(f"Total Tokens: {cb.total_tokens}")
  print(f"Prompt Tokens: {cb.prompt_tokens}")
  print(f"Completion Tokens: {cb.completion_tokens}")
  print(f"Total Cost (USD): ${cb.total_cost}")