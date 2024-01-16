import os
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import OpenAI
from langchain_community.document_loaders import PyPDFLoader

pdf_file_path = "./pdf_files/pdf-demo2.pdf"
pdf_loader = PyPDFLoader(pdf_file_path)
docs = pdf_loader.load_and_split()
llm = OpenAI()
chain = load_summarize_chain(llm, chain_type="map_reduce")
response = chain.invoke(docs)
print(response['output_text'])