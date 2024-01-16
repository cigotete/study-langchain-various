from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

reasoning_prompt = "{question}\nLet's think step by step!"

prompt = PromptTemplate(
template=reasoning_prompt,
input_variables=["question"]
)

model = ChatOpenAI()

chain = prompt | model
print(chain.invoke({"question": "There were 5 apples originally. I ate 2 apples. My friend gave me 3 apples. How many apples do I have now?",
}))