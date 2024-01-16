from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_openai import ChatOpenAI

examples = [
  {
  "input": "I absolutely love the new update! Everything works seamlessly.",
  "output": "Positive",
  },
  {
  "input": "It's okay, but I think it could use more features.",
  "output": "Neutral",
  },
  {
  "input": "I'm disappointed with the service, I expected much better performance.",
  "output": "Negative"
  }
]

example_prompt = PromptTemplate(
  template="{input} -> {output}",
  input_variables=["input", "output"],
)
prompt = FewShotPromptTemplate(
  examples=examples,
  example_prompt=example_prompt,
  suffix="Question: {input}",
  input_variables=["input"]
)
print((prompt | ChatOpenAI()).invoke({"input": " This is an excellent book with high quality explanations."}))