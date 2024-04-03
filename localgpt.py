# import streamlit as st
# from langchain_community.llms import HuggingFaceEndpoint
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# import os
# from dotenv import load_dotenv
# from PyPDF2 import PdfReader

# load_dotenv()
# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "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"


# HUGGINGFACEHUB_API_TOKEN = "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"
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"


# reader = PdfReader("Om Tarkunde Resume_offcampus.pdf")
        
# page = reader.pages[0]

# text = page.extract_text()

# question = "What is the name of the person "

# prompt = f"""
#                 **Question:** {question}

#                 **Context:**  {text}

#                 **Answer Style:** Short Answer
#             """

# def answer_question(question):
#     prompt = PromptTemplate(input_variables=["question"], template="Answer in short and to the point regarding {question}")
#     prompt.format(question=question)

#     llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5, max_new_token=128, token=HUGGINGFACEHUB_API_TOKEN)
#     llm_chain = LLMChain(prompt=prompt, llm=llm)

#     answer = llm_chain.invoke(question)
#     print(answer)
    
# answer_question(question=question)

from transformers import AutoModel, AutoTokenizer
from PyPDF2 import PdfReader
import torch

# Download and load the DistilBERT model and tokenizer
model_name = "distilbert-base-uncased-distilled-squad"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

# Read the PDF file
reader = PdfReader("Om Tarkunde Resume_offcampus.pdf")
page = reader.pages[0]
text = page.extract_text()

# Define the question
question = "What is the name of the person "

# Function to answer the question
def answer_question(question, context):
  """
  Answers a question based on the provided context using the DistilBERT model.

  Args:
      question (str): The question to be answered.
      context (str): The text to be analyzed (e.g., extracted from PDF).

  Returns:
      str: The predicted answer based on the model's output.
  """

  # Preprocess text using the tokenizer
  inputs = tokenizer(question, context, return_tensors="pt")

  # Pass the input to the model
  outputs = model(**inputs)

  # Extract the predicted answer tokens
  answer_start = torch.argmax(outputs.start_logits)
  answer_end = torch.argmax(outputs.end_logits)
  answer_tokens = inputs.input_ids[0][answer_start:answer_end + 1]

  # Convert answer tokens back to text
  answer = tokenizer.convert_tokens_to_strings(tokenizer.convert_ids_to_tokens(answer_tokens))[0]

  return answer

# Call the function to get the answer
answer = answer_question(question, text)

# Print the answer
print(f"Answer: {answer}")
