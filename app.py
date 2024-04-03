from PyPDF2 import PdfReader
#import os
import streamlit as st
from dotenv import load_dotenv
from streamlit_extras.add_vertical_space import add_vertical_space
#from transformers import pipeline
#from langchain_community.llms import HuggingFaceEndpoint
#from langchain.chains import LLMChain
#from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain import HuggingFaceHub

# with st.sidebar:
#     st.title(' LLM Chat App')
#     st.markdown('''
#         ## About
#         This app is an LLM-powered chatbot built using:
#         - [Streamlit](https://streamlit.io/)
#         - [Transformers](https://huggingface.co/transformers/)
#         - [Mistral-7B-Instruct-v0.2](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.2) LLM model
#     ''')
#     add_vertical_space(5)
#     st.write('Made By Om Avinash Tarkunde')

# def answer_question(question):
#     prompt = PromptTemplate(input_variables=["question"], template="Answer in short and to the point regarding {question}")
#     prompt.format(question=question)

#     llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.5, max_new_token=128, token=HUGGINGFACEHUB_API_TOKEN)
#     llm_chain = LLMChain(prompt=prompt, llm=llm)

#     answer = llm_chain.invoke(question)
#     return answer


# def main():
#     st.header("Chat with PDF 💬")
    
#     pdf = st.file_uploader("Upload your PDF", type='pdf')
#     if pdf is not None:
#         reader = PdfReader(pdf)
        

#         page = reader.pages[0]

#         text = page.extract_text()

#         #quetion using streamlit
#         question = st.text_input("Ask questions about your PDF file:")

#         if question:
#             prompt = f"""
#                 **Question:** {question}

#                 **Context:**  {text}

#                 **Answer Style:** Short Answer
#             """

#             answer = answer_question(prompt)

#             st.write(f"**Answer:** {answer}")

#         else:
#             st.info("Please enter a question.")
    
#     else:
#         st.info("Please upload a PDF file first.")

# if __name__=="__main__":
#     main()


with st.sidebar:
    st.title(' LLM Chat App')
    st.markdown('''
        ## About
        This app is an LLM-powered chatbot built using:
        - [Streamlit](https://streamlit.io/)
        - [Transformers](https://huggingface.co/transformers/)
        - [Mistral-7B-Instruct-v0.2](https://huggingface.co/google/flan-t5-xxl) LLM model
    ''')
    add_vertical_space(5)
    st.write('Made By Om Avinash Tarkunde')

git
def main():

    st.header("Chat with PDF 💬")
    
    
    load_dotenv()

    pdf = st.file_uploader("Upload your PDF", type='pdf')
    
    if pdf:
        reader = PdfReader(pdf)
        text = ""
        for i in range(len(reader.pages)):
            page = reader.pages[i]
            text += page.extract_text() 
        # page = reader.pages[0]
        # text = page.extract_text()
        text = "\n".join(line for line in text.splitlines() if line.strip())
        # spilit ito chuncks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len)
        chunks = text_splitter.split_text(text)

        embeddings = SentenceTransformerEmbeddings(model_name = "all-MiniLM-L6-v2")

        db = Chroma.from_texts(chunks,embeddings)

        query = st.text_input("Ask questions about your PDF file:")
        if query:
            docs = db.similarity_search(query,k=5)

            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5, "max_length":64})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            st.write(f"**Answer:** {response}")

        else:
            st.info("Please enter a question.")
    
    else:
        st.info("Please upload the pdf")

if __name__=="__main__":
    main()