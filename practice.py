from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain import HuggingFaceHub
from dotenv import load_dotenv
from langchain.chains.question_answering import load_qa_chain
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma




def main():
    load_dotenv()
    reader = PdfReader("Om Tarkunde Resume_offcampus.pdf")
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

    while True:
        query = input("your input -> ")
        if query:
            docs = db.similarity_search(query,k=5)

            llm = HuggingFaceHub(repo_id="google/flan-t5-large", model_kwargs={"temperature":5, "max_length":64})
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=query)
            print("chatpdf> " + response)

if __name__ == '__main__':
    main()


