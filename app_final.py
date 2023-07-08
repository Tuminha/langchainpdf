from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.retrievers import SVMRetriever, TFIDFRetriever
from langchain.retrievers.self_query.base import SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.llms import OpenAI
from langchain.retrievers import ContextualCompressionRetriever
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import DocArrayInMemorySearch
from langchain.document_loaders import TextLoader, PyPDFLoader
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import streamlit as st




import os
import openai
import sys
sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key  = os.environ['OPENAI_API_KEY']




# Initialize everything outside the main function 
# to avoid re-initializing them every time the Streamlit app reruns
embedding = OpenAIEmbeddings()
llm = ChatOpenAI(model_name="gpt-4", temperature=0)
persist_directory = 'docs/chroma/'  # specify your persist directory

QA_CHAIN_PROMPT = PromptTemplate.from_template(
    """
    You are a very important professor at a dental school.
    You are answering questions from your students about the course material.
    Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer. 
    Keep the answer as concise as possible. Always say "thanks for asking!" at the end of the answer. 
    {context}
    Question: {question}
   
    Helpful Answer:"""
)


# Streamlit app
# Streamlit app
def main():
    st.title("Chatbot App")

    uploaded_files = st.file_uploader("Upload PDF", type="pdf", accept_multiple_files=True)

    pdf_directory = "docs"

    if uploaded_files:
        pages = []
        for uploaded_file in uploaded_files:
            # Save the uploaded file temporarily
            with open(pdf_directory + uploaded_file.name, "wb") as f:
                f.write(uploaded_file.getbuffer())

            loader = PyPDFLoader(pdf_directory + uploaded_file.name)
            pages += loader.load()
        
            # Remove the file after loading
            os.remove(pdf_directory + uploaded_file.name)
        
        # Split pages
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1500,
            chunk_overlap=150,
            length_function=len
        )

        docs = text_splitter.split_documents(pages)
        splits = text_splitter.split_documents(docs)
        

        # Create an instance of Chroma using the from_documents class method
        vectordb = Chroma.from_documents(
            documents=splits,
            embedding=embedding,
            persist_directory=persist_directory,
        )

        # Print the size of the vectorstore
        st.write("The size of the vectorstore is: ", vectordb._collection.count())

        # Initialize retrieval and memory
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=vectordb.as_retriever(),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_CHAIN_PROMPT},
        )

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )

        retriever=vectordb.as_retriever()
        qa = ConversationalRetrievalChain.from_llm(
            llm,
            retriever=retriever,
            memory=memory
        )

        st.subheader("Chat with the bot")
        user_input = st.text_input("Type your question here...")

        if st.button('Send'):
            if user_input:
                result = qa_chain({"query": user_input})
                st.write(result["result"])


# Run the app
if __name__ == '__main__':
    main()