import os
import openai
import sys
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

embedding = OpenAIEmbeddings(openai_api_key="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")

sys.path.append('../..')

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv()) # read local .env file

openai.api_key = os.environ['OPENAI_API_KEY']

# The course will show the pip installs you would need to install packages on your own machine.
# These packages are already installed on this platform and should not be run again.
# !pip install pypdf 

from langchain.document_loaders import PyPDFLoader

# Change the path to the PDF you want to load
loader = PyPDFLoader("docs/The 17 immutable laws In Implant Dentistry.pdf")
pages = loader.load()

len(pages)

from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1500,
    chunk_overlap=150,
    length_function=len
)

docs = text_splitter.split_documents(pages)

splits = text_splitter.split_documents(docs)

len(docs)

print(len(docs))

# Vectorstores
persist_directory = "docs/chroma"
embedding = OpenAIEmbeddings()
vectordb = Chroma(persist_directory=persist_directory, embedding_function=embedding)

print(vectordb._collection.count())

question = "What are the clinical benefits of Platform Switching?"
llm = ChatOpenAI(model_name="gpt-4", temperature=0.3)

qa_chain = RetrievalQA.from_chain_type(
    llm,
    retriever=vectordb.as_retriever(),
)

result = qa_chain({"query": question})

result["result"]

print(result["result"])
