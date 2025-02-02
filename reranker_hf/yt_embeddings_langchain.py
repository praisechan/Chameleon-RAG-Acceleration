# -*- coding: utf-8 -*-
"""YT_Embeddings_LangChain.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1NaEyuFWCkDtkufHIsWQhfgPLHybXeYA1

### Information Retreival - LangChain
- Without Using OpenAI Embeddings
- Without OpenAI LLM

Two Applications:
- Text Documents
- Multiple PDF Files
"""

"""### Get HUGGINGFACEHUB_API_KEY"""

import os
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "HUGGINGFACEHUB_API_TOKEN"


########################### 1. Download from url and RAG#################################
import requests

url = "https://raw.githubusercontent.com/hwchase17/langchain/master/docs/modules/state_of_the_union.txt"
res = requests.get(url)
with open("state_of_the_union.txt", "w") as f:
  f.write(res.text)

# Document Loader
from langchain.document_loaders import TextLoader
loader = TextLoader('./state_of_the_union.txt')
documents = loader.load()

documents

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')

    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]

    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)

    return wrapped_text

print(wrap_text_preserve_newlines(str(documents[0])))


# Text Splitter
from langchain.text_splitter import CharacterTextSplitter
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
docs = text_splitter.split_documents(documents)

len(docs)

docs[0]

"""### Embeddings"""

# Embeddings
from langchain.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings()

# Vectorstore: https://python.langchain.com/en/latest/modules/indexes/vectorstores.html
from langchain.vectorstores import FAISS

db = FAISS.from_documents(docs, embeddings)

query = "What did the president say about the Supreme Court"
docs = db.similarity_search(query)

print(wrap_text_preserve_newlines(str(docs[0].page_content)))

"""### Create QA Chain"""
from langchain.chains.question_answering import load_qa_chain
from langchain import HuggingFaceHub

llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

chain = load_qa_chain(llm, chain_type="stuff")

query = "What did the president say about the Supreme Court"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)

query = "What did the president say about economy?"
docs = db.similarity_search(query)
chain.run(input_documents=docs, question=query)


########################### 2. Working with PDF Files#################################
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.indexes import VectorstoreIndexCreator

# connect your Google Drive
from google.colab import drive
drive.mount('/content/gdrive', force_remount=True)

pdf_folder_path = '/content/gdrive/My Drive/data_2/'
os.listdir(pdf_folder_path)

loaders = [UnstructuredPDFLoader(os.path.join(pdf_folder_path, fn)) for fn in os.listdir(pdf_folder_path)]
loaders

index = VectorstoreIndexCreator(
    embedding=HuggingFaceEmbeddings(),
    text_splitter=CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)).from_loaders(loaders)

llm = HuggingFaceHub(repo_id="google/flan-t5-xl", model_kwargs={"temperature":0, "max_length":512})

from langchain.chains import RetrievalQA
chain = RetrievalQA.from_chain_type(llm=llm,
                                    chain_type="stuff",
                                    retriever=index.vectorstore.as_retriever(),
                                    input_key="question")

chain.run('How was the GPT4all model trained?')
chain.run('Who are the authors of GPT4all technical report?')
chain.run('What is the model size of GPT4all?')