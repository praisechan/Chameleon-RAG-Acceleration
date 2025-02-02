# -*- coding: utf-8 -*-
"""rag_zephyr_langchain.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/github/huggingface/cookbook/blob/main/notebooks/en/rag_zephyr_langchain.ipynb

# Simple RAG for GitHub issues using Hugging Face Zephyr and LangChain

_Authored by: [Maria Khalusova](https://github.com/MKhalusova)_

This notebook demonstrates how you can quickly build a RAG (Retrieval Augmented Generation) for a project's GitHub issues using [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta) model, and LangChain.


**What is RAG?**

RAG is a popular approach to address the issue of a powerful LLM not being aware of specific content due to said content not being in its training data, or hallucinating even when it has seen it before. Such specific content may be proprietary, sensitive, or, as in this example, recent and updated often.

If your data is static and doesn't change regularly, you may consider fine-tuning a large model. In many cases, however, fine-tuning can be costly, and, when done repeatedly (e.g. to address data drift), leads to "model shift". This is when the model's behavior changes in ways that are not desirable.

**RAG (Retrieval Augmented Generation)** does not require model fine-tuning. Instead, RAG works by providing an LLM with additional context that is retrieved from relevant data so that it can generate a better-informed response.

Here's a quick illustration:

![RAG diagram](https://huggingface.co/datasets/huggingface/cookbook-images/resolve/main/rag-diagram.png)

* The external data is converted into embedding vectors with a separate embeddings model, and the vectors are kept in a database. Embeddings models are typically small, so updating the embedding vectors on a regular basis is faster, cheaper, and easier than fine-tuning a model.

* At the same time, the fact that fine-tuning is not required gives you the freedom to swap your LLM for a more powerful one when it becomes available, or switch to a smaller distilled version, should you need faster inference.

Let's illustrate building a RAG using an open-source LLM, embeddings model, and LangChain.

First, install the required dependencies:
"""

!pip install -q faiss-gpu

# If running in Google Colab, you may need to run this cell to make sure you're using UTF-8 locale to install LangChain
import locale
locale.getpreferredencoding = lambda: "UTF-8"

"""## Prepare the data

In this example, we'll load all of the issues (both open and closed) from [PEFT library's repo](https://github.com/huggingface/peft).

First, you need to acquire a [GitHub personal access token](https://github.com/settings/tokens?type=beta) to access the GitHub API.
"""

from getpass import getpass
ACCESS_TOKEN = getpass("YOUR_GITHUB_PERSONAL_TOKEN")

"""Next, we'll load all of the issues in the [huggingface/peft](https://github.com/huggingface/peft) repo:
- By default, pull requests are considered issues as well, here we chose to exclude them from data with by setting `include_prs=False`
- Setting `state = "all"` means we will load both open and closed issues.
"""

from langchain.document_loaders import GitHubIssuesLoader

loader = GitHubIssuesLoader(
    repo="huggingface/peft",
    access_token=ACCESS_TOKEN,
    include_prs=False,
    state="all"
)

docs = loader.load()

"""The content of individual GitHub issues may be longer than what an embedding model can take as input. If we want to embed all of the available content, we need to chunk the documents into appropriately sized pieces.

The most common and straightforward approach to chunking is to define a fixed size of chunks and whether there should be any overlap between them. Keeping some overlap between chunks allows us to preserve some semantic context between the chunks. The recommended splitter for generic text is the [RecursiveCharacterTextSplitter](https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter), and that's what we'll use here.
"""

from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=30)

chunked_docs = splitter.split_documents(docs)

"""## Create the embeddings + retriever

Now that the docs are all of the appropriate size, we can create a database with their embeddings.

To create document chunk embeddings we'll use the `HuggingFaceEmbeddings` and the [`BAAI/bge-base-en-v1.5`](https://huggingface.co/BAAI/bge-base-en-v1.5) embeddings model. There are many other embeddings models available on the Hub, and you can keep an eye on the best performing ones by checking the [Massive Text Embedding Benchmark (MTEB) Leaderboard](https://huggingface.co/spaces/mteb/leaderboard).


To create the vector database, we'll use `FAISS`, a library developed by Facebook AI. This library offers efficient similarity search and clustering of dense vectors, which is what we need here. FAISS is currently one of the most used libraries for NN search in massive datasets.

We'll access both the embeddings model and FAISS via LangChain API.
"""

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

db = FAISS.from_documents(chunked_docs,
                          HuggingFaceEmbeddings(model_name='BAAI/bge-base-en-v1.5'))

"""We need a way to return(retrieve) the documents given an unstructured query. For that, we'll use the `as_retriever` method using the `db` as a backbone:
- `search_type="similarity"` means we want to perform similarity search between the query and documents
- `search_kwargs={'k': 4}` instructs the retriever to return top 4 results.

"""

retriever = db.as_retriever(
    search_type="similarity",
    search_kwargs={'k': 4}
)

"""The vector database and retriever are now set up, next we need to set up the next piece of the chain - the model.

## Load quantized model

For this example, we chose [`HuggingFaceH4/zephyr-7b-beta`](https://huggingface.co/HuggingFaceH4/zephyr-7b-beta), a small but powerful model.

With many models being released every week, you may want to substitute this model to the latest and greatest. The best way to keep track of open source LLMs is to check the [Open-source LLM leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard).

To make inference faster, we will load the quantized version of the model:
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

model_name = 'HuggingFaceH4/zephyr-7b-beta'

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(model_name)

"""## Setup the LLM chain

Finally, we have all the pieces we need to set up the LLM chain.

First, create a text_generation pipeline using the loaded model and its tokenizer.

Next, create a prompt template - this should follow the format of the model, so if you substitute the model checkpoint, make sure to use the appropriate formatting.
"""

from langchain.llms import HuggingFacePipeline
from langchain.prompts import PromptTemplate
from transformers import pipeline
from langchain_core.output_parsers import StrOutputParser

text_generation_pipeline = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    temperature=0.2,
    do_sample=True,
    repetition_penalty=1.1,
    return_full_text=True,
    max_new_tokens=400,
)

llm = HuggingFacePipeline(pipeline=text_generation_pipeline)

prompt_template = """
<|system|>
Answer the question based on your knowledge. Use the following context to help:

{context}

</s>
<|user|>
{question}
</s>
<|assistant|>

 """

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template,
)

llm_chain = prompt | llm | StrOutputParser()

"""Note: _You can also use `tokenizer.apply_chat_template` to convert a list of messages (as dicts: `{'role': 'user', 'content': '(...)'}`) into a string with the appropriate chat format._


Finally, we need to combine the `llm_chain` with the retriever to create a RAG chain. We pass the original question through to the final generation step, as well as the retrieved context docs:
"""

from langchain_core.runnables import RunnablePassthrough

retriever = db.as_retriever()

rag_chain = (
 {"context": retriever, "question": RunnablePassthrough()}
    | llm_chain
)

"""## Compare the results

Let's see the difference RAG makes in generating answers to the library-specific questions.
"""

question = "How do you combine multiple adapters?"

"""First, let's see what kind of answer we can get with just the model itself, no context added:"""

llm_chain.invoke({"context":"", "question": question})

"""As you can see, the model interpreted the question as one about physical computer adapters, while in the context of PEFT, "adapters" refer to LoRA adapters.
Let's see if adding context from GitHub issues helps the model give a more relevant answer:
"""

rag_chain.invoke(question)

"""As we can see, the added context, really helps the exact same model, provide a much more relevant and informed answer to the library-specific question.

Notably, combining multiple adapters for inference has been added to the library, and one can find this information in the documentation, so for the next iteration of this RAG it may be worth including documentation embeddings.
"""