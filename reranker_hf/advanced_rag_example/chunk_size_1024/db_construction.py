from tqdm import tqdm
from datasets import load_dataset
from langchain.docstore.document import Document as LangchainDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer
import pickle

# Load dataset
ds = load_dataset("m-ric/huggingface_doc", split="train")

# Prepare raw knowledge base
RAW_KNOWLEDGE_BASE = [
    LangchainDocument(page_content=doc["text"], metadata={"source": doc["source"]})
    for doc in tqdm(ds)
]

# Define tokenizer and text splitter
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)

MARKDOWN_SEPARATORS = [
    "\n#{1,6} ", "```\n", "\n\\*\\*\\*+\n", "\n---+\n", "\n___+\n", "\n\n", "\n", " ", ""
]

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer,
    chunk_size=1024,  # Adjusted to the embedding model
    chunk_overlap=50,
    add_start_index=True,
    strip_whitespace=True,
    separators=MARKDOWN_SEPARATORS,
)

# Split documents into chunks
docs_processed = []
for doc in RAW_KNOWLEDGE_BASE:
    docs_processed += text_splitter.split_documents([doc])

# Remove duplicates
unique_texts = {}
docs_processed_unique = []
for doc in docs_processed:
    if doc.page_content not in unique_texts:
        unique_texts[doc.page_content] = True
        docs_processed_unique.append(doc)

# Save processed documents
with open("chunked_documents.pkl", "wb") as f:
    pickle.dump(docs_processed_unique, f)

print("Document chunking completed and saved to 'chunked_documents.pkl'.")
