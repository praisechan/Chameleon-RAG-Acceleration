import pickle
from langchain.vectorstores import FAISS
import torch
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.utils import DistanceStrategy
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from ragatouille import RAGPretrainedModel
import torch.cuda.nvtx as nvtx
# Load processed documents
with open("chunked_documents.pkl", "rb") as f:
    docs_processed = pickle.load(f)

# Initialize embedding model and vector store
EMBEDDING_MODEL_NAME = "thenlper/gte-small"
embedding_model = HuggingFaceEmbeddings(
    model_name=EMBEDDING_MODEL_NAME,
    multi_process=False,
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

KNOWLEDGE_VECTOR_DATABASE = FAISS.from_documents(
    docs_processed, embedding_model, distance_strategy=DistanceStrategy.COSINE
)

# Initialize Reranker
RERANKER = RAGPretrainedModel.from_pretrained("colbert-ir/colbertv2.0")

# Initialize Reader LLM
READER_MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)
model = AutoModelForCausalLM.from_pretrained(READER_MODEL_NAME, quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained(READER_MODEL_NAME)
READER_LLM = pipeline(
    model=model,
    tokenizer=tokenizer,
    task="text-generation",
    do_sample=True,
    temperature=0.2,
    repetition_penalty=1.1,
    return_full_text=False,
    max_new_tokens=500,
)

# Define RAG prompt template
prompt_in_chat_format = [
    {
        "role": "system",
        "content": "Using the information contained in the context, give a comprehensive answer to the question. Provide the source document number when relevant. If the answer cannot be deduced from the context, do not give an answer.",
    },
    {
        "role": "user",
        "content": "Context:\n{context}\n---\nNow here is the question you need to answer.\n\nQuestion: {question}",
    },
]
RAG_PROMPT_TEMPLATE = tokenizer.apply_chat_template(
    prompt_in_chat_format, tokenize=False, add_generation_prompt=True
)

######################### Warm-up the Reranker#####################
dummy_query = "dummy query"
dummy_documents = ["This is a test document.", "Another dummy document."]
_ = RERANKER.rerank(dummy_query, dummy_documents, k=1)
_ = READER_LLM(dummy_query)
###################################################################

# Retrieval and generation function
def answer_with_rag(question: str, num_retrieved_docs: int = 20, num_docs_final: int = 5):
    # Retrieve documents
    nvtx.range_push("retrieval")
    relevant_docs = KNOWLEDGE_VECTOR_DATABASE.similarity_search(
        query=question, k=num_retrieved_docs
    )
    relevant_docs = [doc.page_content for doc in relevant_docs]  # Keep only the text
    print(f"retrieved doc num : {len(relevant_docs)}")
    nvtx.range_pop()
    
    nvtx.range_push("reranker call")
    if RERANKER:
        relevant_docs = RERANKER.rerank(question, relevant_docs, k=num_docs_final)
        relevant_docs = [doc["content"] for doc in relevant_docs]
    nvtx.range_pop()
    
    # Build prompt
    nvtx.range_push("prompt build")
    context = "\nExtracted documents:\n"
    context += "".join(
        [f"Document {str(i)}:::\n" + doc for i, doc in enumerate(relevant_docs)]
    )
    
    final_prompt = RAG_PROMPT_TEMPLATE.format(question=question, context=context)
    nvtx.range_pop()

    # Generate answer
    nvtx.range_push("LLM")
    answer = READER_LLM(final_prompt)[0]["generated_text"]
    nvtx.range_pop()

    return answer, relevant_docs

# Test query
question = "how to create a pipeline object?"
answer, relevant_docs = answer_with_rag(question)

print("Answer:", answer)
