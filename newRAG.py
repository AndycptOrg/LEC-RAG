# V4 (page size too small on my computer for gemma 2 2B)
import os
from langchain_huggingface import HuggingFacePipeline, HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from dotenv import load_dotenv
load_dotenv()
# --- STEP 1: INITIALISE THE GENERATOR (LLM) ---
# We use a 3B parameter model optimized for conversational text generation
model_id = "meta-llama/Llama-3.2-3B-Instruct"#"google/gemma-2-1b-it"#
# print(os.getenv('HF_TOKEN'))
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

# Configure a text-generation pipeline that automatically hides the prompt input
hf_pipeline = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=150,
    return_full_text=False
)
llm = HuggingFacePipeline(pipeline=hf_pipeline)

# --- STEP 2: SETUP THE RETRIEVER (VECTOR DB) ---
# Create an embedding model to turn text sentences into mathematical vectors
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Your source documents (this can be thousands of pages long)
documents = [
    "Python is an interpreted, high-level, general-purpose programming language.",
    "The capital of France is Paris, known for its art and fashion.",
    "Deep learning neural networks mimic the structural layout of the human brain.",
    "Machine learning models require clean and structured training data to function efficiently.",
    "Artificial intelligence encompasses machine learning, deep learning, and robotics."
]

# Store documents inside an in-memory vector database
vector_store = FAISS.from_texts(documents, embeddings)
retriever = vector_store.as_retriever(search_kwargs={"k": 2}) # Pulls top 2 matches

# --- STEP 3: CONSTRUCT THE PROMPT TEMPLATE ---
# This forces the conversational model to strictly use the retrieved facts
prompt_template = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Answer the question using ONLY the provided context snippets.\n\nContext:\n{context}"),
    ("user", "{question}")
])

# --- STEP 4: LINK EVERYTHING INTO A CHAIN ---
# This connects: User Question -> Database Search -> Prompt Formatting -> LLM Generation
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()}
    | prompt_template
    | llm
    | StrOutputParser()
)

# --- STEP 5: RUN THE PIPELINE ---
query = "Tell me about AI and neural networks"
response = rag_chain.invoke(query)

print("\n--- RAG Response ---")
print(response)
