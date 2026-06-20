import faiss
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

# 2. Sample preprocessed corpus chunks
documents = [
    "Python is an interpreted, high-level, general-purpose programming language.",
    "The capital of France is Paris, known for its art and fashion.",
    "Machine learning models require clean and structured training data.",
    "Deep learning neural networks mimic the structure of the human brain.",
    "Artificial intelligence encompasses machine learning, deep learning, and robotics."
]

def load_model():
    from sentence_transformers import SentenceTransformer
    import numpy as np  
    # 1. Load the embedding model
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # 3. Generate embeddings and convert to float32 (required by FAISS)
    document_embeddings = model.encode(documents)
    embedding_matrix = np.array(document_embeddings).astype('float32')

    # 4. Build a FAISS Index for inner product / cosine similarity
    dimension = embedding_matrix.shape[1]
    # Normalize vectors to use IndexFlatIP for true cosine similarity
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)
    return model, index

def search(model, index, queries, k):
    # 5. Execute a Semantic Search
    query_vector = model.encode(queries).astype('float32')
    faiss.normalize_L2(query_vector)

    # Search for top k closest documents
    similarities, indices = index.search(query_vector, k)

    return [zip(similarities[i], [documents[j] for j in indices[i]]) for i in range(len(queries))]

def query(query, k, model_index=None):

    # import ollama
    if model_index is None:
        model, index = load_model()
    else:
        model, index = model_index

    docs = search(model, index, [query], k)[0]
    
    # feed query with docs to the LLM for answer generation
    context = "" #f"Answer the following question based on the provided documents:\n\nQuestion: {query}\n\nDocuments:\n"
    for score, doc in docs:
        context += f"{doc}. "# (Score: {score:.4f})\n"

    print(query, context)
    
    # V6 slimmed local langchain?
    from langchain_huggingface import HuggingFacePipeline
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
    
    LLM_MODEL = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, tqdm_class=None)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, tqdm_class=None)
    
    pipe = pipeline(
        "text-generation",
        model=LLM_MODEL,
        tokenizer=tokenizer,
        max_new_tokens=256
    )
    
    llm = HuggingFacePipeline(
        pipeline=pipe
    )

    response = llm.invoke(f"Context: {context}\n Question: {query}\n Answer:")
    print("Answer:", response)

    # response = ollama.generate(model="llama3", prompt=prompt)
    return response.strip() #['answer']

# models for extracting corpus
model, index = load_model()

# docs = search(model, index, [query], k=2)

# # 6. Output the best matches
# print(f"Query: '{query}'\n")
# for score, doc in docs[0]:
#     print(f"Score: {score:.4f} -> Document: {doc}")

print(query("Tell me about AI and neural networks", k=3, model_index=(model, index)))
