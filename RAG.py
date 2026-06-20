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
    input("e")
    # V2 (Extractive QA models cannot answer open-ended queries)
    # from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    # import torch
    # tokenizer = AutoTokenizer.from_pretrained('distilbert-base-cased-distilled-squad')
    # model = AutoModelForQuestionAnswering.from_pretrained('distilbert-base-cased-distilled-squad')

    # inputs = tokenizer(query, context, return_tensors="pt")
    # with torch.no_grad():
    #     outputs = model(**inputs)
    
    # # Extract the start and end positions with the highest logits
    # start_scores = outputs.start_logits
    # end_scores = outputs.end_logits

    # start_index = start_scores.argmax()
    # end_index = end_scores.argmax()

    # print("Start Index:", int(start_index))
    # print("End Index:", int(end_index))
    # print("Tokens layout:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))
    # # Convert tokens back to text
    # print(type(outputs))
    # print(outputs)
    # answer_tokens = inputs["input_ids"][0][start_index : end_index+1]
    # response = tokenizer.decode(answer_tokens)

    # V3
    # from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    # import torch
    # LLM_MODEL = "google/flan-t5-base"
    # tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, tqdm_class=None)
    # model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, tqdm_class=None)
    # prompt = f"Context: {context}\nAnswer the following question using the context provided.\nQuestion: {query}"

    # # 3. Tokenize input sequence
    # inputs = tokenizer(prompt, return_tensors="pt", truncation=True)

    # # 4. Generate the response text tokens (instead of computing raw start/end logits)
    # with torch.no_grad():
    #     output_tokens = model.generate(
    #         **inputs, 
    #         max_new_tokens=100,  # Limits length of generated answer
    #         temperature=0.7,     # Adds mild creativity/fluency
    #         do_sample=True       # Enables text generation sampling
    #     )

    # print("Tokens layout:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    # # 5. Decode the newly created tokens back to human text
    # response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    
    
    # V5 Langchain-ified
    from langchain_huggingface import HuggingFaceEndpoint, HuggingFaceEmbeddings
    from langchain_core.prompts import PromptTemplate
    from langchain_core.runnables import RunnablePassthrough
    from langchain_core.output_parsers import StrOutputParser
    
    LLM_MODEL = "google/flan-t5-base"
    llm = HuggingFaceEndpoint(
        repo_id=LLM_MODEL,
        task="text2text-generation",
        temperature=0.7,
        max_new_tokens=256
    )

    # template = f"Context: {context}\nQuestion: {{question}}\nAnswer:"
    # prompt = PromptTemplate.from_template(template)

    # rag_chain = (
    #     {"question": RunnablePassthrough()}
    #     | prompt
    #     | llm
    #     | StrOutputParser()
    # )
    # response = rag_chain.invoke(query)
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
