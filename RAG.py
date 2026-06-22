import faiss
from transformers.utils.logging import disable_progress_bar
disable_progress_bar()

def search(model, index, queries, k):
    from corpus import documents
    # 5. Execute a Semantic Search
    query_vector = model.encode(queries).astype('float32')
    faiss.normalize_L2(query_vector)

    # Search for top k closest documents
    similarities, indices = index.search(query_vector, k)

    return [zip(similarities[i], [documents[j] for j in indices[i]]) for i in range(len(queries))]

def query(query, k=10, model_index=None, debug=False):

    # import ollama
    if model_index is None:
        from corpus import model, get_embedding
        index = get_embedding()
    else:
        model, index = model_index

    docs = search(model, index, [query], k)[0]
    
    # feed query with docs to the LLM for answer generation
    context = "" #f"Answer the following question based on the provided documents:\n\nQuestion: {query}\n\nDocuments:\n"
    for score, doc in docs:
        context += f"{doc} "# (Score: {score:.4f})\n"
        if (debug):
            print(score)
    
    if (debug):
        print(query, context)
    
    # V6 slimmed local
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
    import torch

    LLM_MODEL = "google/flan-t5-base"#"Qwen/Qwen2.5-3B-Instruct"/"HuggingFaceTB/SmolLM3-3B"
    # TODO: migrate to CasualLM model: summarasation -> rerank -> CasualLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, tqdm_class=None)
    model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, tqdm_class=None)
    
    prompt = f"{context}\n Question: {query}\n Answer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        output_tokens = model.generate(
            **inputs, 
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

    if (debug):
        print("Tokens layout:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    # 5. Decode the newly created tokens back to human text
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    # response = ollama.generate(model="llama3", prompt=prompt)
    return response.strip() #['answer']


if __name__ == "__main__":
    # models for extracting corpus
    from corpus import model, get_embedding
    index = get_embedding()

    question = "Tell me about AI and neural networks"
    question = "How do you set up ssh?"
    answer = query(question, k=3, model_index=(model, index), debug=True)
    print("Question:", question)
    print("Answer:", answer)
