import faiss
from transformers.utils.logging import disable_progress_bar
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
disable_progress_bar()

from corpus import get_model, get_embedding, get_docs

def search(model, index, queries, k):
    documents = get_docs()

    # 5. Execute a Semantic Search
    query_vector = model.encode(queries).astype('float32')
    faiss.normalize_L2(query_vector)

    # Search for top k closest documents
    similarities, indices = index.search(query_vector, k)

    return [zip(similarities[i], [documents[j] for j in indices[i]]) for i in range(len(queries))]

def query(query, k=10, model_index=None, debug=False, return_docs=False):
    if model_index is None:
        index = get_embedding()
        encoding_model = get_model()
    else:
        encoding_model, index = model_index

    docs = search(encoding_model, index, [query], k)[0]

    documents = []
    # feed query with docs to the LLM for answer generation
    context = ""
    for score, doc in docs:
        context += f"{doc} "
        documents.append(doc)
        if (debug):
            print(score)
    
    if (debug):
        print(query, context)
    
    # V6 slimmed local
    LLM_MODEL = "google/flan-t5-base"#"Qwen/Qwen2.5-3B-Instruct"/"HuggingFaceTB/SmolLM3-3B"
    # TODO: migrate to CasualLM model: summarasation -> rerank -> CasualLM
    tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, tqdm_class=None)
    llm_model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL, tqdm_class=None)
    
    prompt = f"{context}\n Question: {query}\n Answer:"

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    
    with torch.no_grad():
        output_tokens = llm_model.generate(
            **inputs, 
            max_new_tokens=512,
            temperature=0.5,
            do_sample=True
        )

    if (debug):
        print("Tokens layout:", tokenizer.convert_ids_to_tokens(inputs["input_ids"][0]))

    # 5. Decode the newly created tokens back to human text
    response = tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    response = {
        "response": response.strip()
    }

    if (return_docs):
        response["documents"] = documents
    return response


if __name__ == "__main__":
    # models for extracting corpus
    index = get_embedding()

    question = "Tell me about AI and neural networks"
    question = "How do you set up ssh?"
    answer = query(question, k=3, model_index=(get_model(), index), debug=True)
    print("Question:", question)
    print("Answer:", answer)
