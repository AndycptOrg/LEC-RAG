from sentence_transformers import SentenceTransformer
# 1. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = []

def get_docs():
    import csv

    with open('synthetic-it-related-knowledge-items\synthetic_knowledge_items.csv', 'r', newline='', encoding='utf-8') as file:
        reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        first_column = [row['ki_text']for row in reader]
        documents.extend(first_column)
    return first_column

def embed_docs(docs):
    # returns faiss index
    import faiss
    import numpy as np  

    # 3. Generate embeddings and convert to float32 (required by FAISS)
    document_embeddings = model.encode(docs)
    embedding_matrix = np.array(document_embeddings).astype('float32')

    # 4. Build a FAISS Index for inner product / cosine similarity
    dimension = embedding_matrix.shape[1]
    # Normalize vectors to use IndexFlatIP for true cosine similarity
    faiss.normalize_L2(embedding_matrix)
    index = faiss.IndexFlatIP(dimension)
    index.add(embedding_matrix)
    
    return index

INDEX_NAME = "embedded_docs.index"

def save_embedding(index):
    import faiss
    faiss.write_index(index, INDEX_NAME)

def get_embedding():
    import os
    index_file = os.path.join(os.path.dirname(__file__), INDEX_NAME)

    if (os.path.exists(index_file)):
        import faiss
        index = faiss.read_index(INDEX_NAME)
        get_docs()
    else:
        docs = get_docs()
        index = embed_docs(docs)
        save_embedding(index)
    return index

if __name__ == "__main__":
    get_embedding()