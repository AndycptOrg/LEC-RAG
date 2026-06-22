from pathlib import Path
import csv
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
# 1. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')
documents = [] # TODO: singleton

def get_model():
    return model

CORPUS_PATH = Path('.') / 'synthetic-it-related-knowledge-items' / 'synthetic_knowledge_items.csv'

def get_docs():
    if len(documents) == 0:
        
        with CORPUS_PATH.open('r', newline='', encoding='utf-8') as file:
            reader = csv.DictReader(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            first_column = [row['ki_text']for row in reader]
            documents.extend(first_column)
        return first_column
    else:
        return documents

def embed_docs(docs):
    # returns faiss index

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
    faiss.write_index(index, INDEX_NAME)

def get_embedding():
    index_file = Path('.') / INDEX_NAME

    if (index_file.exists()):
        index = faiss.read_index(INDEX_NAME)
        get_docs()
    else:
        docs = get_docs()
        index = embed_docs(docs)
        save_embedding(index)
    return index

if __name__ == "__main__":
    get_embedding()