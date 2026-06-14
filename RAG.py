import numpy as np
from sentence_transformers import SentenceTransformer
import faiss

# 1. Load the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# 2. Sample preprocessed corpus chunks
documents = [
    "Python is an interpreted, high-level, general-purpose programming language.",
    "The capital of France is Paris, known for its art and fashion.",
    "Machine learning models require clean and structured training data.",
    "Deep learning neural networks mimic the structure of the human brain."
]

# 3. Generate embeddings and convert to float32 (required by FAISS)
document_embeddings = model.encode(documents)
embedding_matrix = np.array(document_embeddings).astype('float32')

# 4. Build a FAISS Index for inner product / cosine similarity
dimension = embedding_matrix.shape[1]
# Normalize vectors to use IndexFlatIP for true cosine similarity
faiss.normalize_L2(embedding_matrix)
index = faiss.IndexFlatIP(dimension)
index.add(embedding_matrix)

# 5. Execute a Semantic Search
query = "Tell me about AI and neural networks"
query_vector = model.encode([query]).astype('float32')
faiss.normalize_L2(query_vector)

# Search for top 2 closest documents
k = 2 
similarities, indices = index.search(query_vector, k)

# 6. Output the best matches
print(f"Query: '{query}'\n")
for idx, score in zip(indices[0], similarities[0]):
    print(f"Score: {score:.4f} -> Document: {documents[idx]}")