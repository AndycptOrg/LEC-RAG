# RAG Experiment

## How to run
1. Clone the repo into a directory and enter
```bash
git clone https://github.com/AndycptOrg/LEC-RAG.git
cd LEC-RAG
```
2. Create and activate virtual environment
```bash
python -m venv .venv
```
For Windows users
```bash
Set-ExecutionPolicy -ExecutionPolicy AllSigned -Scope Process
.venv\Scripts\activate
```
For Linux users
```bash
source .venv/bin/activate
```
3. Load requirements
```bash
python -m pip install -r requirements.txt
```
4. Start server
```bash
fastapi dev
```

Fastapi should tell you where the server is running (typically at http://localhost:8000)

5. Querying
```bash
curl -X 'POST' 'http://localhost:8000/ask?question=How%20to%20reset%20password%3F'
```

## Reflection
### Choices
For general model choices, I decided to only use open weight models so that they are more likely to be available to whoever runs it, and lightweight models as I would like to run the models on my computer to test if the pipline can execute with sound output. A small model also helped with the speed of the response.

For model type, I'm currently using Seq2Seq model. This is because I've read that they perform better for extraction tasks which is what the final answering stage was. This worked well with the shorter documents I used as a template for testing. But when I migrated to a corpus with a larger document size, the model started to fail with queries that the corpus did not have.
Prior to using Seq2Seq, I was tried to use an ExtractiveQA model but it failed to respond to the more open ended example question given. ("How to set up ssh?")
Moving forward, to improve the responses generally, I would like to experiment with a CasualLM model for more flexibity, long context support, and general purpose answers.

I used FAISS for the semantic search layer as it would be simple and contained within the program. But in the future with a more persistant corpus, an actual database such as chromadb or postgres+pgvector would be more preferable as FAISS does not store the document along with the index and requires reindexing the corpus. Further discussion in Pipeline.

I chose to use this dataset becuase it promised to be a typical knowledge item catalog, and it was a decent size and oriented for basic RAG.

The current pipeline goes as follows:
1. when a query is asked of the api, it retrieves the encoding model(Sentence Transformer) and embeded index(FAISS) and performs semantic search over the index wrt the query
2. the top k(10) results are directly injected into a prompt with the query to the LLM(Seq2Seq) for generation
3. the answer is returned via API

I implemented this pipline as this was the most faithful interpretation to the specification given the timeline and constrained by my choices along the way.
In the future, I would like to rework the pipline as follows:
1. the corpus is chunked and layered according to the source document creating a layered embedded index
2. when a query is asked of the api, it retrieves the encoding model(Sentence Transformer) and embeded index(chromaDB/Postgress+pgvector) and performs semantic search over the index wrt the query
3. the top 10 results are summarised(summary model) wrt the query and reincoded into an updated semantic index(FAISS) for reranking
4. the top 3 results are fed along with the query to the LLM(CasualLM) for generation
5. the answer is returned via API

I chose to use FastAPI because it was fast and simple and the I had used it most recently.

I decided not to use langchain becuase for this small pipeline, it only introduced more dependancy than it was worth and with the rapid changes that were made to the pipline, I found the API to not be as stable as I had hoped. Also, I wasn't too familiar with the library anyways, so it wouldn't have been smart to try and force it in.
### Evaluation
queries:
- How to reset password?
- How to fix printer?
- How to set up new employee accounts?

Using precision and recall metrics defined [here](https://docs.ragas.io/en/latest/concepts/metrics/available_metrics/).

| metric | password | printer | employee | 
| --- | --- | --- | --- |
| precision | 0.8 | 0.4 | 1.00 |
| recall | 0.00 | 1.0 | 0.00 |