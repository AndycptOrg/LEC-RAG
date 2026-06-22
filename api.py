from fastapi import FastAPI

app = FastAPI(title="LEC RAG")

@app.get("/health")
def health():
    return {"ok": True}

@app.post("/ask")
async def ask(
    question: str,
    docs: bool=False
):
    from RAG import query
    response = query(question, return_docs=docs)
    return response

if __name__ == "__main__":
    app