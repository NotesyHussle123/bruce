from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
import os
import numpy as np

from dotenv import load_dotenv
from supabase import create_client
import openai
from langchain_groq import ChatGroq
from langchain.schema import HumanMessage

# Load env vars
load_dotenv()

# === Setup ===
app = FastAPI()

# Env vars
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

openai.api_key = OPENAI_API_KEY
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
groq_llm = ChatGroq(groq_api_key=GROQ_API_KEY, model_name="llama3-8b-8192")

# === Data Models ===
class QueryRequest(BaseModel):
    question: str
    top_k: Optional[int] = 3

# === Helper Functions ===

def get_query_embedding(question, model="text-embedding-3-small"):
    response = openai.embeddings.create(input=[question], model=model)
    return response.data[0].embedding

def search_similar_documents(query_embedding, top_k=3):
    embedding_str = f"ARRAY{np.array(query_embedding).tolist()}"
    sql = f"""
    SELECT source_table, source_id, content, embedding <-> ({embedding_str})::vector AS similarity
    FROM rag_embeddings
    ORDER BY similarity ASC
    LIMIT {top_k}
    """
    response = supabase.rpc("execute_sql", {"sql": sql}).execute()
    return response.data or []

def format_context(matches):
    return "\n\n".join([match["content"] for match in matches])

def ask_groq(question, matches):
    context = format_context(matches)

    prompt = f"""
You are an internal assistant designed to help staff understand and work with company data.
You have access to information about customers, products, and orders.

Answer the question below using only the provided context.
Be concise, accurate, and professional.
If the answer cannot be found in the context, respond with "I'm not sure based on the current data."

Context:
{context}

Question:
{question}
"""

    response = groq_llm.invoke([HumanMessage(content=prompt)])
    return response.content.strip()

# === API Route ===

@app.post("/query")
def query_rag(req: QueryRequest):
    try:
        query_embedding = get_query_embedding(req.question)
        matches = search_similar_documents(query_embedding, top_k=req.top_k)
        answer = ask_groq(req.question, matches)
        return {"answer": answer, "matches": matches}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# === Local Dev Entrypoint ===
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
