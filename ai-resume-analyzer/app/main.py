from fastapi import FastAPI, UploadFile
from pdf_utils import extract_text_from_pdf
from embeddings import get_embedding
from rag import store_embedding, search
from openai import OpenAI
import os

app = FastAPI()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.post("/upload")
async def upload_resume(file: UploadFile):
    text = extract_text_from_pdf(file.file)
    embedding = get_embedding(text)
    store_embedding(embedding, text)
    return {"message": "Resume processed successfully"}

@app.post("/ask")
async def ask_question(question: str):
    query_embedding = get_embedding(question)
    context = search(query_embedding)

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a resume analyzer."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ]
    )

    return {"answer": response.choices[0].message.content}
