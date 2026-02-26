from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pdf_utils import extract_text_from_pdf
from embeddings import get_embedding
from rag import store_embedding, search
from openai import OpenAI
import os

app = FastAPI()

# ADD THIS BLOCK
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

@app.get("/")
def root():
    return {"status": "AI Resume Analyzer API is running ✅"}

@app.post("/upload")
async def upload_resume(file: UploadFile = File(...)):
    text = extract_text_from_pdf(file.file)
    embedding = get_embedding(text)
    store_embedding(embedding, text)
    return {"message": "Resume processed successfully", "text_preview": text[:200]}

@app.post("/analyze")
async def analyze_resume(file: UploadFile = File(...), job_description: str = ""):
    # Extract and embed resume
    text = extract_text_from_pdf(file.file)
    embedding = get_embedding(text)
    store_embedding(embedding, text)

    # Build analysis prompt
    job_part = f"\n\nJob Description:\n{job_description}" if job_description else ""
    prompt = f"""You are an expert resume analyst. Analyze this resume{' against the job description' if job_description else ''} and respond ONLY with valid JSON (no markdown, no extra text).

Resume:
{text[:3000]}
{job_part}

Return this exact JSON:
{{
  "score": <number 0-100>,
  "scoreLabel": "<Poor/Fair/Good/Excellent> Match",
  "summary": "<2-3 sentence overall assessment>",
  "matchedSkills": ["skill1", "skill2"],
  "missingSkills": ["skill1", "skill2"],
  "suggestions": ["suggestion1", "suggestion2", "suggestion3", "suggestion4", "suggestion5"]
}}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a resume analyzer. Always respond with valid JSON only."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3,
        max_tokens=1000
    )

    import json
    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    result = json.loads(raw)
    return result

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