import os, re, json, time, io
from fastapi import FastAPI, UploadFile, File, Header, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI(title="ScanRobodam API")

# CORS (tweak as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Config via env vars ---
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
ASSISTANT_ID   = os.environ.get("ASSISTANT_ID", "")
VALID_API_KEY  = os.environ.get("VALID_API_KEY", "")

if not OPENAI_API_KEY or not ASSISTANT_ID or not VALID_API_KEY:
    # Fail fast if something critical is missing
    raise RuntimeError("Missing required env vars: OPENAI_API_KEY, ASSISTANT_ID, VALID_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# --- Auth dependency (kept as Bearer) ---
def authenticate(authorization: str = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Nenurodytas API raktas")
    token = authorization.split("Bearer ")[1]
    if token != VALID_API_KEY:
        raise HTTPException(status_code=403, detail="Neteisingas API raktas")

@app.get("/", dependencies=[Depends(authenticate)])
def health():
    return {"zinute": "ScanRobodam veikia!"}

class SaskaitosTekstas(BaseModel):
    tekstas: str

def parse_gpt_json(result_text: str) -> dict:
    raw = result_text.strip()
    if not raw:
        raise ValueError("GPT grąžino tuščią atsakymą.")
    triple_block = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", raw, re.DOTALL)
    if triple_block:
        raw = triple_block.group(1).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    fixed = raw.replace('""', '"')
    try:
        return json.loads(fixed)
    except json.JSONDecodeError as e:
        raise ValueError(f"Nepavyko konvertuoti JSON. Originalus atsakymas:\n{raw}\n\nBandėm fixed:\n{fixed}\n\nKlaida: {str(e)}")

@app.post("/istrauktisaskaita", dependencies=[Depends(authenticate)])
async def is_teksto(ivedimas: SaskaitosTekstas):
    try:
        thread = client.beta.threads.create()
        client.beta.threads.messages.create(thread_id=thread.id, role="user", content=ivedimas.tekstas)
        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)

        # Add a timeout to avoid infinite loops (e.g., 90s)
        t0, timeout = time.time(), 90
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            if run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Asistentas nebaigė darbo: {run_status.status}")
            if time.time() - t0 > timeout:
                raise Exception("Viršytas apdorojimo laukimo laikas")
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        result_raw = messages.data[0].content[0].text.value
        return parse_gpt_json(result_raw)

    except Exception as klaida:
        return JSONResponse(status_code=500, content={"klaida": str(klaida)})

@app.post("/ikeltipdf", dependencies=[Depends(authenticate)])
async def ikelti_pdf(pdf: UploadFile = File(...)):
    try:
        if not pdf.filename.lower().endswith(".pdf"):
            raise Exception("Galima įkelti tik PDF formato failus.")

        failas = await pdf.read()
        pdf_stream = io.BytesIO(failas)

        uploaded = client.files.create(
            file=(pdf.filename, pdf_stream, "application/pdf"),
            purpose="assistants"  # per OpenAI guidance
        )

        thread = client.beta.threads.create()

        client.beta.threads.messages.create(
            thread_id=thread.id,
            role="user",
            content=[{"type": "text", "text": "Prašau ištraukti sąskaitos duomenis iš PDF failo."}],
            # v2 uses 'attachments' with tools for File Search
            attachments=[{"file_id": uploaded.id, "tools": [{"type": "file_search"}]}]
        )

        run = client.beta.threads.runs.create(thread_id=thread.id, assistant_id=ASSISTANT_ID)

        t0, timeout = time.time(), 120
        while True:
            run_status = client.beta.threads.runs.retrieve(thread_id=thread.id, run_id=run.id)
            if run_status.status == "completed":
                break
            if run_status.status in ["failed", "cancelled", "expired"]:
                raise Exception(f"Asistentas nebaigė darbo: {run_status.status}")
            if time.time() - t0 > timeout:
                raise Exception("Viršytas apdorojimo laukimo laikas")
            time.sleep(1)

        messages = client.beta.threads.messages.list(thread_id=thread.id)
        result_raw = messages.data[0].content[0].text.value
        return parse_gpt_json(result_raw)

    except Exception as klaida:
        return JSONResponse(status_code=500, content={"klaida": str(klaida)})
