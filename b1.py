# ai_librarian_backend.py
"""
FastAPI backend for AI Librarian app.
Features:
- Upload PDFs and store on disk
- Extract text from PDFs
- Create embeddings (sentence-transformers) and index with FAISS
- Search by keyword and return best-matching PDFs + download link

Run:
    pip install -r requirements.txt
    uvicorn ai_librarian_backend:app --host 0.0.0.0 --port 8000 --reload

Files created at runtime:
- ./data/pdfs/       (saved PDF files)
- ./data/meta.json   (metadata mapping of doc_id -> info)
- ./data/index.faiss (faiss index)

Note: choose a machine with enough RAM. This implementation keeps a simple per-document embedding (mean of sentence embeddings) for efficient document-level search.
"""
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Depends
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
from transformers import BartForConditionalGeneration, BartTokenizer
import torch
from jose import JWTError, jwt
from passlib.context import CryptContext
from datetime import datetime, timedelta
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
import os
import uuid
import shutil
import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import tempfile
import PyPDF2



# -----------------------------
# Authentication Configuration
# -----------------------------
SECRET_KEY = "super_secret_key_change_this"  # Change this to something secure
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24  # 1 day

pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="login")

def hash_password(password: str):
    return pwd_context.hash(password)

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def create_access_token(data: dict, expires_delta: timedelta | None = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

def get_current_user(token: str = Depends(oauth2_scheme)):
    credentials_exception = HTTPException(
        status_code=401,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # ✅ Always reload the users DB from file
    users_path = os.path.join(DATA_DIR, "users_db.json")
    if not os.path.exists(users_path):
        raise credentials_exception

    with open(users_path, "r", encoding="utf-8") as f:
        users = json.load(f)

    user = next((u for u in users if u["username"] == username), None)
    if user is None:
        raise credentials_exception

    return user


# --- Configuration ---
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
PDF_DIR = os.path.join(DATA_DIR, "pdfs")
META_PATH = os.path.join(DATA_DIR, "meta.json")
INDEX_PATH = os.path.join(DATA_DIR, "index.faiss")
USER_DIR = os.path.join(DATA_DIR, "users")
os.makedirs(USER_DIR, exist_ok=True)
EMBD_DTYPE = "float32"
MODEL_NAME = "all-MiniLM-L6-v2"  # small & effective
# --- Summarization model ---
SUMMARIZER_MODEL = "facebook/bart-large-cnn"
tokenizer = BartTokenizer.from_pretrained(SUMMARIZER_MODEL)
summarizer_model = BartForConditionalGeneration.from_pretrained(SUMMARIZER_MODEL)
USERS_DB_PATH = os.path.join(DATA_DIR, "users_db.json")


os.makedirs(PDF_DIR, exist_ok=True)

app = FastAPI(title="AI Librarian Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utilities / State ---
model = SentenceTransformer(MODEL_NAME)
embed_dim = model.get_sentence_embedding_dimension()

# metadata mapping: list of dicts with keys: id, filename, original_filename, text, upload_ts
# we'll persist minimal metadata to disk (no full text required, but keep short excerpt)
if os.path.exists(META_PATH):
    with open(META_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)
else:
    metadata = []

if os.path.exists(USERS_DB_PATH):
    with open(USERS_DB_PATH, "r", encoding="utf-8") as f:
        users_db = json.load(f)
else:
    users_db = []



# Load or initialize FAISS index. We'll use IndexFlatIP (cosine via normalized vectors)
index = None
id_to_idx = {}  # mapping from doc_id to row index in faiss


def save_meta():
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False, indent=2)


def extract_text_from_pdf(path: str) -> str:
    """Extract text from PDF using PyPDF2. Returns concatenated text."""
    text_parts = []
    try:
        with open(path, "rb") as fh:
            reader = PyPDF2.PdfReader(fh)
            for page in reader.pages:
                try:
                    page_text = page.extract_text() or ""
                except Exception:
                    page_text = ""
                text_parts.append(page_text)
    except Exception as e:
        print("PDF read error:", e)
    return "\n".join(text_parts).strip()


def initialize_index():
    global index, id_to_idx
    n = len(metadata)
    if os.path.exists(INDEX_PATH) and n > 0:
        try:
            index = faiss.read_index(INDEX_PATH)
            # rebuild id_to_idx mapping from metadata order
            id_to_idx = {m['id']: i for i, m in enumerate(metadata)}
            print(f"Loaded FAISS index from {INDEX_PATH}, documents={n}")
            return
        except Exception as e:
            print("Failed to load FAISS index, creating new. Error:", e)
    # create fresh index
    index = faiss.IndexFlatIP(embed_dim)  # inner product on normalized vectors = cosine
    if n > 0:
        # compute embeddings array from metadata stored embeddings if present
        emb_list = []
        for i, m in enumerate(metadata):
            # metadata may contain 'embedding' (saved) or not; if not, compute now
            if 'embedding' in m:
                emb = np.array(m['embedding'], dtype=EMBD_DTYPE)
            else:
                # extract file and compute (expensive)
                pdfpath = os.path.join(PDF_DIR, m['filename'])
                text = extract_text_from_pdf(pdfpath)
                emb = compute_doc_embedding(text)
                m['embedding'] = emb.tolist()
            emb_list.append(emb)
        emb_matrix = np.vstack(emb_list).astype(EMBD_DTYPE)
        # normalize
        faiss.normalize_L2(emb_matrix)
        index.add(emb_matrix)
        id_to_idx = {m['id']: i for i, m in enumerate(metadata)}
        save_meta()
    else:
        id_to_idx = {}


def compute_doc_embedding(text: str) -> np.ndarray:
    # split text into chunks to avoid truncation; here we do naive splitting by 500 tokens ~ 500 words
    if not text:
        return np.zeros((embed_dim,), dtype=EMBD_DTYPE)
    words = text.split()
    chunk_size = 400
    chunks = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size)]
    embs = model.encode(chunks, show_progress_bar=False, convert_to_numpy=True)
    # mean pooling
    emb = np.mean(embs, axis=0)
    return emb.astype(EMBD_DTYPE)



def summarize_long_text(text, chunk_size=3000):
    """
    Summarize long text by splitting into chunks and summarizing each,
    then summarizing the combined summaries.
    """
    if not text.strip():
        return "No text found for summarization."

    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    partial_summaries = []

    for chunk in chunks:
        inputs = tokenizer([chunk], max_length=1024, truncation=True, return_tensors="pt")
        summary_ids = summarizer_model.generate(
            inputs["input_ids"],
            num_beams=4,
            length_penalty=2.0,
            max_length=250,
            min_length=80,
            early_stopping=True,
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        partial_summaries.append(summary)

    # Combine partial summaries and summarize again
    combined = " ".join(partial_summaries)
    inputs = tokenizer([combined], max_length=1024, truncation=True, return_tensors="pt")
    final_ids = summarizer_model.generate(
        inputs["input_ids"],
        num_beams=4,
        length_penalty=2.0,
        max_length=350,
        min_length=100,
        early_stopping=True,
    )
    final_summary = tokenizer.decode(final_ids[0], skip_special_tokens=True)
    return final_summary




initialize_index()


# --- Pydantic models ---
class UploadResponse(BaseModel):
    id: str
    filename: str
    original_filename: str


class PdfInfo(BaseModel):
    id: str
    filename: str
    original_filename: str
    excerpt: Optional[str]


class SearchRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5


class SearchResultItem(BaseModel):
    id: str
    filename: str
    original_filename: str
    score: float
    download_url: str
    excerpt: Optional[str]


from fastapi import Depends

@app.post("/signup")
def signup(username: str = Form(...), password: str = Form(...)):
    # check if user already exists
    if next((u for u in users_db if u["username"] == username), None):
        raise HTTPException(status_code=400, detail="Username already exists")

    # hash password and store
    hashed_pw = hash_password(password)
    users_db.append({"username": username, "hashed_password": hashed_pw})

    # ✅ Add this block to make users persistent
    USERS_DB_PATH = os.path.join(DATA_DIR, "users_db.json")
    with open(USERS_DB_PATH, "w", encoding="utf-8") as f:
        json.dump(users_db, f, indent=2)

    return {"message": f"User '{username}' created successfully"}


from fastapi.security import OAuth2PasswordRequestForm

@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    username = form_data.username
    password = form_data.password

    # 🔄 Load users from JSON file every time
    users_path = os.path.join(DATA_DIR, "users_db.json")
    if not os.path.exists(users_path):
        raise HTTPException(status_code=400, detail="No registered users found")

    with open(users_path, "r", encoding="utf-8") as f:
        users_db = json.load(f)

    user = next((u for u in users_db if u["username"] == username), None)
    if not user:
        raise HTTPException(status_code=400, detail="Invalid username or password")

    if not verify_password(password, user["hashed_password"]):
        raise HTTPException(status_code=400, detail="Invalid username or password")

    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}



# --- Endpoints ---
@app.post("/upload", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...), current_user: dict = Depends(get_current_user)):
    if file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    username = current_user["username"]

    # 🟢 create user-specific folder
    user_folder = os.path.join(USER_DIR, username)
    pdf_folder = os.path.join(user_folder, "pdfs")
    os.makedirs(pdf_folder, exist_ok=True)

    user_meta_path = os.path.join(user_folder, "meta.json")
    user_index_path = os.path.join(user_folder, "index.faiss")

    # 🧱 Load or initialize user's metadata and index
    if os.path.exists(user_meta_path):
        with open(user_meta_path, "r", encoding="utf-8") as f:
            user_metadata = json.load(f)
    else:
        user_metadata = []

    # FAISS setup
    if os.path.exists(user_index_path) and len(user_metadata) > 0:
        index = faiss.read_index(user_index_path)
    else:
        index = faiss.IndexFlatIP(embed_dim)

    # Save the PDF in user's folder
    unique_id = str(uuid.uuid4())
    stored_name = f"{unique_id}.pdf"
    dest_path = os.path.join(pdf_folder, stored_name)
    with open(dest_path, "wb") as out_f:
        shutil.copyfileobj(file.file, out_f)

    # Extract text & embedding
    text = extract_text_from_pdf(dest_path)
    excerpt = (text[:800] + "...") if len(text) > 800 else text
    emb = compute_doc_embedding(text)
    emb_norm = emb.copy()
    faiss.normalize_L2(emb_norm.reshape(1, -1))
    index.add(emb_norm.reshape(1, -1))

    # Append metadata
    user_metadata.append({
        "id": unique_id,
        "filename": stored_name,
        "original_filename": file.filename,
        "excerpt": excerpt,
        "embedding": emb.tolist(),
        "owner": username
    })

    # Save metadata and FAISS index
    with open(user_meta_path, "w", encoding="utf-8") as f:
        json.dump(user_metadata, f, indent=2)
    faiss.write_index(index, user_index_path)

    return UploadResponse(id=unique_id, filename=stored_name, original_filename=file.filename)



@app.get("/pdfs", response_model=List[PdfInfo])
def list_pdfs(current_user: dict = Depends(get_current_user)):
    username = current_user["username"]

    # 🧭 Locate the user’s folder
    user_folder = os.path.join(USER_DIR, username)
    user_meta_path = os.path.join(user_folder, "meta.json")

    # 🧱 Check if metadata exists
    if not os.path.exists(user_meta_path):
        return []  # No uploads yet

    # 🧾 Load metadata for this user only
    with open(user_meta_path, "r", encoding="utf-8") as f:
        user_metadata = json.load(f)

    # 🧩 Convert metadata into PdfInfo models
    return [
        PdfInfo(
            id=m["id"],
            filename=m["filename"],
            original_filename=m.get("original_filename", m["filename"]),
            excerpt=m.get("excerpt", "")
        )
        for m in user_metadata
    ]




@app.get("/download/{filename}")
def download_pdf(filename: str):
    path = os.path.join(PDF_DIR, filename)
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="Not found")
    return FileResponse(path, media_type='application/pdf', filename=filename)


@app.post("/search", response_model=List[SearchResultItem])
def search(req: SearchRequest, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]

    user_folder = os.path.join(USER_DIR, username)
    user_meta_path = os.path.join(user_folder, "meta.json")
    user_index_path = os.path.join(user_folder, "index.faiss")
    pdf_folder = os.path.join(user_folder, "pdfs")

    if not os.path.exists(user_meta_path) or not os.path.exists(user_index_path):
        return []

    with open(user_meta_path, "r", encoding="utf-8") as f:
        user_metadata = json.load(f)

    index = faiss.read_index(user_index_path)

    query = req.query.strip().lower()
    if not query:
        raise HTTPException(status_code=400, detail="Empty query")

    q_emb = model.encode([query], convert_to_numpy=True)[0].astype(EMBD_DTYPE)
    faiss.normalize_L2(q_emb.reshape(1, -1))

    top_k = min(req.top_k, len(user_metadata))
    D, I = index.search(q_emb.reshape(1, -1), top_k)

    results = []
    for score, idx in zip(D[0], I[0]):
        if idx < 0 or idx >= len(user_metadata):
            continue
        m = user_metadata[idx]

        pdf_path = os.path.join(pdf_folder, m["filename"])
        text = extract_text_from_pdf(pdf_path).lower()
        if query not in text:
            continue

        download_url = f"/download/{m['filename']}"
        results.append(SearchResultItem(
            id=m["id"],
            filename=m["filename"],
            original_filename=m["original_filename"],
            score=float(score),
            download_url=download_url,
            excerpt=m.get("excerpt")
        ))

    return results

    return results
# --- Helper: rebuild FAISS index after deletion ---
def rebuild_index():
    global index, id_to_idx
    index = faiss.IndexFlatIP(embed_dim)
    id_to_idx = {}

    if len(metadata) == 0:
        faiss.write_index(index, INDEX_PATH)
        return

    emb_list = []
    for i, m in enumerate(metadata):
        emb = np.array(m["embedding"], dtype=EMBD_DTYPE)
        emb_list.append(emb)

    emb_matrix = np.vstack(emb_list).astype(EMBD_DTYPE)
    faiss.normalize_L2(emb_matrix)
    index.add(emb_matrix)

    id_to_idx = {m["id"]: i for i, m in enumerate(metadata)}
    faiss.write_index(index, INDEX_PATH)
    save_meta()
    print(f"Rebuilt FAISS index with {len(metadata)} PDFs.")


# --- Delete PDF Endpoint ---
@app.delete("/delete/{pdf_id}")
def delete_pdf(pdf_id: str, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]

    # 🧭 Locate user directories
    user_folder = os.path.join(USER_DIR, username)
    pdf_folder = os.path.join(user_folder, "pdfs")
    user_meta_path = os.path.join(user_folder, "meta.json")
    user_index_path = os.path.join(user_folder, "index.faiss")

    # 🧱 Ensure user data exists
    if not os.path.exists(user_meta_path):
        raise HTTPException(status_code=404, detail="No uploaded files found for this user")

    # 🧾 Load metadata and find target PDF
    with open(user_meta_path, "r", encoding="utf-8") as f:
        user_metadata = json.load(f)

    pdf_meta = next((m for m in user_metadata if m["id"] == pdf_id), None)
    if not pdf_meta:
        raise HTTPException(status_code=404, detail="PDF not found")

    # 🧹 Remove file from disk
    pdf_path = os.path.join(pdf_folder, pdf_meta["filename"])
    if os.path.exists(pdf_path):
        os.remove(pdf_path)

    # 🧩 Remove entry from metadata
    user_metadata = [m for m in user_metadata if m["id"] != pdf_id]

    # 🧮 Rebuild FAISS index for this user
    index = faiss.IndexFlatIP(embed_dim)
    if len(user_metadata) > 0:
        emb_list = [np.array(m["embedding"], dtype=EMBD_DTYPE) for m in user_metadata]
        emb_matrix = np.vstack(emb_list).astype(EMBD_DTYPE)
        faiss.normalize_L2(emb_matrix)
        index.add(emb_matrix)

    # 💾 Save updated metadata and index
    with open(user_meta_path, "w", encoding="utf-8") as f:
        json.dump(user_metadata, f, indent=2)
    faiss.write_index(index, user_index_path)

    return {"message": f"PDF '{pdf_meta['original_filename']}' deleted successfully."}


@app.get("/summarize/{pdf_id}")
def summarize_pdf(pdf_id: str, current_user: dict = Depends(get_current_user)):
    username = current_user["username"]

    # 🧭 Locate user directories
    user_folder = os.path.join(USER_DIR, username)
    pdf_folder = os.path.join(user_folder, "pdfs")
    user_meta_path = os.path.join(user_folder, "meta.json")

    # 🧱 Ensure metadata exists
    if not os.path.exists(user_meta_path):
        raise HTTPException(status_code=404, detail="No uploaded files found for this user")

    # 🧾 Load metadata and find file
    with open(user_meta_path, "r", encoding="utf-8") as f:
        user_metadata = json.load(f)

    pdf_meta = next((m for m in user_metadata if m["id"] == pdf_id), None)
    if not pdf_meta:
        raise HTTPException(status_code=404, detail="PDF not found")

    # 🧩 Check file existence
    pdf_path = os.path.join(pdf_folder, pdf_meta["filename"])
    if not os.path.exists(pdf_path):
        raise HTTPException(status_code=404, detail="File not found on disk")

    # 🧠 Extract and summarize
    text = extract_text_from_pdf(pdf_path)
    if not text.strip():
        raise HTTPException(status_code=400, detail="No text could be extracted from PDF")

    summary = summarize_long_text(text)

    return {
        "pdf_id": pdf_id,
        "original_filename": pdf_meta["original_filename"],
        "summary": summary,
    }

# --- simple health endpoint ---
@app.get("/health")
def health():
    return {"status": "ok", "documents_indexed": len(metadata)}
