# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException
# pyrefly: ignore [missing-import]
from fastapi.responses import RedirectResponse
# pyrefly: ignore [missing-import]    
from fastapi.middleware.cors import CORSMiddleware
# pyrefly: ignore [missing-import]    
from fastapi.staticfiles import StaticFiles
# pyrefly: ignore [missing-import]    
from pydantic import BaseModel, Field
from indexing import save_and_ingest, list_documents, ingest
from learning import summarize, generate_quiz, generate_flashcards
from rag import answer 
from schemas import (
    Summary, 
    QuizSet, 
    FlashcardSet, 
    RagAnswer, 
    DocumentInfo, 
    MetadataFilter, 
    UploadResponse, SummarizeRequest, AskRequest, QuizRequest, FlashcardsRequest
)
from filters import filters_to_dict
import os

app = FastAPI(
    title = "RAG Learning API",
    description = "Grounded Q&A, summaries, quizzes, and flashcards over indexed PDFs.",
)

# CORS middleware for frontend communication
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend static files
_frontend_dir = os.path.join(os.path.dirname(__file__), "frontend")
if os.path.isdir(_frontend_dir):
    app.mount("/app", StaticFiles(directory=_frontend_dir, html=True), name="frontend")

@app.get("/health")
def health():
    """Check if the API is running"""
    return {"status": "ok"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/documents", response_model=list[DocumentInfo])
def documents():
    try:
        return list_documents()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload", response_model=UploadResponse)
async def upload(file: UploadFile = File(...)):
    try:
        content = await file.read()
        return save_and_ingest(
            file_bytes=content, 
            filename=file.filename or "uploaded_document.pdf"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class IngestRequest(BaseModel):
    recreate: bool = False

@app.post("/ingest")
def api_ingest(req: IngestRequest = IngestRequest()):
    """Re-indexes all PDF documents currently in the data directory into Qdrant."""
    try:
        count = ingest(recreate=req.recreate)
        return {"status": "ok", "chunks_indexed": count}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/ask", response_model=RagAnswer)
def ask(req: AskRequest):
    """General Q&A over the documents."""
    try:
        return answer(
            question=req.question, 
            k=req.k, 
            filters=filters_to_dict(req.filters)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/summarize"
, response_model=Summary)
def api_summarize(req: SummarizeRequest):
    return summarize(
        document=req.document,
        query=req.query,
        filters=filters_to_dict(req.filters),
        k=req.k,
    )

@app.post("/quiz", response_model=QuizSet)
def api_generate_quiz(req: QuizRequest):
    """Generates multiple-choice questions."""
    try:
        return generate_quiz(
            document=req.document,
            query=req.query,
            filters=filters_to_dict(req.filters),
            count=req.count,
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/flashcards", response_model=FlashcardSet)
def api_generate_flashcards(req: FlashcardsRequest):
    """Generates study flashcards (front/back)."""
    try:
        return generate_flashcards(
            document=req.document,
            query=req.query,
            filters=filters_to_dict(req.filters),
            count=req.count,
            k=req.k
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def read_root():
    """Redirects the root URL automatically to the interactive API docs."""
    return RedirectResponse(url="/docs")