from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from config import APP_HOST, APP_PORT, SEARCH_LIMIT, UPLOAD_DIR
from services.rag import ingest_pdf
from services.rag import answer_question

app = FastAPI(title="Conflux RAG API", version="0.1.0")

# Allow CORS for development/frontends. Adjust origins for production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class UploadResponse(BaseModel):
    message: str
    chunks: int


class StoreResponse(BaseModel):
    message: str
    filename: str


class QueryRequest(BaseModel):
    question: str
    limit: Optional[int] = None


class QueryResponse(BaseModel):
    answer: str
    retrieved_chunks: int


@app.on_event("startup")
async def ensure_upload_dir() -> None: UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/upload_pdf/", response_model=UploadResponse)
async def upload_pdf(file: UploadFile = File(...)) -> UploadResponse:
    temp_path = UPLOAD_DIR / file.filename

    try:
        with temp_path.open("wb") as f:
            f.write(await file.read())

        chunks_added = ingest_pdf(temp_path, source_name=file.filename)
    except Exception as exc: # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to ingest PDF: {exc}") from exc
    finally:
      temp_path.unlink(missing_ok=True)

    return UploadResponse(message=f"Stored {chunks_added} chunks successfully", chunks=chunks_added)


@app.post(
    "/upload_store/",
    response_model=StoreResponse,
    summary="Upload a PDF and save to uploads folder",
)
async def upload_and_store(
    file: UploadFile = File(..., description="Select a PDF to store"),
) -> StoreResponse:
    temp_path = UPLOAD_DIR / file.filename

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=415, detail="Only PDF files are supported")

    try:
        with temp_path.open("wb") as f:
            f.write(await file.read())
    except Exception as exc: # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Failed to store PDF: {exc}") from exc

    return StoreResponse(message="File stored successfully", filename=file.filename)


# Query endpoint is disabled until `answer_question` is implemented in `services.rag`.
@app.post("/query/", response_model=QueryResponse)
async def query_pdf(payload: QueryRequest) -> QueryResponse:
    # import here to avoid import-time errors if `services.rag` has heavy or optional deps
    answer, retrieved = answer_question(payload.question, limit=payload.limit or SEARCH_LIMIT)
    return QueryResponse(answer=answer, retrieved_chunks=retrieved)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host=APP_HOST, port=APP_PORT, reload=True)