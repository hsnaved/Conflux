# Conflux

A simple Retrieval-Augmented Generation (RAG) service that ingests data from
PDFs or Confluence, embeds text, stores it in Qdrant, and answers natural
language questions using OpenAI.

## Requirements

- Python 3.11+ (virtual environment recommended)
- Docker (for Qdrant) or a running Qdrant instance
- An OpenAI API key
- (Optional) Confluence API token and space key if using Confluence data

## Setup

1. **Install dependencies**

   ```powershell
   python -m venv .venv
   .\.venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment variables**

   The project relies on the `OPENAI_API_KEY` environment variable. Set it in
   PowerShell as shown:

   ```powershell
   $env:OPENAI_API_KEY = "sk-..."
   ```

   Optionally, you can export other settings via `config.py` directly if
   preferred (e.g. Confluence credentials, Qdrant URL, collection name).

3. **Start Qdrant**

   ```powershell
   docker run -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
   ```

   (see earlier in this README for troubleshooting if the container fails)

4. **Run the API**

   ```powershell
   python app.py
   ```

## Usage

- `POST /ingest_confluence/` – pull pages from the configured Confluence space
  and ingest them into Qdrant.
- `POST /upload_store/` – upload a PDF file to the `uploads/` folder.
- `POST /query/` – ask a question over the ingested content.

Example query with `curl` in PowerShell:

```powershell
curl -X POST http://localhost:8000/query/ -H "Content-Type: application/json" -d "{\"question\":\"What is the project about?\",\"limit\":5}"
```

