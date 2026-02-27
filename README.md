# Conflux RAG Server

A **Retrieval-Augmented Generation (RAG)** service that ingests data from Confluence and PDFs, embeds text into vectors, stores embeddings in Qdrant, and answers questions using OpenAI. Exposes tools via Model Context Protocol (MCP) for IDE/AI assistant integration.

**📖 For comprehensive documentation, see [README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)**

---

## Quick Start

### 1. Setup Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Start Qdrant
```powershell
docker run -d -p 6333:6333 -p 6334:6334 qdrant/qdrant:latest
```

### 3. Configure
Update `.env` with your API keys:
```env
OPENAI_API_KEY=sk-proj-...
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=ATATT3x...
CONFLUENCE_SPACE_KEY=YourSpace
QDRANT_URL=http://localhost:6333
```

### 4. Run Server
```powershell
.venv\Scripts\python -m mcp_server
```

---

## Project Structure

```
Conflux/
├── config.py                    # Configuration & environment vars
├── mcp_server.py                # MCP tools: retrieve, ingest, ask
├── services/
│   ├── embedding.py             # SentenceTransformers embeddings
│   ├── vectorstore.py           # Qdrant client wrapper
│   ├── rag.py                   # RAG pipeline (chunking, LLM)
│   └── confluence.py            # Confluence API client
├── debug_search.py              # Test vector search
├── test_retrieval.py            # Test full pipeline
├── retrieve_results.py          # Display search results
├── uploads/                     # PDF storage directory
└── requirements.txt             # Dependencies
```

---

## Key Features

| Feature | Details |
|---------|---------|
| **Data Ingestion** | Confluence pages, PDF files |
| **Embeddings** | SentenceTransformers (all-MiniLM-L6-v2, 384-dim) |
| **Vector Store** | Qdrant (COSINE similarity) |
| **LLM** | OpenAI GPT-4o-mini |
| **Interface** | MCP tools (stdio-based) |
| **Text Chunking** | RecursiveCharacterTextSplitter (800 chars, 150 overlap) |

---

## Workflow

```
Data (Confluence/PDF)
    ↓
Chunking (RecursiveCharacterTextSplitter)
    ↓
Embedding (SentenceTransformers)
    ↓
Vector Storage (Qdrant)
    ↓
Question → Embed → Search → Retrieve Context
    ↓
Build Context + Prompt
    ↓
LLM (OpenAI) → Answer
```

---

## MCP Tools

Three tools exposed via Model Context Protocol:

### 1. `retrieve_chunks(question, limit=5)`
Return matching text chunks from vector store.

### 2. `ingest_confluence_data()`
Fetch and ingest all pages from configured Confluence space.

### 3. `ask_question(question, limit=5)`
Full RAG: retrieve context and provide answer hint.

---

## Testing

### Run Diagnostics
```powershell
.venv\Scripts\python debug_search.py          # 5-step test
.venv\Scripts\python test_retrieval.py        # Full pipeline
.venv\Scripts\python retrieve_results.py      # Display results
```

---

## Common Errors & Fixes

### ❌ Docker Qdrant Fails
```powershell
docker ps                           # Check if running
docker run -d -p 6333:6333 qdrant/qdrant:latest
```

### ❌ `query_points` Not Found
```powershell
pip uninstall qdrant-client -y
pip install qdrant-client==1.17.0
```

### ❌ Qdrant Connection Refused
```powershell
curl http://localhost:6333/health   # Check health
docker logs qdrant                  # Check logs
```

### ❌ No Vector Search Results
1. Check collection has points: `config.py` → `COLLECTION_NAME`
2. Ingest data: `mcp_server` → `ingest_confluence_data`

### ❌ OPENAI_API_KEY Not Found
```powershell
$env:OPENAI_API_KEY = "sk-proj-..."
# Or update .env file
```

### ❌ Confluence 401 Unauthorized
- Use email address as username (not username)
- Regenerate API token at: https://id.atlassian.com/manage-profile/security/api-tokens
- Ensure `cloud=True` in confluence.py

---

## Detailed Documentation

For architecture, workflow diagrams, all error explanations, and troubleshooting:

👉 **[README_COMPREHENSIVE.md](README_COMPREHENSIVE.md)**

Covers:
- Complete folder structure with file purposes
- Data flow & architecture layers
- Step-by-step installation
- All configuration options
- 7 detailed error scenarios with solutions
- Component testing guide
- Model selection & customization

---

## Requirements

- Python 3.11+
- Docker (for Qdrant)
- OpenAI API key
- (Optional) Confluence API token

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
docker run -d -p 6333:6333 qdrant/qdrant:latest
# Update .env with API keys
.venv\Scripts\python -m mcp_server
```

---

## Architecture Layers

**Layer 1: Embedding** – SentenceTransformers (384-dim vectors)  
**Layer 2: Vector DB** – Qdrant with COSINE distance  
**Layer 3: RAG** – Text chunking + LLM prompting  
**Layer 4: Connectors** – Confluence & PDF support  
**Layer 5: MCP Server** – FastMCP with 3 tools  

---

## Key Files

| File | Purpose |
|------|---------|
| `mcp_server.py` | Main entry point; exposes MCP tools |
| `config.py` | Central config; loads from .env |
| `services/embedding.py` | Text-to-vector conversion |
| `services/vectorstore.py` | Qdrant client with search |
| `services/rag.py` | RAG pipeline implementation |
| `services/confluence.py` | Confluence API integration |

---

**Status:** ✅ Production-ready | **Last Updated:** Feb 2026

