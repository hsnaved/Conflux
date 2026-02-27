# Conflux RAG Server

A **Retrieval-Augmented Generation (RAG)** service that ingests data from Confluence and PDFs, embeds text into vector form, stores it in Qdrant, and answers natural language questions using OpenAI's LLM. The system uses Model Context Protocol (MCP) integration for seamless tool availability.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Folder Structure](#folder-structure)
3. [Workflow & Architecture](#workflow--architecture)
4. [Installation & Setup](#installation--setup)
5. [Configuration](#configuration)
6. [Running the System](#running-the-system)
7. [Known Issues & Errors](#known-issues--errors)
8. [Testing & Debugging](#testing--debugging)

---

## Project Overview

**Conflux** implements a complete RAG pipeline:

- **Data Ingestion**: Fetch pages from Confluence or extract text from PDFs
- **Embeddings**: Convert text to vector embeddings using SentenceTransformers
- **Vector Storage**: Store embeddings in Qdrant for fast similarity search
- **Question Answering**: Retrieve relevant context and use OpenAI GPT to generate answers

### Key Components

| Component | Purpose |
|-----------|---------|
| **FastMCP Server** | Exposes RAG tools as Model Context Protocol tools |
| **Qdrant Vector DB** | Stores and searches document embeddings |
| **SentenceTransformers** | Generates semantic embeddings (all-MiniLM-L6-v2) |
| **OpenAI API** | Generates answers using retrieved context |
| **Confluence API** | Fetches pages from Confluence spaces |

---

## Folder Structure

```
Conflux/
├── config.py                 # Configuration management (Qdrant, OpenAI, Confluence URLs)
├── mcp_server.py             # Main MCP server with RAG tools (retrieve, ingest, ask)
├── requirements.txt          # Python dependencies
├── .env                       # Environment variables (API keys, credentials)
├── README.md                 # Original quick-start guide
├── README_COMPREHENSIVE.md   # This comprehensive guide
│
├── services/                 # Core service modules
│   ├── __init__.py
│   ├── embedding.py          # Text-to-vector embedding using SentenceTransformers
│   ├── vectorstore.py        # Qdrant vector DB operations (search, upsert)
│   ├── rag.py                # RAG pipeline (chunking, context building, LLM prompting)
│   └── confluence.py         # Confluence API client (fetch & parse pages)
│
├── uploads/                  # Directory for uploaded PDF files
│
├── debug_search.py           # Test script to debug vector search
├── test_retrieval.py         # Test retrieval pipeline
├── retrieve_results.py       # Retrieve and display search results
├── mcp_server_test.py        # Minimal MCP test server
└── tmp_check_qdrant.py       # Quick Qdrant client check
```

### Key File Descriptions

| File | Purpose |
|------|---------|
| `config.py` | Central configuration: loads env vars, sets defaults for Qdrant, embedding model, chunking params |
| `mcp_server.py` | FastMCP-based server exposing 3 tools: `retrieve_chunks`, `ingest_confluence_data`, `ask_question` |
| `services/embedding.py` | Lazy-loaded SentenceTransformer model; functions: `embed_text()`, `embed_texts()` |
| `services/vectorstore.py` | Qdrant client wrapper; uses `query_points()` API for vector similarity search |
| `services/rag.py` | High-level RAG logic: text chunking, context building, LLM prompting, PDF/Confluence ingestion |
| `services/confluence.py` | Confluence API integration: fetches pages, converts HTML to plain text |

---

## Workflow & Architecture

### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    CONFLUX RAG SYSTEM FLOW                      │
└─────────────────────────────────────────────────────────────────┘

1. DATA INGESTION PHASE
   ├─ Confluence Pages
   │  └→ fetch_confluence_pages() [confluence.py]
   │  └→ Confluence API (cloud=True)
   │  └→ Fetch pages from space
   │  └→ html_to_text() [confluence.py]
   │  └→ BeautifulSoup HTML parsing
   │
   └─ PDF Files
      └→ extract_text_from_pdf() [rag.py]
      └→ PyPDFLoader from LangChain
      └→ Extract text from each page

2. TEXT PREPROCESSING
   └→ chunk_text() [rag.py]
      └→ RecursiveCharacterTextSplitter
      └→ chunk_size: 800 chars
      └→ overlap: 150 chars
      └→ Output: List[str] of chunks

3. EMBEDDING GENERATION
   ├→ embed_texts() [embedding.py]
   │  └→ SentenceTransformer: all-MiniLM-L6-v2
   │  └→ Batch processing (batch_size=32)
   │  └→ Output: List[List[float]] (384-dim vectors)
   │
   └→ OR embed_text() for single strings

4. VECTOR STORAGE (Qdrant)
   └→ upsert_chunks() [vectorstore.py]
      └→ Create PointStruct objects
      └→ Add metadata: text, source, chunk_index, doc_id
      └→ Upsert into Qdrant collection
      └→ Collection: "rag_pdf_collection"
      └→ Distance: COSINE

5. QUESTION ANSWERING PHASE
   ├→ User Question Input
   │
   ├→ embed_text() [embedding.py]
   │  └→ Same model (cached): all-MiniLM-L6-v2
   │  └→ Convert question to 384-dim vector
   │
   ├→ search_similar() [vectorstore.py]
   │  └→ Qdrant client.query_points()
   │  └→ COSINE similarity search
   │  └→ Return top-k results (default k=5)
   │  └→ Include metadata in results
   │
   ├→ build_context() [rag.py]
   │  └→ Extract text payloads from results
   │  └→ Concatenate with \n\n separator
   │
   ├→ build_prompt() [rag.py]
   │  └→ Create system + context + question prompt
   │  └→ "Answer ONLY using provided context"
   │
   └→ answer_question() [rag.py]
      └→ OpenAI API (GPT-4o-mini)
      └→ Generate answer from context
      └→ Return answer text + result count

6. MCP TOOLS (Model Context Protocol)
   ├→ retrieve_chunks(question, limit)
   │  │  Directly return matching chunks
   │  └→ Response: {'status', 'chunks', 'count'}
   │
   ├→ ingest_confluence_data()
   │  │  Full pipeline: fetch → chunk → embed → store
   │  └→ Response: {'status', 'message', 'total_chunks', 'pages_count'}
   │
   └→ ask_question(question, limit)
      │  Full RAG: retrieve → context → answer
      └→ Response: {'status', 'question', 'context', 'chunks_retrieved', 'answer_hint'}
```

### Architecture Layers

**Layer 1: Embedding Service** (`services/embedding.py`)
- Model: SentenceTransformers "all-MiniLM-L6-v2"
- Dimensions: 384
- Normalization: L2 (cosine-compatible)
- Caching: LRU cache for model (maxsize=1)
- Device: GPU (cuda) if available, else CPU

**Layer 2: Vector Store** (`services/vectorstore.py`)
- Database: Qdrant (v1.x)
- Client: `qdrant_client==1.17.0+` (requires query_points API)
- Connection: HTTP (default: http://localhost:6333)
- Collection: `rag_pdf_collection`
- Distance Metric: COSINE
- Index Type: Default (HNSW)

**Layer 3: RAG Pipeline** (`services/rag.py`)
- Text Splitter: RecursiveCharacterTextSplitter
  - chunk_size: 800 tokens
  - chunk_overlap: 150 tokens
  - Strategies: ["\n\n", "\n", " ", ""]
- LLM: OpenAI GPT-4o-mini
- Temperature: 0 (deterministic)

**Layer 4: Data Connectors**
- **Confluence**: REST API (cloud-specific)
  - Authentication: username + API token
  - Html parsing: BeautifulSoup
- **PDF**: File-based via PyPDFLoader
  - Supports multi-page PDFs
  - Extracts text per page

**Layer 5: MCP Server** (`mcp_server.py`)
- Framework: FastMCP (Anthropic's lightweight MCP)
- Transport: stdio (suitable for IDE integration)
- Tools: 3 exposed functions
- Error handling: try-catch + stderr logging

---

## Installation & Setup

### Prerequisites

- **Python**: 3.11+
- **Docker**: Required for Qdrant (unless using cloud instance)
- **API Keys**: 
  - OpenAI API key (for LLM)
  - (Optional) Confluence API token
- **Network**: Connectivity to Qdrant (local or remote)

### Step 1: Clone or Navigate to Project

```powershell
cd c:\Projects\Conflux
```

### Step 2: Create Virtual Environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation fails, use:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
.\.venv\Scripts\Activate.ps1
```

### Step 3: Install Python Dependencies

```powershell
pip install --upgrade pip
pip install -r requirements.txt
```

**Key Dependencies Installed:**
- `fastmcp==2.13.3` – MCP server framework
- `qdrant-client==1.17.0` – Vector DB client (MUST be ≥1.17.0)
- `sentence-transformers==2.6.1` – Embedding models
- `langchain==0.1.20` + `langchain-community` – LLM/loader framework
- `langchain-openai==0.1.7` – OpenAI integration
- `atlassian-python-api==3.41.10` – Confluence client
- `beautifulsoup4==4.12.3` – HTML parsing
- Plus: transformers, huggingface_hub, torch, etc.

Verify installation:
```powershell
.venv\Scripts\python -c "from sentence_transformers import SentenceTransformer; print('✓ All imports OK')"
```

### Step 4: Start Qdrant Vector Database

**Option A: Docker (Recommended)**
```powershell
docker run -d --name qdrant-instance `
  -p 6333:6333 `
  -p 6334:6334 `
  qdrant/qdrant:latest

# Verify it's running
docker ps | Select-String qdrant
```

**Option B: Check if Already Running**
```powershell
docker ps | Select-String qdrant
curl http://localhost:6333/health
```

**Option C: Use Remote Qdrant**
- Set `QDRANT_URL` in `.env` to your remote instance
- Example: `QDRANT_URL=http://qdrant.example.com:6333`

### Step 5: Configure Environment Variables

Create or update `.env` file with:

```env
# REQUIRED: OpenAI API Key
OPENAI_API_KEY=sk-proj-YOUR_KEY_HERE

# OPTIONAL: Qdrant Configuration
QDRANT_URL=http://localhost:6333

# OPTIONAL: Confluence Configuration (if ingesting from Confluence)
CONFLUENCE_BASE_URL=https://yourcompany.atlassian.net/wiki
CONFLUENCE_USERNAME=your-email@example.com
CONFLUENCE_API_TOKEN=ATATT3xFfGF0wPdJUkRl3vJp...
CONFLUENCE_SPACE_KEY=YourSpaceKey
```

**How to get Confluence credentials:**
1. Go to https://id.atlassian.com/manage-profile/security/api-tokens
2. Create API token
3. Copy token and your email

### Step 6: Verify Setup

```powershell
.venv\Scripts\python debug_search.py
```

Expected output:
```
[TEST 1] Importing vectorstore module...
  ✓ Import successful
[TEST 2] Getting Qdrant client...
  ✓ Client obtained: <class 'qdrant_client.http.client.QdrantClient'>
[TEST 3] Ensuring collection exists...
  ✓ Collection ensured
[TEST 4] Creating embedding...
  ✓ Vector created, length: 384
[TEST 5] Calling search_similar...
  ✓ Search succeeded, got 0 results
  (0 results expected if no data ingested yet)

✓ All tests passed!
```

---

## Configuration

### `config.py` Reference

Central configuration file that loads environment variables and provides defaults:

```python
# Vector Store Configuration
COLLECTION_NAME = "rag_pdf_collection"      # Qdrant collection name
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
VECTOR_SIZE = 384                           # all-MiniLM-L6-v2 dimension

# Embedding Model
MODEL_NAME = "all-MiniLM-L6-v2"             # HuggingFace model ID

# Text Chunking Parameters
CHUNK_SIZE = 800                            # Characters per chunk
CHUNK_OVERLAP = 150                         # Overlap between chunks

# Search & LLM Parameters
SEARCH_LIMIT = 5                            # Default top-k results
LLM_MODEL_NAME = "gpt-4o-mini"              # OpenAI model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# API Server
APP_HOST = "0.0.0.0"
APP_PORT = 8000

# Confluence
CONFLUENCE_BASE_URL = os.getenv("CONFLUENCE_BASE_URL")
CONFLUENCE_USERNAME = os.getenv("CONFLUENCE_USERNAME")
CONFLUENCE_API_TOKEN = os.getenv("CONFLUENCE_API_TOKEN")
CONFLUENCE_SPACE_KEY = os.getenv("CONFLUENCE_SPACE_KEY")
```

### Customization Examples

**Change Embedding Model:**
```python
# In config.py
MODEL_NAME = "all-mpnet-base-v2"  # 768 dimensions
VECTOR_SIZE = 768
```

**Adjust Chunk Size:**
```python
CHUNK_SIZE = 1500      # Larger chunks
CHUNK_OVERLAP = 300    # More overlap
```

**Use Different LLM:**
```python
LLM_MODEL_NAME = "gpt-4"  # Faster/cheaper: gpt-4o-mini
```

**Change Qdrant Collection:**
```python
COLLECTION_NAME = "my_custom_collection"
```

---

## Running the System

### Option 1: Start MCP Server (Primary Usage)

```powershell
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python -m mcp_server
```

**What this does:**
- Initializes FastMCP framework
- Imports all services (embedding, vectorstore, rag, confluence)
- Registers 3 MCP tools
- Starts stdio-based server for external clients
- Logs all activity to stderr

**Expected startup output:**
```
[MCP] Starting Conflux RAG Server
[MCP] Imported embed_text
[MCP] Imported vectorstore functions
[MCP] Imported confluence
[MCP] Imported rag
[MCP] FastMCP initialized
[MCP] All tools registered
[MCP] Running MCP server
```

**Exposed Tools:**
1. **`retrieve_chunks(question: str, limit: int = 5) → dict`**
   - Returns matching text chunks from vector store
   - Response: `{"status": "success/error", "chunks": [...], "count": 0}`

2. **`ingest_confluence_data() → dict`**
   - Fetches all pages from configured Confluence space
   - Chunks, embeds, and stores them
   - Response: `{"status", "message", "total_chunks", "pages_count"}`

3. **`ask_question(question: str, limit: int = 5) → dict`**
   - Full RAG pipeline: retrieve → context → answer hint
   - Response: `{"status", "question", "context", "chunks_retrieved", "answer_hint"}`

### Option 2: Test Full Retrieval Pipeline

```powershell
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python test_retrieval.py
```

**Outputs:**
- Ingests sample question
- Retrieves matching chunks
- Displays results with scores

**Required:** Data must be ingested first (run ingest_confluence or upload PDF)

### Option 3: Retrieve and Display Results

```powershell
.venv\Scripts\python retrieve_results.py
```

Searches for "What are common deployment approaches of AI?" and displays top 5 results with scores.

### Option 4: Debug Search Pipeline Step-by-Step

```powershell
.venv\Scripts\python debug_search.py
```

Runs 5 diagnostic tests:
1. Import vectorstore
2. Get Qdrant client
3. Ensure collection exists
4. Create embedding
5. Perform search

Great for troubleshooting connection issues.

### Option 5: Test MCP Server in Isolation

```powershell
.venv\Scripts\python mcp_server_test.py
```

Minimal standalone MCP server for testing without full Conflux initialization.

---

## Known Issues & Errors

### **ERROR 1: Docker Qdrant Container Fails to Start**

**Symptom:**
```powershell
docker run -p 6333:6333 qdrant/qdrant:latest
# [Exit Code: 1]
```

**Root Cause:**
- Port 6333 already in use
- Docker daemon not running
- Insufficient resources
- Image not found/pulled

**Solution:**

1. **Check if Qdrant already running:**
   ```powershell
   docker ps | Select-String qdrant
   docker ps -a | Select-String qdrant  # includes stopped containers
   ```

2. **If running, use it; if stopped, start it:**
   ```powershell
   docker start qdrant-instance
   ```

3. **If not found, create with custom name:**
   ```powershell
   docker run -d --name qdrant-instance `
     -p 6333:6333 -p 6334:6334 `
     qdrant/qdrant:latest
   ```

4. **Check port availability:**
   ```powershell
   netstat -ano | Select-String 6333
   ```

5. **If port in use, kill process:**
   ```powershell
   # Find PID from netstat, then:
   taskkill /PID <PID> /F
   ```

6. **Or run on different port:**
   ```powershell
   docker run -p 6334:6333 qdrant/qdrant:latest
   # Then set QDRANT_URL=http://localhost:6334 in .env
   ```

7. **Check Docker status:**
   ```powershell
   docker info  # Verifies daemon is running
   docker logs qdrant-instance | Select-Object -First 20
   ```

---

### **ERROR 2: `qdrant-client` Version Mismatch**

**Symptom:**
```
RuntimeError: Vector query failed - ensure qdrant-client 1.17.0+ is installed.
AttributeError: 'QdrantClient' has no attribute 'query_points'
```

**Root Cause:**
- Installed qdrant-client version < 1.17.0
- The `query_points()` API was added in 1.17.0
- Old version has `search()` or other methods instead

**Solution:**

1. **Check installed version:**
   ```powershell
   pip show qdrant-client | Select-String Version
   ```

2. **Force upgrade to 1.17.0:**
   ```powershell
   pip uninstall qdrant-client -y
   pip install qdrant-client==1.17.0
   ```

3. **Verify:**
   ```powershell
   .venv\Scripts\python -c "
   from qdrant_client import QdrantClient
   c = QdrantClient('http://localhost:6333')
   print('Has query_points:', hasattr(c, 'query_points'))
   print('Client methods:', [m for m in dir(c) if 'query' in m.lower() or 'search' in m.lower()])
   "
   ```

---

### **ERROR 3: Qdrant Connection Refused**

**Symptom:**
```
ConnectionError: Failed to connect to http://localhost:6333
```

**Root Cause:**
- Qdrant container not running
- Wrong QDRANT_URL
- Firewall blocking port
- Port forwarding issues in Docker

**Solution:**

1. **Check if Qdrant is responsive:**
   ```powershell
   curl http://localhost:6333/health
   # Should return: {"status":"ok"}
   ```

2. **If curl fails, start Qdrant:**
   ```powershell
   docker run -d -p 6333:6333 qdrant/qdrant:latest
   ```

3. **Verify QDRANT_URL in .env:**
   ```powershell
   type .env | Select-String QDRANT_URL
   ```

4. **Check Qdrant logs:**
   ```powershell
   docker logs qdrant | Select-Object -Last 20
   ```

5. **If using remote Qdrant:**
   ```env
   QDRANT_URL=http://qdrant.example.com:6333
   ```

6. **Test client directly:**
   ```powershell
   .venv\Scripts\python -c "
   from services.vectorstore import get_client
   c = get_client()
   print('Connected OK')
   print('Collections:', c.get_collections())
   "
   ```

---

### **ERROR 4: Confluence Authentication Fails**

**Symptom:**
```
atlassian.rest_client.exceptions.HttpError: 401 Unauthorized
ConnectionError: Failed to authenticate with Confluence
```

**Root Cause:**
- Invalid API token
- Wrong username/email
- Confluence URL incorrect
- Token expired
- Cloud vs Server configuration mismatch

**Solution:**

1. **Verify credentials in .env:**
   ```powershell
   Get-Content .env | Select-String CONFLUENCE
   ```

2. **Test API token manually:**
   ```powershell
   .venv\Scripts\python -c "
   import os
   from atlassian import Confluence
   
   c = Confluence(
       url=os.getenv('CONFLUENCE_BASE_URL'),
       username=os.getenv('CONFLUENCE_USERNAME'),
       password=os.getenv('CONFLUENCE_API_TOKEN'),
       cloud=True
   )
   print('Connected OK')
   spaces = c.get_all_spaces()
   print(f'Found {len(spaces)} spaces')
   "
   ```

3. **If fails, regenerate token:**
   - Go to: https://id.atlassian.com/manage-profile/security/api-tokens
   - Delete old token
   - Create new token
   - Update .env

4. **Ensure using email, not username:**
   ```env
   CONFLUENCE_USERNAME=your-email@company.com  # ✓ Correct
   # NOT: your-username                         # ✗ Wrong
   ```

5. **Verify cloud flag:**
   ```python
   # In confluence.py, ensure:
   cloud=True  # For Confluence Cloud
   cloud=False  # For self-hosted Server
   ```

6. **Check space exists:**
   ```powershell
   .venv\Scripts\python -c "
   from services.confluence import fetch_confluence_pages
   import os
   
   try:
       pages = fetch_confluence_pages(
           os.getenv('CONFLUENCE_BASE_URL'),
           os.getenv('CONFLUENCE_USERNAME'),
           os.getenv('CONFLUENCE_API_TOKEN'),
           'TESTSPACE'  # Try a known space
       )
       print(f'Fetched {len(pages)} pages')
   except Exception as e:
       print(f'Error: {e}')
   "
   ```

---

### **ERROR 5: OPENAI_API_KEY Not Found**

**Symptom:**
```
RuntimeError: OPENAI_API_KEY environment variable required
KeyError: 'OPENAI_API_KEY'
```

**Root Cause:**
- .env file not loaded
- Variable not set in PowerShell session
- config.py not reading it correctly
- Empty .env file

**Solution:**

1. **Set in current PowerShell session:**
   ```powershell
   $env:OPENAI_API_KEY = "sk-proj-YOUR_KEY"
   ```

2. **Or update .env file:**
   ```env
   # .env
   OPENAI_API_KEY=sk-proj-YOUR_KEY
   ```

3. **Verify it's loaded:**
   ```powershell
   .venv\Scripts\python -c "
   from config import OPENAI_API_KEY
   print('Key found:', bool(OPENAI_API_KEY))
   print('Key starts with:', OPENAI_API_KEY[:7] if OPENAI_API_KEY else 'NOT SET')
   "
   ```

4. **Check .env is being read:**
   ```powershell
   .venv\Scripts\python -c "
   from dotenv import load_dotenv
   import os
   
   load_dotenv()
   print('After load_dotenv:', os.getenv('OPENAI_API_KEY', 'NOT FOUND'))
   "
   ```

5. **If using in production:**
   - Don't use .env file
   - Set via environment: `$env:OPENAI_API_KEY = "..."` before running
   - Or use key management system

---

### **ERROR 6: No Search Results (Empty Vector Store)**

**Symptom:**
```
Search returns empty list []
[TEST] Got 0 results
```

**Root Cause:**
- Collection exists but is empty
- Vector size mismatch
- Data not ingested
- Search query too specific
- Collection name mismatch

**Solution:**

1. **Check collection status:**
   ```powershell
   .venv\Scripts\python -c "
   from services.vectorstore import get_client, COLLECTION_NAME
   c = get_client()
   coll = c.get_collection(COLLECTION_NAME)
   print(f'Collection: {COLLECTION_NAME}')
   print(f'Points: {coll.points_count}')
   print(f'Vector size: {coll.config.params.vectors.size}')
   "
   ```

2. **If 0 points, ingest data:**
   
   **Option A: From Confluence**
   ```powershell
   .venv\Scripts\python -c "
   from services.rag import ingest_confluence
   print('Ingesting Confluence pages...')
   count = ingest_confluence()
   print(f'Ingested {count} chunks')
   "
   ```

   **Option B: From PDF**
   ```powershell
   # Place PDF in uploads/ folder, then:
   .venv\Scripts\python -c "
   from pathlib import Path
   from services.rag import ingest_pdf
   
   pdf_path = Path('uploads/my_document.pdf')
   count = ingest_pdf(pdf_path, 'my_document')
   print(f'Ingested {count} chunks from PDF')
   "
   ```

3. **Verify ingestion:**
   ```powershell
   .venv\Scripts\python -c "
   from services.vectorstore import get_client, COLLECTION_NAME
   c = get_client()
   coll = c.get_collection(COLLECTION_NAME)
   print(f'Points after ingest: {coll.points_count}')
   "
   ```

4. **Test search again:**
   ```powershell
   .venv\Scripts\python test_retrieval.py
   ```

5. **If still empty, check for errors during ingestion:**
   ```powershell
   # Look for error messages in the ingest output
   # Check .env for Confluence credentials
   ```

---

### **ERROR 7: SentenceTransformer Model Download Fails**

**Symptom:**
```
OSError: Can't load model. Model not found.
huggingface_hub.utils._errors.RepositoryNotFound: 404 Client Error
```

**Root Cause:**
- No internet connectivity
- HuggingFace server down
- Firewall blocking downloads
- Network timeout
- Model name typo

**Solution:**

1. **Verify model name in config.py:**
   ```python
   MODEL_NAME = "all-MiniLM-L6-v2"  # ✓ Correct
   ```

2. **Download model manually:**
   ```powershell
   .venv\Scripts\python -c "
   from sentence_transformers import SentenceTransformer
   model = SentenceTransformer('all-MiniLM-L6-v2')
   print('Model downloaded successfully')
   "
   ```

3. **Check internet connectivity:**
   ```powershell
   curl https://huggingface.co -UseBasicParsing
   ```

4. **If offline, use a smaller/cached model:**
   ```python
   # config.py
   MODEL_NAME = "all-MiniLM-L6-v2"  # Pre-downloaded
   # Alternative: "all-mpnet-base-v2"
   ```

5. **Clear cache and retry:**
   ```powershell
   rm -Recurse $env:USERPROFILE\.cache\huggingface\* -ErrorAction SilentlyContinue
   rm -Recurse .venv\Lib\site-packages\sentence_transformers\** -ErrorAction SilentlyContinue
   pip install sentence-transformers --force-reinstall
   ```

---

## Testing & Debugging

### Recommended Test Sequence

```powershell
# 1. Activate environment
.\.venv\Scripts\Activate.ps1

# 2. Ensure Qdrant is running
docker ps | Select-String qdrant

# 3. Run diagnostic test
.venv\Scripts\python debug_search.py

# 4. If test passes, try retrieval
.venv\Scripts\python test_retrieval.py  # Requires data

# 5. If retrieval works, test MCP server
.venv\Scripts\python -m mcp_server
```

### Test Scripts Summary

| Script | Purpose | Output |
|--------|---------|--------|
| `debug_search.py` | 5-step diagnostic of search pipeline | Step-by-step results |
| `test_retrieval.py` | Full retrieval test with sample question | Retrieved chunks |
| `retrieve_results.py` | Search & display pretty results | Formatted results |
| `mcp_server_test.py` | Minimal MCP server for isolation | MCP tool output |
| `tmp_check_qdrant.py` | Quick client method check | Available methods |

### Clear Python Cache (If Experiencing Weird Errors)

```powershell
# Remove __pycache__ directories
rm -Recurse .pycache, services\__pycache__ -ErrorAction SilentlyContinue -Force

# Reinstall dependencies cleanly
pip uninstall -r requirements.txt -y
pip install -r requirements.txt

echo "Cache cleared and deps reinstalled"
```

### Manual Testing of Each Component

**Test 1: Embedding Service**
```powershell
.venv\Scripts\python -c "
from services.embedding import embed_text
vec = embed_text('Hello world')
print(f'Vector length: {len(vec)}')
print(f'Sample values: {vec[:3]}')
"
```

**Test 2: Vector Store**
```powershell
.venv\Scripts\python -c "
from services.vectorstore import get_client, ensure_collection
client = get_client()
ensure_collection()
print('Vector store OK')
"
```

**Test 3: RAG Pipeline**
```powershell
.venv\Scripts\python -c "
from services.rag import chunk_text
text = 'This is a sample text that will be split into chunks.'
chunks = chunk_text(text)
print(f'Created {len(chunks)} chunks')
"
```

**Test 4: Confluence Connection**
```powershell
.venv\Scripts\python -c "
from services.confluence import fetch_confluence_pages
import os
pages = fetch_confluence_pages(
    os.getenv('CONFLUENCE_BASE_URL'),
    os.getenv('CONFLUENCE_USERNAME'),
    os.getenv('CONFLUENCE_API_TOKEN'),
    os.getenv('CONFLUENCE_SPACE_KEY'),
    limit=1
)
print(f'Fetched {len(pages)} page(s)')
if pages:
    print(f'First page: {pages[0][\"title\"]}')
"
```

---

## Summary

**Conflux** is a production-quality RAG system with:

✅ **Modular Architecture** – Clear separation of concerns  
✅ **Multiple Data Sources** – Confluence, PDF support  
✅ **Fast Search** – Qdrant vector similarity  
✅ **Semantic Embeddings** – SentenceTransformers  
✅ **LLM Integration** – OpenAI API  
✅ **MCP Support** – Integration with Claude/Copilot  
✅ **Comprehensive Testing** – Multiple test scripts  
✅ **Error Handling** – Detailed error messages  
✅ **Easy Configuration** – Environment-based settings  

### Next Steps

1. **Setup**: Follow Installation & Setup section
2. **Test**: Run `debug_search.py` to verify
3. **Ingest Data**: Use `ingest_confluence()` or PDF upload
4. **Run Server**: `python -m mcp_server`
5. **Debug Issues**: Reference Known Issues section

For detailed troubleshooting, check the error corresponding to your issue in the [Known Issues & Errors](#known-issues--errors) section.
