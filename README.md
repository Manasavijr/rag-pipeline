# Local RAG & LLM Intelligence Pipeline

A privacy-preserving RAG (Retrieval-Augmented Generation) system that runs entirely locally — no API keys, no data leaving your machine. Built with LangChain, Ollama, FAISS, and FastAPI.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              FastAPI REST API                        │
│  /ingest  /qa/ask  /summarize  /classify  /docs     │
└──────────────────┬──────────────────────────────────┘
                   │
        ┌──────────▼──────────────┐
        │     LangChain Pipeline   │
        │  ┌─────────────────┐    │
        │  │ Document Loader  │    │
        │  │ PDF/TXT/DOCX/MD  │    │
        │  └────────┬────────┘    │
        │  ┌────────▼────────┐    │
        │  │  Text Splitter   │    │
        │  │ chunk+overlap    │    │
        │  └────────┬────────┘    │
        │  ┌────────▼────────┐    │
        │  │  FAISS Index     │    │
        │  │  HF Embeddings   │    │
        │  └────────┬────────┘    │
        └──────────┼──────────────┘
                   │
        ┌──────────▼──────────────┐
        │   Ollama (Local LLM)    │
        │   llama3.2 / mistral    │
        │   100% private, no key  │
        └─────────────────────────┘
```

## Tech Stack

| Component | Technology |
|---|---|
| API Framework | FastAPI + Uvicorn |
| LLM Orchestration | LangChain |
| Local LLM | Ollama (llama3.2) |
| Vector Store | FAISS |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 |
| Document Parsing | PyPDF, python-docx, Unstructured |
| Containerization | Docker + docker-compose |
| Cloud Deployment | GCP Cloud Run |
| CI/CD | GitHub Actions |

---

## Folder Structure

```
rag-pipeline/
├── app/
│   ├── main.py                    # FastAPI entry point
│   ├── core/
│   │   ├── config.py              # Pydantic settings
│   │   └── logging_config.py      # Structured JSON logging
│   ├── api/routes/
│   │   ├── ingest.py              # POST /api/v1/ingest/upload
│   │   ├── qa.py                  # POST /api/v1/qa/ask
│   │   ├── summarize.py           # POST /api/v1/summarize
│   │   ├── classify.py            # POST /api/v1/classify
│   │   └── health.py              # GET /health
│   └── schemas/schemas.py         # All Pydantic models
│
├── pipeline/
│   ├── ingestion/
│   │   └── document_loader.py     # PDF/TXT/DOCX/MD loader + chunker
│   ├── retrieval/
│   │   └── vector_store.py        # FAISS index manager
│   └── generation/
│       └── llm_chain.py           # LangChain + Ollama chains
│
├── data/
│   ├── generate_sample_docs.py    # Sample document generator
│   ├── raw/                       # Uploaded documents
│   ├── processed/                 # Intermediate processing
│   └── indexes/                   # Persisted FAISS index
│
├── tests/unit/test_api.py         # 15+ unit tests
├── .github/workflows/ci-cd.yml    # CI/CD pipeline
├── Dockerfile                     # Multi-stage build
├── docker-compose.yml             # Full stack: API + Ollama
└── README.md
```

---

## Local Setup (No Docker)

### Step 1 — Install Ollama

```bash
# Mac
brew install ollama

# Or download from https://ollama.ai
```

### Step 2 — Pull the LLM model

```bash
ollama pull llama3.2
```

This downloads ~2GB. While it downloads, continue setup.

### Step 3 — Install Python dependencies

```bash
cd rag-pipeline
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 4 — Generate sample documents

```bash
python data/generate_sample_docs.py
```

Creates 4 sample documents in `data/raw/`:
- `legal_contract.txt` — services agreement
- `financial_report.txt` — Q3 2025 fund report
- `technical_architecture.txt` — system design doc
- `ai_research_summary.txt` — ML research paper

### Step 5 — Start Ollama

```bash
# In a new terminal tab
ollama serve
```

### Step 6 — Start the API

```bash
source .venv/bin/activate
uvicorn app.main:app --reload --port 8080
```

Open **http://localhost:8080/docs** — Swagger UI

---

## API Usage

### 1. Upload a document

```bash
curl -X POST http://localhost:8080/api/v1/ingest/upload \
  -F "file=@data/raw/legal_contract.txt"
```

Response:
```json
{
  "status": "success",
  "filename": "legal_contract.txt",
  "chunks_created": 12,
  "pages": 1,
  "index_size": 12,
  "processing_time_ms": 234.5
}
```

### 2. Ask a question

```bash
curl -X POST http://localhost:8080/api/v1/qa/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are the payment terms in the contract?", "top_k": 3}'
```

Response:
```json
{
  "question": "What are the payment terms in the contract?",
  "answer": "The contract specifies a monthly retainer of $15,000, due within 30 days of invoice. Late payments incur a 1.5% monthly interest charge.",
  "sources": [
    {"content": "Client agrees to pay...", "source": "legal_contract.txt", "score": 0.94}
  ],
  "latency_ms": 1823.4
}
```

### 3. Summarize documents

```bash
# Concise summary
curl -X POST http://localhost:8080/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"style": "concise"}'

# Bullet-point summary focused on risks
curl -X POST http://localhost:8080/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"style": "bullet", "query": "key risks and mitigation strategies"}'

# Detailed summary of specific file
curl -X POST http://localhost:8080/api/v1/summarize \
  -H "Content-Type: application/json" \
  -d '{"style": "detailed", "filename": "financial_report.txt"}'
```

### 4. Classify a document

```bash
curl -X POST http://localhost:8080/api/v1/classify \
  -H "Content-Type: application/json" \
  -d '{
    "text": "This agreement governs the terms of the service engagement between parties.",
    "categories": ["legal", "financial", "technical", "medical", "general"]
  }'
```

Response:
```json
{
  "category": "legal",
  "confidence": "high",
  "reasoning": "Contains contract terminology, party references, and legal obligations.",
  "all_scores": {"legal": "high", "financial": "low", "technical": "low"}
}
```

### 5. Index stats

```bash
curl http://localhost:8080/api/v1/ingest/stats | python3 -m json.tool
```

---

## Run Tests

```bash
pytest tests/ -v
```

Expected:
```
test_liveness            PASSED
test_health              PASSED
test_root                PASSED
test_ask_question        PASSED
test_ask_no_index        PASSED
test_ask_question_too_short  PASSED
test_ask_empty_question  PASSED
test_summarize           PASSED
test_summarize_bullet_style  PASSED
test_summarize_invalid_style PASSED
test_classify_text       PASSED
test_classify_no_input   PASSED
test_classify_too_few_categories  PASSED
test_ingest_stats        PASSED

14 passed
```

---

## Docker (Full Stack with Ollama)

```bash
# Start everything (Ollama + model pull + API)
docker compose up

# Takes ~5 min on first run (downloads llama3.2)
# API: http://localhost:8080/docs
# Ollama: http://localhost:11434
```

---

## GCP Cloud Run Deployment

> Note: Cloud Run deployment works for the API. Ollama runs locally or on a separate VM.

```bash
export PROJECT_ID=your-project-id

# Build and push
gcloud auth configure-docker gcr.io
docker build --platform linux/amd64 \
  -t gcr.io/$PROJECT_ID/rag-pipeline:latest .
docker push gcr.io/$PROJECT_ID/rag-pipeline:latest

# Deploy
gcloud run deploy rag-pipeline \
  --image gcr.io/$PROJECT_ID/rag-pipeline:latest \
  --platform managed \
  --region us-central1 \
  --memory 4Gi --cpu 2 \
  --allow-unauthenticated \
  --set-env-vars OLLAMA_BASE_URL=YOUR_OLLAMA_VM_URL
```

---

## GitHub Actions Secrets

| Secret | Description |
|---|---|
| `GCP_SA_KEY` | GCP Service Account JSON |
| `GCP_PROJECT_ID` | GCP project ID |
| `OLLAMA_BASE_URL` | Ollama server URL (VM or tunnel) |


