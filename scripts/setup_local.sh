#!/usr/bin/env bash
# setup_local.sh — Bootstrap local RAG pipeline environment
set -euo pipefail

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  RAG Pipeline — Local Setup"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Python environment
echo "▶ Creating virtual environment..."
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

# Environment file
cp .env .env.backup 2>/dev/null || true

# Create directories
mkdir -p data/raw data/processed data/indexes

# Generate sample documents
echo "▶ Generating sample documents..."
python data/generate_sample_docs.py

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  ✅ Setup complete!"
echo ""
echo "  Next steps:"
echo "  1. Install Ollama: https://ollama.ai"
echo "  2. ollama pull llama3.2"
echo "  3. ollama serve"
echo "  4. source .venv/bin/activate"
echo "  5. uvicorn app.main:app --reload --port 8080"
echo "  6. open http://localhost:8080/docs"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
