# Enterprise RAG Pipeline

[![Hugging Face Spaces](https://img.shields.io/badge/🤗%20Hugging%20Face-Spaces-blue.svg)](https://huggingface.co/spaces/Keziya13/enterprise-rag-pipeline)

> A production-grade Retrieval-Augmented Generation (RAG) system built with **ChromaDB**, **LangChain**, **HuggingFace Embeddings**, and **FastAPI**.

## 🔴 Live Demo
🚀 **[Try the interactive UI live on Hugging Face Spaces!](https://huggingface.co/spaces/Keziya13/enterprise-rag-pipeline)**

## 🏗️ Architecture

```
PDF / TXT Documents
        ↓
  Document Loader (LangChain)
        ↓
  Sliding Window Chunking (512 tokens, 64 overlap)
        ↓
  HuggingFace Embeddings (all-MiniLM-L6-v2)
        ↓
  ChromaDB Vector Store (persisted)
        ↓
  MMR Retrieval (top-5, diversity-aware)
        ↓
  LLM (OpenAI GPT / Ollama Mistral)
        ↓
  FastAPI REST API → Answer + Sources
```

## 🚀 Features

- **ChromaDB** vector store with persistent storage
- **Sliding window chunking** (512 tokens, 64 token overlap) for better context
- **MMR retrieval** (Maximal Marginal Relevance) for diverse, relevant results
- **HuggingFace embeddings** (all-MiniLM-L6-v2) — no API key needed
- **FastAPI** REST API with async endpoints
- Supports **PDF and TXT** document ingestion
- Compatible with **OpenAI GPT** or **local Ollama** LLMs

## 📦 Setup

```bash
pip install -r requirements.txt
```

For local LLM (no OpenAI key needed):
```bash
# Install Ollama: https://ollama.com
ollama pull mistral
```

## ▶️ Run

```bash
uvicorn main:app --reload --port 8000
```

Then open: **http://localhost:8000/docs**

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/health` | Vectorstore status |
| `POST` | `/ingest/upload` | Upload PDF/TXT file |
| `POST` | `/ingest/path` | Ingest from server path |
| `POST` | `/query` | Ask a question |
| `GET` | `/stats` | Pipeline statistics |

## 📊 Example

```bash
# Ingest a document
curl -X POST "http://localhost:8000/ingest/upload" \
     -F "file=@document.pdf"

# Ask a question
curl -X POST "http://localhost:8000/query" \
     -H "Content-Type: application/json" \
     -d '{"question": "What is the main topic of the document?"}'
```

## 🛠️ Tech Stack

- **vector DB**: ChromaDB
- **embeddings**: HuggingFace `all-MiniLM-L6-v2`
- **orchestration**: LangChain
- **api**: FastAPI + Uvicorn
- **llm**: OpenAI GPT-3.5 or Ollama Mistral (local)
