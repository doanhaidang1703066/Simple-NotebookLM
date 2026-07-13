# Simple-NotebookLM

<![CDATA[<div align="center">

# 📚 RAG-Based Intelligent Learning System

**Design and Implementation of an Intelligent Learning Support System Based on Retrieval-Augmented Generation (RAG) Architecture**

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=for-the-badge&logo=llama&logoColor=white)](https://ollama.com/)
[![Qdrant](https://img.shields.io/badge/Qdrant-Vector_DB-DC382D?style=for-the-badge&logo=qdrant&logoColor=white)](https://qdrant.tech/)
[![LangChain](https://img.shields.io/badge/LangChain-1.3+-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)](https://langchain.com/)

*A privacy-preserving, locally-deployed RAG system that transforms academic PDF documents into an interactive learning environment with citation-backed Q&A, automated summarization, quiz generation, and flashcard creation.*

---

[Features](#-features) • [Architecture](#-architecture) • [Quick Start](#-quick-start) • [Usage](#-usage) • [API Reference](#-api-reference) • [Evaluation](#-evaluation) • [Tech Stack](#-tech-stack) • [License](#-license)

</div>

---

## 🎯 Problem Statement

Students face **information overload** when dealing with numerous academic PDF documents — lecture notes, research papers, and textbooks. Conventional LLMs suffer from hallucinations, lack domain-specific knowledge, cannot provide verifiable source citations, and risk exposing sensitive data through cloud APIs.

**This system solves all of these problems** by implementing Retrieval-Augmented Generation (RAG) to ground LLM responses in dynamically retrieved context from the student's personal document repository — ensuring **accurate**, **verifiable**, and **privacy-preserving** educational assistance, all running **locally on your machine**.

---

## ✨ Features

### 🔍 Citation-Backed Q&A
Ask any question about your uploaded documents and receive answers grounded in actual content with precise source citations (`[S1]`, `[S2]`) linking to specific pages and files.

### 📝 Intelligent Summarization (Single-Pass & Map-Reduce)
Generate executive summaries with key points. For lengthy documents exceeding LLM context limits, the system automatically switches to a **hierarchical Map-Reduce** architecture — splitting text into batches, summarizing each independently, then synthesizing a cohesive final overview.

### 🧠 Automated Quiz Generation (MCQ)
Generate rigorous multiple-choice quizzes with 4 options, correct answers, step-by-step explanations, difficulty ratings, topic tags, and source traceability. Self-validating Pydantic schemas enforce exactly 4 options and valid `correct_index` bounds.

### 🃏 Flashcard Generation
Create structured study flashcard decks with front (question/term), back (answer/definition), optional hints, topic tags, and source markers for effective spaced repetition study.

### 🔒 Privacy-First Local Deployment
Everything runs on your machine — Ollama for LLM inference with Apple Silicon Metal GPU acceleration, Qdrant in local file-based persistence mode. **Zero data leaves your computer.**

### 🌐 Full-Stack Web Application
Interactive Single-Page Application (SPA) built with vanilla HTML5/CSS3/JavaScript, plus a comprehensive RESTful API with auto-generated Swagger documentation.

---

## 🏗 Architecture

### System Overview

The system operates in two primary phases:

```
┌─────────────────────────── Phase 1: Knowledge Ingestion (Offline) ───────────────────────────┐
│                                                                                               │
│   📄 PDF Files ──▶ PyPDFLoader ──▶ Recursive/Semantic Splitter ──▶ Metadata Tagging           │
│                                          │                                                    │
│                                          ▼                                                    │
│                    GreenNode Embedding Model ──▶ 🗄️ Qdrant Vector Database                    │
│                                                                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘

┌─────────────────────── Phase 2: Retrieval & Generation (Online) ─────────────────────────────┐
│                                                                                               │
│   👤 User Request ──▶ Filter Coercion ──▶ Query Embedding ──▶ Qdrant Top-k Search             │
│                                                                      │                        │
│                                                                      ▼                        │
│                          Jinja2 Prompt Template ◀── Retrieved Chunks + Citations              │
│                                   │                                                           │
│                                   ▼                                                           │
│                       🤖 Ollama LLM (Qwen 2.5 3B) ──▶ Structured JSON Output                 │
│                                                                                               │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
```

### Map-Reduce Summarization Architecture

```
   MAP PHASE                                              REDUCE PHASE

 ┌──────────────┐
 │ Batch 1      │──┐
 │ (Chunks 1-10)│  │     ┌──────────┐     ┌─────────────┐
 └──────────────┘  ├────▶│ Ollama   │────▶│ Partial     │──┐
 ┌──────────────┐  │     │ LLM      │     │ Summary 1   │  │     ┌──────────┐     ┌─────────────────┐
 │ Batch 2      │──┤     │(Map      │     ├─────────────┤  ├────▶│ Ollama   │────▶│ Final Executive │
 │(Chunks 11-20)│  │     │ Prompt)  │     │ Partial     │  │     │ LLM      │     │ Summary &       │
 └──────────────┘  │     └──────────┘     │ Summary 2   │  │     │(Reduce   │     │ Key Points      │
       ⋮           │                      ├─────────────┤  │     │ Prompt)  │     └─────────────────┘
 ┌──────────────┐  │                      │ Partial     │──┘     └──────────┘
 │ Batch N      │──┘                      │ Summary N   │
 │ (Chunks ...) │                         └─────────────┘
 └──────────────┘
```

### Project Structure

```
notebook_lm/
├── data/                           # Input academic PDF repository
├── src/
│   ├── config.py                   # Centralized Pydantic Settings & validation
│   ├── schemas.py                  # Data contracts (Pydantic BaseModels)
│   ├── indexing.py                 # ETL pipeline: PDF loader, chunking, indexing
│   ├── store.py                    # Qdrant vector store & embedding singleton
│   ├── rag.py                      # Semantic retrieval & Q&A generation pipeline
│   ├── learning.py                 # Pedagogical tools (summary, quiz, flashcards)
│   ├── llm.py                      # Multi-provider LLM factory (Ollama/Gemini/vLLM)
│   ├── filters.py                  # Metadata filter normalization & Qdrant coercion
│   ├── export.py                   # Multi-format export engine (Markdown, JSON)
│   ├── utils.py                    # Helper utilities
│   ├── prompts/                    # Decoupled Jinja2 prompt templates
│   │   ├── answer.jinja2           # Q&A with source citations
│   │   ├── summary_single.jinja2   # Single-pass summarization
│   │   ├── summary_map.jinja2      # Map phase (batch summarization)
│   │   ├── summary_reduce.jinja2   # Reduce phase (synthesis)
│   │   ├── quiz.jinja2             # MCQ quiz generation
│   │   └── flashcards.jinja2       # Flashcard generation
│   ├── interfaces/
│   │   ├── api.py                  # RESTful backend (FastAPI)
│   │   └── frontend/              # Interactive Web SPA
│   │       ├── index.html
│   │       ├── style.css
│   │       └── app.js
│   └── evaluation/                 # RAGAS automated evaluation suite
│       ├── ragas_evaluator.py      # RAGAS metric configuration & wrapper
│       ├── chunking_strategy.py    # 7 experimental chunking strategies
│       ├── run_chunking.py         # Automated benchmark execution script
│       └── Data-Benchmark-Rag.csv  # 200-pair Q&A benchmark dataset
└── storage/qdrant/                 # Local persistent Qdrant vector storage
```

---

## Quick Start

### Prerequisites

| Requirement | Version | Purpose |
|---|---|---|
| Python | 3.10+ | Runtime environment |
| [Ollama](https://ollama.com/) | Latest | Local LLM inference engine |
| Git | Any | Clone repository |

### 1. Clone the Repository

```bash
git clone https://github.com/YOUR_USERNAME/rag-learning-system.git
cd rag-learning-system
```

### 2. Create Virtual Environment & Install Dependencies

```bash
python3 -m venv .rag_env
source .rag_env/bin/activate    # On macOS/Linux
pip install -r requirements.txt
```

### 3. Install & Pull Ollama Models

```bash
# Install Ollama (if not already installed)
# Download from https://ollama.com/download

# Pull required models
ollama pull qwen2.5:3b          # Primary answering model
ollama pull phi3.5:3.8b         # Judge model for evaluation
```

### 4. Add PDF Documents

Place your academic PDF files into the `notebook_lm/data/` directory:

```bash
cp /path/to/your/documents/*.pdf notebook_lm/data/
```

### 5. Launch the Application

```bash
cd notebook_lm/src
uvicorn interfaces.api:app --host 0.0.0.0 --port 8000 --reload
```

### 6. Open in Browser

| Interface | URL |
|---|---|
| 🌐 **Web Application** | [http://localhost:8000/app](http://localhost:8000/app) |
| 📖 **Swagger API Docs** | [http://localhost:8000/docs](http://localhost:8000/docs) |

### 7. Index Your Documents

On first launch, trigger document ingestion via the Web UI (upload button) or API:

```bash
curl -X POST http://localhost:8000/ingest
```

---

## Usage

### Web Interface

1. **Upload PDFs** — Drag & drop or click "Upload" in the sidebar
2. **Select Documents** — Check one or more documents from the Sources panel
3. **Chat / Q&A** — Ask questions and receive citation-backed answers
4. **Summarize** — Click "Generate Summary" for executive overviews with key points
5. **Quiz** — Click "Generate Quiz" for interactive MCQ assessments with explanations
6. **Flashcards** — Click "Generate Flashcards" for study cards (click to flip!)

### CLI / API Examples

```bash
# Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is Naive Bayes classification?"}'

# Summarize a specific document
curl -X POST http://localhost:8000/summarize \
  -H "Content-Type: application/json" \
  -d '{"document": "Operating System -Chap 1 - Eng.pdf"}'

# Generate a quiz
curl -X POST http://localhost:8000/quiz \
  -H "Content-Type: application/json" \
  -d '{"document": "Reading-Reasoning-LLMs.pdf", "count": 5}'

# Generate flashcards
curl -X POST http://localhost:8000/flashcards \
  -H "Content-Type: application/json" \
  -d '{"document": "Reading-LLM-Alignment.pdf", "count": 10}'
```

---

## API Reference

| Method | Endpoint | Description | Request Body | Response |
|---|---|---|---|---|
| `GET` | `/health` | System health check | — | `{"status": "ok"}` |
| `GET` | `/documents` | List all indexed PDFs | — | `list[DocumentInfo]` |
| `POST` | `/upload` | Upload & index a new PDF | `MultipartFile` | `UploadResponse` |
| `POST` | `/ingest` | Re-index all documents | `{"recreate": bool}` | `{"chunks_indexed": int}` |
| `POST` | `/ask` | RAG Q&A with citations | `AskRequest` | `RagAnswer` |
| `POST` | `/summarize` | Document summarization | `SummarizeRequest` | `Summary` |
| `POST` | `/quiz` | MCQ quiz generation | `QuizRequest` | `QuizSet` |
| `POST` | `/flashcards` | Flashcard generation | `FlashcardsRequest` | `FlashcardSet` |

> 💡 Full interactive API documentation available at `http://localhost:8000/docs`

---

## Evaluation

### RAGAS Benchmark Framework

The system includes a comprehensive automated evaluation suite using the **RAGAS** (Retrieval-Augmented Generation Assessment) framework, benchmarking across **4 core metrics**:

| Metric | What It Measures |
|---|---|
| **Faithfulness** | Factual consistency of answers against retrieved context (hallucination detection) |
| **Answer Relevancy** | How directly the answer addresses the user's question |
| **Context Precision** | Signal-to-noise ratio and ranking accuracy of retrieved chunks |
| **Context Recall** | Completeness of retrieved context against ground-truth answers |

### 7 Chunking Strategies Benchmarked

| Strategy | Type | Parameters |
|---|---|---|
| `rc_500_50` | Recursive Character | `chunk_size=500`, `overlap=50` |
| `rc_800_100` | Recursive Character | `chunk_size=800`, `overlap=100` |
| `rc_1000_150` | Recursive Character | `chunk_size=1000`, `overlap=150` |
| `rc_1500_200` | Recursive Character | `chunk_size=1500`, `overlap=200` |
| `semantic_percentile` | Semantic Embedding | `breakpoint_type="percentile"` |
| `semantic_std_dev` | Semantic Embedding | `breakpoint_type="standard_deviation"` |
| `semantic_interquartile` | Semantic Embedding | `breakpoint_type="interquartile"` |

### Running the Evaluation

```bash
cd notebook_lm/src
python -m evaluation.run_chunking
```

> Evaluation runs ~8,400 LLM inference calls (200 test cases × 7 strategies × 6 metric calls). Recommended to run on GPU-equipped machines (e.g., Google Colab / Kaggle with NVIDIA T4).

---

## Tech Stack

| Layer | Technology | Purpose |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **LLM Inference** | [Ollama](https://ollama.com/) + Qwen 2.5 3B | Local LLM with Apple Silicon Metal GPU acceleration |
| **Vector Database** | [Qdrant](https://qdrant.tech/) (Local Mode) | Persistent vector storage with payload indexing |
| **Embeddings** | GreenNode-Embedding-Large-VN-Mixed-V1 | Multilingual sentence embeddings (Vietnamese/English) |
| **Orchestration** | [LangChain](https://langchain.com/) 1.3+ | LLM chains, document loaders, text splitters |
| **Backend API** | [FastAPI](https://fastapi.tiangolo.com/) + Uvicorn | Async REST API with auto-generated OpenAPI docs |
| **Frontend** | Vanilla HTML5 / CSS3 / JavaScript | Lightweight SPA (no framework dependencies) |
| **Data Validation** | [Pydantic](https://docs.pydantic.dev/) v2 | Type-safe schemas, settings, and API contracts |
| **Prompt Engine** | [Jinja2](https://jinja.palletsprojects.com/) | Decoupled, programmable prompt templates |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) v0.4.3 | Automated RAG quality benchmarking |
| **PDF Processing** | PyPDFLoader (LangChain) | Academic PDF text extraction |

---

## Configuration

All settings are managed centrally via `src/config.py` using Pydantic Settings with environment variable overrides (prefix: `RAG_`):

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `1000` | Maximum characters per text chunk |
| `chunk_overlap` | `150` | Overlapping characters between consecutive chunks |
| `top_k` | `5` | Number of chunks retrieved per query |
| `embedding_model` | `GreenNode/GreenNode-Embedding-Large-VN-Mixed-V1` | Sentence embedding model |
| `llm_provider` | `ollama` | LLM backend (`ollama` / `gemini` / `vllm`) |
| `ollama_model` | `qwen2.5:3b` | Primary answering model |
| `ollama_judge_model` | `phi3.5:3.8b` | Evaluation judge model |
| `llm_temperature` | `0.1` | Generation temperature (0.0–0.2) |
| `summarize_batch_size` | `10` | Chunks per batch in Map-Reduce summarization |
| `generation_retrieval_k` | `8` | Max chunks for quiz/flashcard generation |
| `quiz_default_count` | `8` | Default number of quiz questions generated |
| `flashcards_default_count` | `15` | Default number of flashcards generated |

Override any setting via environment variables:
```bash
export RAG_CHUNK_SIZE=800
export RAG_LLM_PROVIDER=gemini
export GOOGLE_API_KEY=your_api_key_here
```

---

## Key Design Decisions

| Decision | Rationale |
|---|---|
| **Local Ollama over Cloud APIs** | Privacy preservation, zero cost, no internet dependency, Apple Silicon GPU acceleration |
| **Qdrant Local Mode over Docker** | Zero infrastructure overhead, single-process deployment, portable storage |
| **Jinja2 over LangChain Prompts** | Support for loops, conditionals, strict undefined variables — essential for complex RAG prompt composition |
| **Decoupled Judge Model** | Separating the answering model from the evaluation judge prevents self-preference bias during RAGAS benchmarking |
| **Pydantic Self-Validating Schemas** | `@model_validator` ensures data integrity at the boundary (e.g., QuizItem enforces exactly 4 options, valid `correct_index`) |
| **Deterministic Chunk IDs (SHA-1 + UUIDv5)** | Guarantees idempotent ingestion — re-uploading a PDF overwrites existing chunks instead of creating duplicates |
| **LLM Retry Logic with JSON Sanitization** | Small local models occasionally return empty or malformed JSON; automatic retry (3 attempts) with markdown fence stripping ensures reliability |

---

## Known Limitations

- **Single-Process Qdrant Lock:** In local file mode, only one process can access the database at a time. Don't run evaluation scripts while the web server is active.
- **Small Model JSON Compliance:** Ultra-compact LLMs (<4B parameters) occasionally produce malformed structured output, mitigated by retry logic and chunk count calibration.
- **PDF-Only Input:** Currently supports PDF documents only (no DOCX, Markdown, or web pages).
- **No Passage Reranking:** Retrieval relies solely on bi-encoder cosine similarity without cross-encoder reranking.

---

## Future Work

- [ ] **Cross-Encoder Reranking** — Two-stage retrieval with `BAAI/bge-reranker-v2-m3` for improved Context Precision
- [ ] **Real-Time Streaming** — Server-Sent Events (SSE) for token-by-token response streaming in the Web UI
- [ ] **Multi-Format Ingestion** — Support for DOCX, PowerPoint, LaTeX, and web pages
- [ ] **Hybrid Search** — Combining BM25 sparse retrieval with dense embeddings
- [ ] **Docker Compose Deployment** — Containerized multi-service architecture (FastAPI + Qdrant Server + vLLM)
- [ ] **Advanced RAG Paradigms** — Query rewriting, self-RAG, and agentic retrieval

---

## References

- Lewis, P. et al. (2020). *Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks.* NeurIPS 2020.
- Es, S. et al. (2023). *RAGAS: Automated Evaluation of Retrieval Augmented Generation.* arXiv:2309.15217.
- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS 2017.
- Qwen Team (2024). *Qwen2.5 Technical Report.* arXiv:2412.15115.
- Nogueira, R. & Cho, K. (2019). *Passage Re-ranking with BERT.* arXiv:1901.04085.

---

## 📄 License

This project was developed as part of **Project I** at the university program. For academic use and reference.

---

<div align="center">

**Built with love using Ollama, Qdrant, LangChain, and FastAPI**

*Running locally. No data leaves your machine.* 

</div>
]]>
