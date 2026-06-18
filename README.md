# Telecom Customer Care Chatbot (RAG)

## Overview

This project is a Retrieval-Augmented Generation (RAG) based Telecom Customer Care Chatbot that assists users with telecom-related issues using information from:

- FAQs
- Resolved Support Tickets
- Telecom User Guides (PDFs)

The chatbot retrieves relevant information from a vector database and uses an LLM to generate grounded responses with source citations.

---

## Features

### Multi-Source Retrieval
Retrieves information from three knowledge sources:

- FAQ Database
- Historical Resolved Tickets
- Telecom Guides

### Source Citations
Every response includes citations indicating where the information was retrieved from.

Example:

```
Restart your phone after inserting the SIM.
(Source: FAQ-12)

If activation still fails, contact support and provide the SIM serial number.
(Source: Ticket-TK-001)
```

### Confidence-Based Fallback
Uses similarity scores from ChromaDB retrieval.

If no retrieved document exceeds a confidence threshold:

```
I don't know. Please call 611.
```

The system skips the LLM entirely, reducing hallucinations and API costs.

### Retrieval Evaluation
Implemented Recall@3 evaluation using manually created query-ticket pairs.

Result:

```
Recall@3 = 100%
```

---

## Tech Stack

- Python
- LangChain
- ChromaDB
- HuggingFace Embeddings
- Groq API
- Qwen3-32B
- SQLite

---

## Project Structure

```
telecom_chatbot/
│
├── data/
│   ├── faq.csv
│   ├── tickets.db
│   └── telecom_guides.pdf
│
├── ingest_faq.py
├── ingest_tickets.py
├── ingest_guides.py
│
├── retriever.py
├── rag_chain.py
├── main.py
├── evaluate.py
│
└── Chroma_store/
```

---

## How It Works

User Question
↓
Retriever (FAQ + Tickets + Guides)
↓
Similarity Score Check
↓
Low Confidence?
├── Yes → Return fallback response
└── No → Send context to LLM
↓
Generate cited response

---

## Evaluation

The retriever was evaluated using 10 manually crafted query-ticket pairs.

Metric:

- Recall@3

Formula:

Recall@3 = Correct Retrievals / Total Queries

Result:

- 10/10 successful retrievals
- Recall@3 = 100%

---

## Future Improvements

- Cross-encoder reranking
- Hybrid Search (BM25 + Vector Search)
- Conversation Memory
- LangGraph Workflow
- Web Interface using Streamlit
- Agentic RAG
