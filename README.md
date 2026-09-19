# 🎓 University Academic Knowledge Assistant

A Retrieval-Augmented Generation (RAG) application designed
for university students to ask questions from a curated
academic knowledge base.

The application uses pre-computed document embeddings and
FAISS for vector retrieval, with Groq's
`openai/gpt-oss-120b` model for answer generation.

---

## 🚀 Features

- Academic knowledge-based question answering
- Retrieval-Augmented Generation (RAG)
- Multiple PDF document support
- FAISS vector database
- Pre-computed document embeddings
- Source and page-level traceability
- Retrieved-context display
- Groq `openai/gpt-oss-120b`
- Streamlit user interface
- GitHub deployment
- Streamlit Community Cloud deployment

---

## 🏗️ Architecture

```text
Academic PDFs
     │
     ▼
PDF Text Extraction
     │
     ▼
Text Chunking
     │
     ▼
Sentence Transformer
     │
     ▼
Document Embeddings
     │
     ▼
FAISS Vector Database
     │
     ├── academic_documents.faiss
     ├── metadata.json
     └── config.json
     
             DEPLOYMENT
                 │
                 ▼
        ┌───────────────────┐
        │ Streamlit App     │
        └─────────┬─────────┘
                  │
            Student Query
                  │
                  ▼
          Query Embedding
                  │
                  ▼
             FAISS Search
                  │
                  ▼
        Top Relevant Chunks
                  │
                  ▼
        Metadata + Sources
                  │
                  ▼
       Groq GPT-OSS-120B
                  │
                  ▼
       Answer + Citations
