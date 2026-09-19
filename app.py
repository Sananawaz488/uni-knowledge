import json
from pathlib import Path

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

APP_TITLE = "University Academic Knowledge Assistant"

FAISS_DIR = Path(__file__).parent / "faiss-index"

FAISS_INDEX_PATH = FAISS_DIR / "academic_documents.faiss"
METADATA_PATH = FAISS_DIR / "metadata.json"
CONFIG_PATH = FAISS_DIR / "config.json"

GROQ_MODEL = "openai/gpt-oss-120b"

TOP_K = 5


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title=APP_TITLE,
    page_icon="🎓",
    layout="wide"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #6b7280;
        font-size: 1rem;
        margin-bottom: 1.5rem;
    }

    .source-box {
        padding: 0.8rem;
        border-radius: 0.6rem;
        border: 1px solid #d1d5db;
        margin-bottom: 0.6rem;
        background-color: #f9fafb;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 University Academic Knowledge Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Ask questions from the university academic knowledge base.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LOAD CONFIGURATION
# ============================================================

@st.cache_data
def load_config():

    if not CONFIG_PATH.exists():
        raise FileNotFoundError(
            f"Configuration file not found: {CONFIG_PATH}"
        )

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Metadata file not found: {METADATA_PATH}"
        )

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD FAISS INDEX
# ============================================================

@st.cache_resource
def load_faiss_index():

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}"
        )

    return faiss.read_index(str(FAISS_INDEX_PATH))


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):

    return SentenceTransformer(model_name)


# ============================================================
# LOAD GROQ CLIENT
# ============================================================

@st.cache_resource
def load_groq_client():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        raise ValueError(
            "GROQ_API_KEY is missing. "
            "Add it to Streamlit Secrets."
        )

    return Groq(api_key=api_key)


# ============================================================
# INITIALIZE RAG COMPONENTS
# ============================================================

try:

    config = load_config()
    metadata = load_metadata()
    index = load_faiss_index()

    embedding_model = load_embedding_model(
        config["embedding_model"]
    )

    groq_client = load_groq_client()

except Exception as error:

    st.error(f"Application initialization failed: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    st.write(
        f"**Documents:** "
        f"{len(set(item['metadata']['source'] for item in metadata))}"
    )

    st.write(
        f"**Knowledge chunks:** {len(metadata)}"
    )

    st.write(
        f"**Vector dimension:** "
        f"{config['embedding_dimension']}"
    )

    st.write(
        f"**Embedding model:** "
        f"`{config['embedding_model']}`"
    )

    st.write(
        f"**Retrieval:** Top {TOP_K} chunks"
    )

    st.divider()

    st.caption(
        "Answers are generated from the indexed academic "
        "knowledge base."
    )


# ============================================================
# RETRIEVAL FUNCTION
# ============================================================

def retrieve_documents(query: str, top_k: int = TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):

        if idx < 0:
            continue

        item = metadata[int(idx)]

        results.append(
            {
                "text": item["text"],
                "metadata": item["metadata"],
                "score": float(score)
            }
        )

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context_parts = []

    for number, result in enumerate(results, start=1):

        source = result["metadata"].get(
            "source",
            "Unknown source"
        )

        page = result["metadata"].get(
            "page",
            "Unknown page"
        )

        chunk_id = result["metadata"].get(
            "chunk_id",
            "Unknown"
        )

        context_parts.append(
            f"""
SOURCE {number}
File: {source}
Page: {page}
Chunk: {chunk_id}

Content:
{result["text"]}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# GROQ ANSWER GENERATION
# ============================================================

def generate_answer(question, results):

    context = build_context(results)

    system_prompt = """
You are a university academic knowledge assistant.

Your job is to answer questions using ONLY the retrieved
academic context supplied by the application.

Rules:

1. Use the provided context as your primary knowledge source.
2. Do not invent facts that are not supported by the context.
3. If the context does not contain enough information, clearly
   say that the answer could not be found in the indexed
   academic documents.
4. Give a clear and academically useful answer.
5. When making factual claims, include source references
   using the source numbers provided in the context.
6. Do not create fake page numbers or fake sources.
7. Keep the answer focused on the student's question.

Example citation format:

[Source 1, Page 5]

or

[Source 2, Page 12]
"""

    user_prompt = f"""
Student question:

{question}

Retrieved academic context:

{context}

Answer the student's question based on the retrieved context.
Include source references with page numbers.
"""

    completion = groq_client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        temperature=0.2,
        max_completion_tokens=1500
    )

    return completion.choices[0].message.content


# ============================================================
# DISPLAY SOURCES
# ============================================================

def display_sources(results):

    st.subheader("📚 Sources")

    for number, result in enumerate(results, start=1):

        metadata_item = result["metadata"]

        source = metadata_item.get(
            "source",
            "Unknown"
        )

        page = metadata_item.get(
            "page",
            "Unknown"
        )

        chunk_id = metadata_item.get(
            "chunk_id",
            "Unknown"
        )

        score = result["score"]

        with st.expander(
            f"Source {number} — {source} | Page {page}"
        ):

            st.write(
                f"**File:** {source}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Chunk:** {chunk_id}"
            )

            st.write(
                f"**Similarity score:** {score:.4f}"
            )

            st.markdown("**Retrieved content:**")

            st.write(result["text"])


# ============================================================
# CHAT INTERFACE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and "sources" in message
        ):

            display_sources(message["sources"])


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask a question about your academic documents..."
)


if question:

    # Display user question
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    # Retrieve relevant chunks
    with st.spinner("Searching academic knowledge base..."):

        retrieved_results = retrieve_documents(
            question,
            TOP_K
        )

    if not retrieved_results:

        answer = (
            "I could not find relevant information in "
            "the indexed academic documents."
        )

    else:

        # Generate answer using Groq
        with st.spinner("Generating academic answer..."):

            try:

                answer = generate_answer(
                    question,
                    retrieved_results
                )

            except Exception as error:

                answer = (
                    "An error occurred while generating "
                    f"the answer: {error}"
                )

    # Display answer
    with st.chat_message("assistant"):

        st.markdown(answer)

        if retrieved_results:
            display_sources(retrieved_results)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved_results
        }
    )
