import json
from pathlib import Path

import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# =========================
# SETTINGS
# =========================

BASE_DIR = Path(__file__).parent
FAISS_DIR = BASE_DIR / "faiss-index"

FAISS_INDEX_PATH = FAISS_DIR / "academic_documents.faiss"
METADATA_PATH = FAISS_DIR / "metadata.json"
CONFIG_PATH = FAISS_DIR / "config.json"

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5


# =========================
# PAGE CONFIGURATION
# =========================

st.set_page_config(
    page_title="Uni Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


# =========================
# CUSTOM UI
# =========================

st.markdown(
    """
    <style>

    /* =========================
       MAIN APP
       ========================= */

    .stApp {
        background: #f8fafc;
    }

    .main .block-container {
        max-width: 1200px;
        padding-top: 2.5rem;
        padding-bottom: 4rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }


    /* =========================
       SIDEBAR
       ========================= */

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #eef6ff 0%,
            #f8fbff 100%
        );

        border-right: 1px solid #dbe7f3;
    }

    [data-testid="stSidebar"] .block-container {
        padding-top: 2rem;
    }

    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #172554;
    }

    [data-testid="stSidebar"] p {
        color: #475569;
    }


    /* =========================
       TITLE
       ========================= */

    .app-title {
        font-size: 2.8rem;
        font-weight: 800;
        color: #172554;
        line-height: 1.15;
        margin-bottom: 10px;
    }

    .app-subtitle {
        font-size: 1.05rem;
        color: #64748b;
        margin-bottom: 30px;
    }


    /* =========================
       INFO CARDS
       ========================= */

    .info-card {
        background: white;
        border: 1px solid #e2e8f0;
        border-radius: 16px;
        padding: 18px 20px;
        margin-bottom: 12px;
        box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
    }

    .info-title {
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 5px;
    }

    .info-value {
        color: #172554;
        font-size: 1.35rem;
        font-weight: 700;
    }


    /* =========================
       CHAT INPUT
       ========================= */

    [data-testid="stChatInput"] {
        border-radius: 16px;
    }

    [data-testid="stChatInput"] textarea {
        background: white !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px !important;
        padding: 14px !important;
        font-size: 1rem !important;
    }

    [data-testid="stChatInput"] textarea:focus {
        border: 1px solid #6366f1 !important;
        box-shadow: 0 0 0 2px rgba(99, 102, 241, 0.12) !important;
    }


    /* =========================
       CHAT MESSAGES
       ========================= */

    [data-testid="stChatMessage"] {
        border-radius: 14px;
        margin-bottom: 12px;
        padding: 8px;
    }


    /* =========================
       SOURCE EXPANDERS
       ========================= */

    [data-testid="stExpander"] {
        background: white;
        border: 1px solid #dbe4ee;
        border-radius: 12px;
        margin-bottom: 10px;
    }


    /* =========================
       BUTTONS
       ========================= */

    .stButton button {
        border-radius: 10px;
        border: 1px solid #cbd5e1;
    }


    /* =========================
       DIVIDERS
       ========================= */

    hr {
        border-color: #dbe7f3;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# =========================
# HEADER
# =========================

st.markdown(
    """
    <div class="app-title">
        🎓 University Academic Knowledge Assistant
    </div>

    <div class="app-subtitle">
        Ask questions and get answers from the indexed
        university academic knowledge base.
    </div>
    """,
    unsafe_allow_html=True
)


# =========================
# LOAD FILES
# =========================

@st.cache_data
def load_config():
    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_data
def load_metadata():
    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


@st.cache_resource
def load_faiss():
    return faiss.read_index(str(FAISS_INDEX_PATH))


# =========================
# LAZY MODEL LOADING
# =========================

@st.cache_resource
def load_embedding_model(model_name):
    return SentenceTransformer(model_name)


@st.cache_resource
def load_groq():
    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error("GROQ_API_KEY is missing from Streamlit Secrets.")
        st.stop()

    return Groq(api_key=api_key)


# =========================
# LOAD DATA
# =========================

try:
    config = load_config()
    metadata = load_metadata()
    faiss_index = load_faiss()

except Exception as error:
    st.error(f"Application startup error: {error}")
    st.stop()


# =========================
# SIDEBAR
# =========================

with st.sidebar:

    st.markdown("## 📚 Knowledge Base")

    sources = set(
        item["metadata"]["source"]
        for item in metadata
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">📄 Documents</div>
            <div class="info-value">{len(sources)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">🧩 Knowledge Chunks</div>
            <div class="info-value">{len(metadata)}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="info-card">
            <div class="info-title">🔢 Embedding Dimension</div>
            <div class="info-value">
                {config["embedding_dimension"]}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.divider()

    st.markdown("### ℹ️ About")

    st.caption(
        "Original PDF documents are not stored in this application."
    )

    st.caption(
        "The assistant uses the pre-built FAISS index "
        "and metadata to retrieve relevant academic information."
    )


# =========================
# RETRIEVAL
# =========================

def retrieve_documents(query):

    embedding_model = load_embedding_model(
        config["embedding_model"]
    )

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = faiss_index.search(
        query_embedding,
        TOP_K
    )

    results = []

    for score, index_id in zip(scores[0], indices[0]):

        if index_id < 0:
            continue

        item = metadata[int(index_id)]

        results.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": float(score)
        })

    return results


# =========================
# BUILD CONTEXT
# =========================

def build_context(results):

    context = []

    for number, result in enumerate(results, start=1):

        meta = result["metadata"]

        context.append(
            f"""
SOURCE {number}

Document: {meta.get("source", "Unknown")}
Page: {meta.get("page", "Unknown")}
Chunk ID: {meta.get("chunk_id", "Unknown")}

Content:
{result["text"]}
"""
        )

    return "\n".join(context)


# =========================
# GENERATE ANSWER
# =========================

def generate_answer(question, results):

    groq_client = load_groq()

    context = build_context(results)

    system_prompt = """
You are a university academic knowledge assistant.

Answer the student's question using ONLY the retrieved
academic context provided to you.

Rules:

1. Do not invent information.
2. Do not use outside knowledge.
3. If the answer is not available in the retrieved
   context, clearly say that the information was not
   found in the indexed academic documents.
4. Give a clear and useful academic answer.
5. Cite the supporting source and page.
6. Never invent a source or page number.
7. Keep the answer relevant to the question.

Use this citation format:

[Source 1, Page 5]

[Source 2, Page 12]
"""

    user_prompt = f"""
Student question:

{question}

Retrieved academic context:

{context}

Answer using ONLY the retrieved context.
Include source citations with page numbers.
"""

    response = groq_client.chat.completions.create(
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

    return response.choices[0].message.content


# =========================
# SHOW SOURCES
# =========================

def show_sources(results):

    st.subheader("📚 Retrieved Sources")

    for number, result in enumerate(results, start=1):

        meta = result["metadata"]

        source = meta.get("source", "Unknown")
        page = meta.get("page", "Unknown")
        chunk_id = meta.get("chunk_id", "Unknown")
        score = result["score"]

        with st.expander(
            f"Source {number}: {source} — Page {page}"
        ):

            st.write(f"**Document:** {source}")
            st.write(f"**Page:** {page}")
            st.write(f"**Chunk ID:** {chunk_id}")

            st.write(
                f"**Similarity score:** {score:.4f}"
            )

            st.markdown("**Retrieved content:**")

            st.write(result["text"])


# =========================
# CHAT HISTORY
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# =========================
# QUESTION INPUT
# =========================

question = st.chat_input(
    "💬 Ask an academic question..."
)


# =========================
# PROCESS QUESTION
# =========================

if question:

    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    try:

        with st.spinner(
            "🔎 Searching academic knowledge base..."
        ):

            results = retrieve_documents(question)

        if not results:

            answer = (
                "I could not find relevant information "
                "in the indexed academic documents."
            )

        else:

            with st.spinner(
                "🤖 Generating academic answer..."
            ):

                answer = generate_answer(
                    question,
                    results
                )

        with st.chat_message("assistant"):

            st.markdown(answer)

            if results:
                show_sources(results)

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer
        })

    except Exception as error:

        error_message = (
            f"Something went wrong:\n\n"
            f"`{error}`"
        )

        with st.chat_message("assistant"):
            st.error(error_message)

        st.session_state.messages.append({
            "role": "assistant",
            "content": error_message
        })
