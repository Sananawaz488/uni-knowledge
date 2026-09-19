import json
from pathlib import Path

import faiss
import streamlit as st
from groq import Groq
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = Path(__file__).parent
FAISS_DIR = BASE_DIR / "faiss-index"

FAISS_INDEX_PATH = FAISS_DIR / "academic_documents.faiss"
METADATA_PATH = FAISS_DIR / "metadata.json"
CONFIG_PATH = FAISS_DIR / "config.json"

GROQ_MODEL = "openai/gpt-oss-120b"
TOP_K = 5


# ============================================================
# PAGE
# ============================================================

st.set_page_config(
    page_title="Uni Knowledge Assistant",
    page_icon="🎓",
    layout="wide"
)


st.title("🎓 University Academic Knowledge Assistant")

st.write(
    "Ask questions from the indexed university academic "
    "knowledge base."
)


# ============================================================
# LOAD CONFIG
# ============================================================

@st.cache_data
def load_config():

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD METADATA
# ============================================================

@st.cache_data
def load_metadata():

    with open(METADATA_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# LOAD FAISS
# ============================================================

@st.cache_resource
def load_faiss():

    return faiss.read_index(
        str(FAISS_INDEX_PATH)
    )


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model(model_name):

    return SentenceTransformer(model_name)


# ============================================================
# LOAD GROQ
# ============================================================

@st.cache_resource
def load_groq():

    api_key = st.secrets.get("GROQ_API_KEY")

    if not api_key:
        st.error(
            "GROQ_API_KEY is not configured. "
            "Please add it in Streamlit Secrets."
        )
        st.stop()

    return Groq(api_key=api_key)


# ============================================================
# INITIALIZE
# ============================================================

try:

    config = load_config()
    metadata = load_metadata()
    faiss_index = load_faiss()

    embedding_model = load_embedding_model(
        config["embedding_model"]
    )

    groq_client = load_groq()

except Exception as error:

    st.error(f"Application initialization error: {error}")
    st.stop()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📚 Knowledge Base")

    sources = set(
        item["metadata"]["source"]
        for item in metadata
    )

    st.write(f"**Documents:** {len(sources)}")
    st.write(f"**Chunks:** {len(metadata)}")
    st.write(
        f"**Embedding dimension:** "
        f"{config['embedding_dimension']}"
    )

    st.divider()

    st.caption(
        "Original PDF documents are not stored in this "
        "application. The assistant uses the pre-built "
        "FAISS index and metadata."
    )


# ============================================================
# RETRIEVE
# ============================================================

def retrieve_documents(query, top_k=TOP_K):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True
    ).astype("float32")

    scores, indices = faiss_index.search(
        query_embedding,
        top_k
    )

    results = []

    for score, index_id in zip(
        scores[0],
        indices[0]
    ):

        if index_id < 0:
            continue

        item = metadata[int(index_id)]

        results.append({
            "text": item["text"],
            "metadata": item["metadata"],
            "score": float(score)
        })

    return results


# ============================================================
# BUILD CONTEXT
# ============================================================

def build_context(results):

    context = []

    for number, result in enumerate(
        results,
        start=1
    ):

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


# ============================================================
# GENERATE ANSWER
# ============================================================

def generate_answer(question, results):

    context = build_context(results)

    system_prompt = """
You are a university academic knowledge assistant.

Answer the student's question using ONLY the retrieved
academic context provided to you.

Rules:

1. Do not invent information.
2. Do not use outside knowledge when answering.
3. If the retrieved context does not contain enough
   information, say that the information was not found
   in the indexed academic documents.
4. Give a clear and academically useful answer.
5. Cite supporting sources using the provided source number
   and page number.
6. Never invent a source or page number.
7. Keep the answer relevant to the student's question.

Citation format:

[Source 1, Page 5]

[Source 2, Page 12]
"""

    user_prompt = f"""
Student question:

{question}

Retrieved academic context:

{context}

Answer the question using the retrieved context.
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


# ============================================================
# DISPLAY SOURCES
# ============================================================

def show_sources(results):

    st.subheader("📚 Retrieved Sources")

    for number, result in enumerate(
        results,
        start=1
    ):

        meta = result["metadata"]

        source = meta.get(
            "source",
            "Unknown"
        )

        page = meta.get(
            "page",
            "Unknown"
        )

        chunk_id = meta.get(
            "chunk_id",
            "Unknown"
        )

        score = result["score"]

        with st.expander(
            f"Source {number}: {source} — Page {page}"
        ):

            st.write(
                f"**Document:** {source}"
            )

            st.write(
                f"**Page:** {page}"
            )

            st.write(
                f"**Chunk ID:** {chunk_id}"
            )

            st.write(
                f"**Similarity score:** {score:.4f}"
            )

            st.markdown("**Retrieved content:**")

            st.write(result["text"])


# ============================================================
# CHAT HISTORY
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


# ============================================================
# USER QUESTION
# ============================================================

question = st.chat_input(
    "Ask an academic question..."
)


if question:

    # User message
    with st.chat_message("user"):

        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Retrieval
    with st.spinner(
        "Searching academic knowledge base..."
    ):

        results = retrieve_documents(
            question,
            TOP_K
        )


    if not results:

        answer = (
            "I could not find relevant information "
            "in the indexed academic documents."
        )

    else:

        # LLM
        with st.spinner(
            "Generating answer..."
        ):

            try:

                answer = generate_answer(
                    question,
                    results
                )

            except Exception as error:

                answer = (
                    "An error occurred while generating "
                    f"the answer: {error}"
                )


    # Assistant response
    with st.chat_message("assistant"):

        st.markdown(answer)

        if results:
            show_sources(results)


    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })
