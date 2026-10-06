"""
RAG-Based HR Assistant
Professional Streamlit Application
Author: Reem Fayyaz

Run:
    streamlit run app.py

Optional environment variables:
    OPENAI_API_KEY=your_key
    GEMINI_API_KEY=your_key

The application works without an API key using a local
retrieval-based answer mode. If an API key is configured,
the selected LLM can generate more natural answers from
the retrieved HR context.
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import faiss


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="RAG HR Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PROFESSIONAL STYLING
# ============================================================

st.markdown(
    """
    <style>
        .stApp {
            background: linear-gradient(135deg, #f7f9fc 0%, #eef3f9 100%);
        }

        .main-header {
            padding: 1.4rem 1.6rem;
            border-radius: 18px;
            background: linear-gradient(135deg, #172554, #2563eb);
            color: white;
            margin-bottom: 1.2rem;
            box-shadow: 0 10px 30px rgba(37, 99, 235, 0.18);
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.1rem;
            font-weight: 800;
        }

        .main-header p {
            margin: 0.45rem 0 0;
            opacity: 0.9;
            font-size: 1rem;
        }

        .metric-card {
            background: white;
            border-radius: 16px;
            padding: 1rem;
            border: 1px solid #e5e7eb;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
            min-height: 105px;
        }

        .metric-title {
            color: #64748b;
            font-size: 0.85rem;
            font-weight: 600;
        }

        .metric-value {
            color: #172554;
            font-size: 1.55rem;
            font-weight: 800;
            margin-top: 0.35rem;
        }

        .answer-box {
            background: white;
            border-left: 5px solid #2563eb;
            border-radius: 14px;
            padding: 1.2rem 1.3rem;
            margin: 0.8rem 0 1rem;
            box-shadow: 0 5px 18px rgba(15, 23, 42, 0.06);
        }

        .source-box {
            background: #f8fafc;
            border: 1px solid #e2e8f0;
            border-radius: 12px;
            padding: 0.9rem 1rem;
            margin: 0.5rem 0;
        }

        .source-name {
            color: #1d4ed8;
            font-weight: 750;
        }

        .source-score {
            color: #64748b;
            font-size: 0.82rem;
        }

        .small-note {
            color: #64748b;
            font-size: 0.85rem;
        }

        div[data-testid="stSidebar"] {
            background: #f8fafc;
        }

        .stButton > button {
            border-radius: 10px;
            font-weight: 650;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DEMO HR KNOWLEDGE BASE
# ============================================================

DEMO_DOCUMENTS = {
    "leave_policy.txt": """
HR LEAVE POLICY

Employees are entitled to annual paid leave according to
their employment terms.

Employees should submit leave requests through the official
HR portal before taking planned leave.

For emergency or sick leave, employees should inform their
reporting manager and HR as soon as possible.

Leave approval depends on business requirements, employee
eligibility, and manager approval.

Employees should not assume that a leave request has been
approved until confirmation is received.
""",
    "work_from_home_policy.txt": """
WORK FROM HOME POLICY

Work-from-home availability depends on the employee's role,
department, and company policy.

Employees must submit a WFH request through the approved HR
process.

Manager approval may be required before working remotely.

Employees working remotely are expected to remain available
during approved working hours and follow company security
policies.
""",
    "attendance_policy.txt": """
ATTENDANCE POLICY

Employees are expected to follow the company's working hours
and attendance requirements.

Employees must record attendance correctly on every working day.

If an employee forgets to mark attendance or notices an
attendance discrepancy, they should contact HR or their
reporting manager.

Repeated attendance issues may be reviewed according to
company policy.
""",
    "payroll_policy.txt": """
PAYROLL POLICY

Salary is processed according to the company's payroll calendar.

Employees can normally access their payslips through the
employee portal.

Questions regarding salary deductions, taxes, payslips, or
salary discrepancies should be directed to the Payroll or HR team.

Employees should provide relevant employee information when
reporting a payroll issue.
""",
    "resignation_policy.txt": """
RESIGNATION POLICY

Employees who wish to resign should submit a formal resignation
through the company's approved process.

The applicable notice period depends on the employee's employment
agreement and company policy.

Employees should coordinate with their manager and HR regarding
the resignation process, handover responsibilities, final
working date, and company property.

Employees should contact HR if they need clarification about
their specific notice period or final settlement.
""",
}


# ============================================================
# TEXT UTILITIES
# ============================================================

def clean_text(text: str) -> str:
    """Normalize whitespace while preserving readable text."""
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()


def create_chunks(text: str, chunk_size: int = 100, overlap: int = 20):
    """Split text into overlapping word-based chunks."""
    words = clean_text(text).split()

    if not words:
        return []

    chunks = []
    step = max(1, chunk_size - overlap)

    for start in range(0, len(words), step):
        chunk = " ".join(words[start:start + chunk_size])
        if chunk:
            chunks.append(chunk)

        if start + chunk_size >= len(words):
            break

    return chunks


def load_uploaded_documents(uploaded_files):
    """Read TXT and PDF files uploaded through Streamlit."""
    documents = []

    for uploaded_file in uploaded_files:
        filename = uploaded_file.name

        try:
            if filename.lower().endswith(".pdf"):
                reader = PdfReader(uploaded_file)
                text = "\n".join(
                    page.extract_text() or ""
                    for page in reader.pages
                )
            else:
                text = uploaded_file.getvalue().decode(
                    "utf-8",
                    errors="ignore",
                )

            text = clean_text(text)

            if text:
                documents.append(
                    {
                        "filename": filename,
                        "text": text,
                    }
                )

        except Exception as exc:
            st.warning(
                f"Could not read **{filename}**: {exc}"
            )

    return documents


def build_chunks(documents, chunk_size=100, overlap=20):
    """Create a DataFrame containing chunk text and source."""
    rows = []

    for document in documents:
        for chunk in create_chunks(
            document["text"],
            chunk_size=chunk_size,
            overlap=overlap,
        ):
            rows.append(
                {
                    "filename": document["filename"],
                    "text": chunk,
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# MODEL LOADING
# ============================================================

@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Load the same embedding model used by the notebook."""
    return SentenceTransformer("all-MiniLM-L6-v2")


@st.cache_resource(show_spinner=False)
def build_vector_index(texts_tuple):
    """
    Build and cache the FAISS index.

    texts_tuple is used instead of a DataFrame because Streamlit
    caching works more reliably with immutable inputs.
    """
    texts = list(texts_tuple)

    model = load_embedding_model()

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    embeddings = np.asarray(embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    return index


# ============================================================
# RETRIEVAL
# ============================================================

def retrieve_documents(
    query: str,
    chunks_df: pd.DataFrame,
    index,
    top_k: int = 3,
):
    """Retrieve the most semantically relevant HR chunks."""

    if chunks_df.empty or index is None:
        return []

    model = load_embedding_model()

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    query_embedding = np.asarray(
        query_embedding,
        dtype="float32",
    )

    k = min(top_k, len(chunks_df))

    scores, indices = index.search(
        query_embedding,
        k,
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue

        row = chunks_df.iloc[int(idx)]

        results.append(
            {
                "score": float(score),
                "filename": row["filename"],
                "text": row["text"],
            }
        )

    return results


# ============================================================
# LOCAL ANSWER GENERATION
# ============================================================

def extractive_answer(query, retrieved_documents):
    """
    Safe fallback answer when no external LLM is configured.

    It returns the relevant retrieved policy information instead
    of inventing HR rules.
    """

    if not retrieved_documents:
        return (
            "I could not find relevant information in the HR "
            "knowledge base. Please contact HR for clarification."
        )

    best_score = retrieved_documents[0]["score"]

    if best_score < 0.25:
        return (
            "I could not find sufficiently relevant information "
            "in the HR knowledge base. Please contact HR for "
            "clarification rather than relying on an assumption."
        )

    best = retrieved_documents[0]

    return (
        f"Based on the available HR policy information:\n\n"
        f"{best['text']}\n\n"
        f"Please contact HR or your reporting manager if your "
        f"specific situation requires approval or additional "
        f"clarification."
    )


# ============================================================
# OPTIONAL OPENAI GENERATION
# ============================================================

def generate_with_openai(query, retrieved_documents, api_key):
    """Generate an answer using OpenAI if an API key is supplied."""

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        context = "\n\n".join(
            f"Source: {item['filename']}\n{item['text']}"
            for item in retrieved_documents
        )

        prompt = f"""
You are a professional HR Assistant.

Answer the employee's question using ONLY the HR policy
context below.

Rules:
- Never invent a company policy.
- If the context does not contain the answer, say that the
  information is not available and advise the employee to
  contact HR.
- Keep the answer clear, professional, and concise.
- Mention relevant policy conditions or approvals.
- Do not claim that a request is approved unless the context
  explicitly says so.

HR POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}
"""

        response = client.responses.create(
            model="gpt-4.1-mini",
            input=prompt,
        )

        return response.output_text.strip()

    except Exception as exc:
        return (
            "The OpenAI service could not be reached, so I am "
            "showing the retrieved HR policy information instead.\n\n"
            + extractive_answer(query, retrieved_documents)
            + f"\n\nTechnical note: {exc}"
        )


# ============================================================
# OPTIONAL GEMINI GENERATION
# ============================================================

def generate_with_gemini(query, retrieved_documents, api_key):
    """Generate an answer using Google Gemini if configured."""

    try:
        from google import genai

        client = genai.Client(api_key=api_key)

        context = "\n\n".join(
            f"Source: {item['filename']}\n{item['text']}"
            for item in retrieved_documents
        )

        prompt = f"""
You are a professional HR Assistant.

Use ONLY the following HR policy context to answer the
employee's question. Do not invent rules or benefits.

If the answer is not present, clearly state that it is not
available in the provided HR knowledge base and recommend
contacting HR.

HR POLICY CONTEXT:
{context}

EMPLOYEE QUESTION:
{query}
"""

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
        )

        return response.text.strip()

    except Exception as exc:
        return (
            "The Gemini service could not be reached, so I am "
            "showing the retrieved HR policy information instead.\n\n"
            + extractive_answer(query, retrieved_documents)
            + f"\n\nTechnical note: {exc}"
        )


def generate_answer(query, retrieved_documents, provider):
    """Choose local or external generation."""
    if provider == "OpenAI":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if api_key:
            return generate_with_openai(
                query,
                retrieved_documents,
                api_key,
            )

    if provider == "Gemini":
        api_key = os.getenv("GEMINI_API_KEY", "").strip()
        if api_key:
            return generate_with_gemini(
                query,
                retrieved_documents,
                api_key,
            )

    return extractive_answer(
        query,
        retrieved_documents,
    )


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "documents" not in st.session_state:
    st.session_state.documents = [
        {"filename": name, "text": text}
        for name, text in DEMO_DOCUMENTS.items()
    ]


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.markdown("## ⚙️ Assistant Settings")

    provider = st.selectbox(
        "Answer generation",
        [
            "Local RAG",
            "OpenAI",
            "Gemini",
        ],
        help=(
            "Local RAG works without an API key. "
            "OpenAI/Gemini require the corresponding API key."
        ),
    )

    top_k = st.slider(
        "Retrieved policy chunks",
        min_value=1,
        max_value=6,
        value=3,
    )

    chunk_size = st.slider(
        "Chunk size",
        min_value=50,
        max_value=200,
        value=100,
        step=10,
    )

    chunk_overlap = st.slider(
        "Chunk overlap",
        min_value=0,
        max_value=50,
        value=20,
        step=5,
    )

    st.divider()

    st.markdown("### 📄 Add HR Documents")

    uploaded_files = st.file_uploader(
        "Upload HR PDF or TXT files",
        type=["pdf", "txt"],
        accept_multiple_files=True,
    )

    if st.button(
        "➕ Add Uploaded Documents",
        use_container_width=True,
    ):
        if uploaded_files:
            new_documents = load_uploaded_documents(
                uploaded_files
            )

            existing_names = {
                doc["filename"]
                for doc in st.session_state.documents
            }

            added = 0

            for document in new_documents:
                if document["filename"] not in existing_names:
                    st.session_state.documents.append(document)
                    added += 1

            st.success(
                f"Added {added} new document(s)."
            )

            st.rerun()
        else:
            st.info("Please upload a PDF or TXT file first.")

    st.divider()

    if st.button(
        "🔄 Reset to Demo Policies",
        use_container_width=True,
    ):
        st.session_state.documents = [
            {"filename": name, "text": text}
            for name, text in DEMO_DOCUMENTS.items()
        ]
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True,
    ):
        st.session_state.messages = []
        st.rerun()

    st.divider()

    st.markdown(
        """
        **RAG Pipeline**

        `Question`
        ↓  
        `Embedding`
        ↓  
        `FAISS Retrieval`
        ↓  
        `Relevant HR Context`
        ↓  
        `Answer`

        **Supported documents:** PDF, TXT
        """
    )


# ============================================================
# PREPARE KNOWLEDGE BASE
# ============================================================

chunks_df = build_chunks(
    st.session_state.documents,
    chunk_size=chunk_size,
    overlap=chunk_overlap,
)

if not chunks_df.empty:
    index = build_vector_index(
        tuple(chunks_df["text"].tolist())
    )
else:
    index = None


# ============================================================
# MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="main-header">
        <h1>🤖 RAG-Based HR Assistant</h1>
        <p>
            Ask questions about HR policies and receive answers
            grounded in your organization's knowledge base.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# DASHBOARD METRICS
# ============================================================

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">HR Documents</div>
            <div class="metric-value">
                {len(st.session_state.documents)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Knowledge Chunks</div>
            <div class="metric-value">
                {len(chunks_df)}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col3:
    status = "Ready" if index is not None else "Waiting"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Vector Search</div>
            <div class="metric-value">
                {status}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Answer Mode</div>
            <div class="metric-value">
                {provider}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


st.write("")


# ============================================================
# QUICK QUESTIONS
# ============================================================

st.markdown("### 💡 Quick Questions")

quick_questions = [
    "How can I apply for annual leave?",
    "Can I work from home?",
    "What should I do if I forget to mark attendance?",
    "Where can I get my payslip?",
    "What is the resignation process?",
]

quick_cols = st.columns(len(quick_questions))

for col, question in zip(quick_cols, quick_questions):
    with col:
        if st.button(
            question,
            key=f"quick_{question}",
            use_container_width=True,
        ):
            st.session_state.quick_question = question


# ============================================================
# CHAT HISTORY
# ============================================================

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):
            with st.expander("📚 View Retrieved Sources"):
                for source in message["sources"]:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-name">
                                📄 {source["filename"]}
                            </div>
                            <div class="source-score">
                                Similarity:
                                {source["score"]:.3f}
                            </div>
                            <div style="margin-top:6px;">
                                {source["text"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )


# ============================================================
# QUESTION INPUT
# ============================================================

user_query = st.chat_input(
    "Ask an HR question, e.g. 'Can I work from home?'"
)

if not user_query:
    user_query = st.session_state.pop(
        "quick_question",
        None,
    )


if user_query:
    user_query = user_query.strip()

    if not user_query:
        st.warning("Please enter a question.")
        st.stop()

    if index is None:
        st.error(
            "The knowledge base is empty. Please add an HR document."
        )
        st.stop()

    # Save user message
    st.session_state.messages.append(
        {
            "role": "user",
            "content": user_query,
        }
    )

    with st.chat_message("user"):
        st.markdown(user_query)

    # Retrieve relevant context
    with st.chat_message("assistant"):
        with st.spinner("Searching HR knowledge base..."):
            retrieved = retrieve_documents(
                user_query,
                chunks_df,
                index,
                top_k=top_k,
            )

        with st.spinner("Preparing HR response..."):
            answer = generate_answer(
                user_query,
                retrieved,
                provider,
            )

        st.markdown(
            f"""
            <div class="answer-box">
                <strong>🤖 HR Assistant</strong>
                <div style="margin-top:10px;">
                    {answer.replace(chr(10), "<br>")}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        if retrieved:
            with st.expander("📚 View Retrieved Sources"):
                for source in retrieved:
                    st.markdown(
                        f"""
                        <div class="source-box">
                            <div class="source-name">
                                📄 {source["filename"]}
                            </div>
                            <div class="source-score">
                                Similarity:
                                {source["score"]:.3f}
                            </div>
                            <div style="margin-top:6px;">
                                {source["text"]}
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # Save assistant response and sources
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        }
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="text-align:center; color:#64748b; padding:10px;">
        <b>RAG-Based HR Assistant</b> ·
        Semantic Search with SentenceTransformers + FAISS
        <br>
        <span style="font-size:0.8rem;">
            Developed by Reem Fayyaz
        </span>
    </div>
    """,
    unsafe_allow_html=True,
)
