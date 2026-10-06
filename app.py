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
        :root {
            --bg-soft: #f4fbf5;
            --bg-ice: #edfdf4;
            --green-50: #f0fdf4;
            --green-100: #dcfce7;
            --green-200: #bbf7d0;
            --green-300: #86efac;
            --green-500: #22c55e;
            --green-600: #16a34a;
            --green-700: #15803d;
            --green-800: #166534;
            --green-900: #14532d;
            --slate-700: #334155;
            --slate-800: #1e293b;
            --slate-500: #64748b;
            --white: #ffffff;
            --shadow-soft: 0 16px 40px rgba(22, 163, 74, 0.12);
        }

        .stApp {
            background: linear-gradient(135deg, #f5fff8 0%, #ecfdf5 35%, #f3f9ff 100%);
        }

        .main-header {
            padding: 1.6rem 1.7rem;
            border-radius: 22px;
            background: linear-gradient(135deg, #0d3b2f 0%, #0f6b45 35%, #4ade80 100%);
            color: white;
            margin-bottom: 1.3rem;
            box-shadow: 0 18px 36px rgba(21, 128, 61, 0.22);
            border: 1px solid rgba(255,255,255,0.1);
            position: relative;
            overflow: hidden;
        }

        .main-header::after {
            content: "";
            position: absolute;
            inset: 0;
            background: radial-gradient(circle at top right, rgba(255,255,255,0.18), transparent 32%);
            pointer-events: none;
        }

        .main-header h1 {
            margin: 0;
            font-size: 2.2rem;
            font-weight: 900;
            letter-spacing: -0.03em;
            position: relative;
            z-index: 1;
        }

        .main-header p {
            margin: 0.5rem 0 0;
            opacity: 0.96;
            font-size: 1.02rem;
            position: relative;
            z-index: 1;
        }

        .pulse-card {
            background: rgba(255,255,255,0.12);
            border: 1px solid rgba(255,255,255,0.2);
            border-radius: 14px;
            padding: 0.7rem 0.9rem;
            color: white;
            backdrop-filter: blur(4px);
            display: inline-flex;
            align-items: center;
            gap: 0.55rem;
            margin-top: 0.8rem;
            position: relative;
            z-index: 1;
            font-weight: 700;
        }

        .pulse-dot {
            width: 10px;
            height: 10px;
            border-radius: 50%;
            background: #bbf7d0;
            box-shadow: 0 0 0 0 rgba(187,247,208,0.65);
            animation: pulse 2s infinite;
        }

        @keyframes pulse {
            0% { box-shadow: 0 0 0 0 rgba(187,247,208,0.7); }
            70% { box-shadow: 0 0 0 12px rgba(187,247,208,0); }
            100% { box-shadow: 0 0 0 0 rgba(187,247,208,0); }
        }

        .metric-card {
            background: linear-gradient(180deg, #ffffff 0%, #f6fff9 100%);
            border-radius: 18px;
            padding: 1rem 1.1rem;
            border: 1px solid #d9fbe8;
            box-shadow: 0 12px 28px rgba(15, 118, 110, 0.07);
            min-height: 110px;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
            position: relative;
            overflow: hidden;
        }

        .metric-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 18px 32px rgba(21, 128, 61, 0.12);
        }

        .metric-card::before {
            content: "";
            position: absolute;
            top: 0;
            right: 0;
            width: 72px;
            height: 72px;
            background: radial-gradient(circle, rgba(74,222,128,0.16), transparent 62%);
        }

        .metric-title {
            color: #4b7a5a;
            font-size: 0.8rem;
            font-weight: 700;
            letter-spacing: 0.02em;
            text-transform: uppercase;
            position: relative;
            z-index: 1;
        }

        .metric-value {
            color: #0f172a;
            font-size: 1.6rem;
            font-weight: 900;
            margin-top: 0.45rem;
            position: relative;
            z-index: 1;
        }

        .answer-box {
            background: linear-gradient(180deg, #ffffff 0%, #f4fff7 100%);
            border-left: 6px solid #22c55e;
            border-radius: 16px;
            padding: 1.2rem 1.3rem;
            margin: 0.8rem 0 1rem;
            box-shadow: 0 12px 26px rgba(22, 163, 74, 0.08);
            border-top: 1px solid #ddf7e4;
            border-right: 1px solid #ddf7e4;
            border-bottom: 1px solid #ddf7e4;
        }

        .source-box {
            background: linear-gradient(180deg, #ffffff 0%, #f8fdf9 100%);
            border: 1px solid #d9f5df;
            border-left: 5px solid #16a34a;
            border-radius: 14px;
            padding: 0.9rem 1rem;
            margin: 0.6rem 0;
            box-shadow: 0 8px 18px rgba(22, 163, 74, 0.04);
        }

        .source-name {
            color: #166534;
            font-weight: 800;
        }

        .source-score {
            color: #5b7a64;
            font-size: 0.8rem;
            font-weight: 700;
        }

        .small-note {
            color: #4b7a5a;
            font-size: 0.83rem;
        }

        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #f7fff9 0%, #eefaf4 100%);
        }

        .stButton > button {
            border-radius: 12px;
            font-weight: 700;
            border: 1px solid #a7f3d0;
            background: linear-gradient(180deg, #ffffff 0%, #f3fff8 100%);
            color: #14532d;
            transition: all 0.2s ease;
            box-shadow: 0 8px 18px rgba(34, 197, 94, 0.08);
        }

        .stButton > button:hover {
            border-color: #4ade80;
            background: linear-gradient(180deg, #effdf5 0%, #dcfce7 100%);
            transform: translateY(-1px);
            box-shadow: 0 12px 24px rgba(34, 197, 94, 0.12);
        }

        .stButton > button:focus {
            box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.2);
        }

        section[data-testid="stSidebarContent"] .stMarkdown h3,
        section[data-testid="stSidebarContent"] .stMarkdown h2 {
            color: #14532d;
        }

        .policy-badge {
            display: inline-flex;
            align-items: center;
            padding: 0.38rem 0.8rem;
            border-radius: 999px;
            background: #dcfce7;
            color: #166534;
            border: 1px solid #bbf7d0;
            font-size: 0.76rem;
            font-weight: 800;
            letter-spacing: 0.02em;
        }

        @media (max-width: 768px) {
            .main-header h1 {
                font-size: 1.7rem;
            }
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
        <div class="pulse-card">
            <span class="pulse-dot"></span>
            HR intelligence is active and ready
        </div>
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
            <div class="metric-value">{len(st.session_state.documents)}</div>
            <div class="small-note">Loaded policy files</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Knowledge Chunks</div>
            <div class="metric-value">{len(chunks_df)}</div>
            <div class="small-note">Semantic search units</div>
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
            <div class="metric-value">{status}</div>
            <div class="small-note">FAISS retrieval</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-title">Answer Mode</div>
            <div class="metric-value">{provider}</div>
            <div class="small-note">Powered by policy context</div>
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
                            <div class="source-name">📄 {source["filename"]}</div>
                            <div class="source-score">Similarity: {source["score"]:.3f}</div>
                            <div style="margin-top:6px;">{source["text"]}</div>
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
                <div class="policy-badge">🤖 HR Assistant</div>
                <div style="margin-top:12px;">{answer.replace(chr(10), "<br>")}</div>
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
                            <div class="source-name">📄 {source["filename"]}</div>
                            <div class="source-score">Similarity: {source["score"]:.3f}</div>
                            <div style="margin-top:6px;">{source["text"]}</div>
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
    <div style="text-align:center; color:#3b5d4a; padding:10px;">
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

"""}]}