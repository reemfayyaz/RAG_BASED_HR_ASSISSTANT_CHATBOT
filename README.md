# 🤖 RAG-Based HR Assistant

A professional **Retrieval-Augmented Generation (RAG) HR Assistant** built with Python and Streamlit.

The application allows employees to ask questions about HR policies and retrieve relevant information from a knowledge base. It uses **SentenceTransformers** to create semantic embeddings and **FAISS** for fast similarity-based document retrieval.

The application can work in three modes:

- **Local RAG** — works without an external API key
- **OpenAI** — uses OpenAI for natural-language answer generation
- **Gemini** — uses Google Gemini for natural-language answer generation

---

## ✨ Features

- 🤖 AI-powered HR question answering
- 🔎 Semantic document search
- ⚡ FAISS vector similarity search
- 🧠 SentenceTransformer embeddings
- 📄 PDF and TXT document upload
- 📚 Displays retrieved sources
- 📊 Similarity scores for retrieved chunks
- 💬 Chat-style conversation interface
- 💡 Quick HR question buttons
- 🔄 Reset knowledge base
- 🗑️ Clear chat
- 🎨 Professional Streamlit interface
- 🔐 Optional OpenAI API integration
- 🔐 Optional Google Gemini API integration
- 🏠 Built-in HR policy demo documents
- ☁️ Suitable for GitHub and Streamlit deployment

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| Streamlit | Web application interface |
| SentenceTransformers | Text embeddings |
| FAISS | Vector similarity search |
| NumPy | Numerical processing |
| Pandas | Data handling |
| PyPDF | PDF text extraction |
| OpenAI | Optional LLM answer generation |
| Google Gemini | Optional LLM answer generation |

---

## 📁 Project Structure

```text
RAG-Based-HR-Assistant/
│
├── app.py
├── requirements.txt
├── README.md
│
└── optional/
    └── HR policy documents
```

The application also contains several built-in demonstration HR policies, so you can run it immediately without uploading documents.

---

# 🚀 Installation

## 1. Clone or download the project

Download the project files:

```text
app.py
requirements.txt
README.md
```

If the project is hosted on GitHub:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd RAG-Based-HR-Assistant
```

---

## 2. Create a virtual environment

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

The application will open in your browser.

Usually Streamlit runs at:

```text
http://localhost:8501
```

---

# 🧠 How the RAG System Works

The application follows this workflow:

```text
User Question
      ↓
SentenceTransformer Embedding
      ↓
FAISS Similarity Search
      ↓
Retrieve Relevant HR Chunks
      ↓
HR Policy Context
      ↓
Answer Generation
      ↓
Response + Sources
```

### Step 1 — Document Loading

The application loads HR information from:

- Built-in HR policies
- PDF files
- TXT files

### Step 2 — Text Chunking

Large documents are divided into smaller overlapping chunks.

This makes semantic search more accurate.

### Step 3 — Embeddings

Each text chunk is converted into a numerical vector using:

```text
all-MiniLM-L6-v2
```

### Step 4 — FAISS Index

The embeddings are stored in a FAISS vector index.

FAISS makes it possible to quickly find the chunks most similar to a user's question.

### Step 5 — Retrieval

When the user asks a question, the question is also converted into an embedding.

The system searches FAISS and retrieves the most relevant HR policy chunks.

### Step 6 — Answer

The application can then:

- Return a retrieval-based answer locally, or
- Send the retrieved context to OpenAI, or
- Send the retrieved context to Google Gemini.

The application is designed to avoid inventing HR policies when the requested information is not available in the knowledge base.

---

# 📄 Uploading HR Documents

From the Streamlit sidebar:

1. Click **Upload HR PDF or TXT files**
2. Select your HR documents
3. Click **Add Uploaded Documents**
4. Ask questions about the uploaded policies

Supported formats:

```text
.pdf
.txt
```

Examples:

```text
Leave Policy.pdf
Attendance Policy.pdf
Work From Home Policy.pdf
Employee Handbook.pdf
Payroll Policy.pdf
Resignation Policy.pdf
```

---

# 🤖 Answer Generation Modes

## 1. Local RAG

This mode does not require an API key.

It uses:

```text
SentenceTransformers
+
FAISS
```

to find relevant information from the HR knowledge base.

Select:

```text
Answer generation → Local RAG
```

This is the easiest mode for testing and demonstrations.

---

# 🔵 OpenAI Mode

To use OpenAI, create an API key and configure it as an environment variable.

### Windows PowerShell

```powershell
$env:OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

### Windows CMD

```cmd
set OPENAI_API_KEY=YOUR_OPENAI_API_KEY
```

### macOS / Linux

```bash
export OPENAI_API_KEY="YOUR_OPENAI_API_KEY"
```

Then start the application:

```bash
streamlit run app.py
```

Inside the application select:

```text
Answer generation → OpenAI
```

### Important

Never publish your real API key in:

- GitHub
- `app.py`
- `README.md`
- screenshots
- public repositories

---

# 🟣 Google Gemini Mode

To use Google Gemini, configure a Gemini API key.

### Windows PowerShell

```powershell
$env:GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

### Windows CMD

```cmd
set GEMINI_API_KEY=YOUR_GEMINI_API_KEY
```

### macOS / Linux

```bash
export GEMINI_API_KEY="YOUR_GEMINI_API_KEY"
```

Then run:

```bash
streamlit run app.py
```

Select:

```text
Answer generation → Gemini
```

---

# 🔐 Using a `.env` File

For local development, you can use a `.env` file if you want to manage environment variables separately.

Example:

```text
OPENAI_API_KEY=your_openai_key
GEMINI_API_KEY=your_gemini_key
```

Do not upload the `.env` file to GitHub.

Add this to `.gitignore`:

```text
.env
venv/
__pycache__/
*.pyc
```

---

# 💡 Example Questions

You can ask questions such as:

```text
How can I apply for annual leave?
```

```text
Can I work from home?
```

```text
What should I do if I forget to mark attendance?
```

```text
Where can I get my payslip?
```

```text
What is the resignation process?
```

```text
What is the notice period?
```

The answer depends on the information available in the HR knowledge base.

---

# 📊 Application Dashboard

The application displays useful system information:

### HR Documents

Number of HR documents currently loaded.

### Knowledge Chunks

Number of text chunks created from the HR documents.

### Vector Search

Shows whether the FAISS search index is ready.

### Answer Mode

Displays the currently selected answer-generation method.

---

# ⚙️ Advanced Settings

The sidebar includes controls for:

### Retrieved Policy Chunks

Controls how many relevant chunks are retrieved.

Default:

```text
3
```

### Chunk Size

Controls the approximate number of words in each document chunk.

### Chunk Overlap

Controls how much text overlaps between consecutive chunks.

These settings can be adjusted depending on the size and structure of your HR documents.

---

# 📚 Built-in Demo Policies

The application includes demonstration policies for:

- Leave
- Work From Home
- Attendance
- Payroll
- Resignation

This means the project can be demonstrated immediately after installation.

---

# 🔒 Security Notes

This application is designed as an HR information assistant, so sensitive employee information should be handled carefully.

Recommended practices:

- Do not hard-code API keys.
- Do not upload confidential HR documents to public repositories.
- Use appropriate access controls for production deployments.
- Do not expose employee personal information unnecessarily.
- Review generated answers before using the application for official HR decisions.
- Keep company HR policies updated.

The application should be treated as an **information assistant**, not as a replacement for official HR approval or company policy.

---

# ☁️ Streamlit Deployment

The project can be deployed to Streamlit hosting or another Python-compatible hosting platform.

Typical deployment files:

```text
app.py
requirements.txt
README.md
```

For platforms that support secrets, store API keys as secrets rather than putting them directly in the source code.

Example secret names:

```text
OPENAI_API_KEY
GEMINI_API_KEY
```

---

# 🧪 Testing Without an API Key

You do not need OpenAI or Gemini to test the application.

Install the requirements:

```bash
pip install -r requirements.txt
```

Run:

```bash
streamlit run app.py
```

Select:

```text
Local RAG
```

Then ask:

```text
How can I apply for annual leave?
```

The application will retrieve information from its local HR knowledge base.

---

# 🐛 Troubleshooting

## FAISS installation problem

Try:

```bash
pip install --upgrade pip
pip install faiss-cpu
```

Then reinstall the remaining packages:

```bash
pip install -r requirements.txt
```

## Streamlit command not found

Run:

```bash
python -m streamlit run app.py
```

## PDF is not returning useful answers

Make sure the PDF contains selectable text.

Scanned image-only PDFs may require OCR before their contents can be retrieved effectively.

## OpenAI or Gemini is not responding

Check that:

1. The API key is configured.
2. The correct provider is selected.
3. The API account has access to the required model/service.
4. Your internet connection is available.

The application falls back to local retrieval when an external LLM cannot be reached.

---

# 🎯 Project Objective

The main objective of this project is to demonstrate how **Retrieval-Augmented Generation (RAG)** can be applied to an HR knowledge system.

It combines:

```text
Natural Language Questions
        +
Semantic Embeddings
        +
Vector Database/Search
        +
Relevant Context Retrieval
        +
LLM Answer Generation
```

This approach helps produce answers grounded in the organization's available HR documents.

---

# 🌟 Future Improvements

Possible future enhancements include:

- Employee login and authentication
- Admin dashboard
- Database integration
- Conversation memory
- DOCX support
- Excel/CSV HR data integration
- OCR for scanned PDFs
- Department-specific access
- Employee-specific answers
- HR ticket creation
- Email integration
- Advanced analytics
- Cloud vector databases
- Document version management
- Source citations in generated answers

---

# 👩‍💻 Author

**Reem Fayyaz**

RAG-Based HR Assistant  
Python • Streamlit • FAISS • SentenceTransformers • Generative AI

---

# 📜 License

This project is intended for educational, portfolio, and demonstration purposes.

Review and adapt the code according to your organization's security, privacy, and HR requirements before using it with real employee data.
