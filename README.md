# QnABuilderAI

Generate high‑quality questions (MCQs, short Q&A, descriptive) directly from uploaded PDFs. QnABuilderAI chunks PDF pages, embeds them with sentence‑transformers, indexes content in FAISS, detects topics, and uses an LLM (Mistral) via an agentic flow to synthesize questions. Results can be downloaded as CSV or a polished PDF.

## ✨ Features
- Upload a PDF and restrict processing to a page range
- Configurable chunking strategy and size/overlap
- Embeddings with `all-mpnet-base-v2` and FAISS inner‑product retrieval
- Topic auto‑detection (with retry + throttling safeguards)
- Question generation by type and difficulty using Mistral
- Export results to CSV or a styled PDF
- Hugging Face Spaces friendly (Flask app served in an iframe)

## 🧱 Tech Stack
- Backend: Flask, Python 3, Werkzeug
- AI/Agents: `mistralai`, `phidata`
- Embeddings/RAG: `transformers`, `sentence-transformers`, `faiss-cpu`, `langchain` splitters
- PDF: `pypdf`
- Export: `reportlab`

## 📂 Project Structure
```
QnABuilderAI/
  app.py
  requirements.txt
  templates/
    index.html
    upload.html
    configure_chunking.html
    select_pages.html
    manage_topics.html
    select_questions.html
    results.html
  uploads/
  faiss_indices/
```

## 🔐 Environment Variables
Create a `.env` file next to `app.py` with:
```
SECRET_KEY=change_this_in_production
MISTRAL_API_KEY=your_mistral_key
PORT=7860
```

## 🚀 Run Locally
1) Create and activate a virtual environment, then install deps:
```
pip install -r requirements.txt
```
2) Start the app:
```
python app.py
```
3) Open the app at `http://localhost:7860`.

## ▶️ Run on Hugging Face Spaces
This repository is compatible with Spaces (Docker). See the Spaces config reference: `https://huggingface.co/docs/hub/spaces-config-reference`.

## 🧭 User Flow
1) Upload PDF → 2) Configure chunking (optional) → 3) Select page range → 4) Process PDF (embeds + FAISS) → 5) Review auto‑detected topics and/or add your own → 6) Choose question types and counts per difficulty → 7) Generate → 8) Download CSV/PDF.

## 🔌 Key Endpoints (Flask)
- `/` Home
- `/upload` Upload a PDF
- `/configure-chunking` Adjust chunk size/overlap and strategy
- `/select-pages` Pick start/end pages (capped to 30 pages per run)
- `/process-pdf` Build/extend FAISS index for the selection
- `/manage-topics` Auto‑detect topics; add/curate topics
- `/select-questions` Choose types and counts by difficulty
- `/results` Generate questions and display
- `/download/<format>` Export CSV or PDF (`format` = `csv` | `pdf`)

## 🧠 How It Works (High‑Level)
1. Parse PDF pages with `pypdf` and split text using LangChain text splitters (configurable).
2. Create embeddings via `sentence-transformers/all-mpnet-base-v2` and store normalized vectors in FAISS (inner product).
3. Concatenate content for topic detection and prompt the LLM (Mistral) with concise instructions; throttle and retry to avoid API limits.
4. For each selected topic, generate requested question types and difficulties using structured prompts.
5. Render results in the UI and enable export (CSV or a styled PDF via ReportLab).

## ⚙️ Configuration
Update defaults in `app.py`:
- `CHUNK_SIZE`, `CHUNK_OVERLAP`, `CHUNKING_STRATEGY`
- API rate limits in `API_RATE_LIMIT`
- Storage directories: `uploads/`, `faiss_indices/` (auto‑created and write‑tested; falls back to `/tmp` if needed on Spaces)

## 📎 Notes & Limits
- Page selection is limited to 30 pages per run to keep API and compute usage reasonable.
- If Mistral API fails after retries, topic or question generation gracefully degrades and logs an error.
- The app requires `MISTRAL_API_KEY` to be set; it will raise at startup if missing.

## 🧪 Troubleshooting
- Missing key: ensure `.env` includes `MISTRAL_API_KEY` and the process can read it.
- CUDA/torch warnings: the app uses CPU for FAISS and embeddings by default; ensure versions in `requirements.txt` are installed successfully.
- Session issues on Spaces: cookies are configured for iframe usage (`SameSite=None`, `Secure=True`).

## 📸 Application Screenshots

Here's a visual walkthrough of the QnABuilderAI application:

### 1. QnABuilder Landing Page
![QnABuilder Landing Page](snapshots/1.QnABuilder%20Landing%20page.png)
*Welcome screen with features overview and navigation options*

### 2. PDF Upload Interface
![PDF Upload Interface](snapshots/2.PDF%20Upload%20Interface.png)
*Drag & drop file upload with progress tracking*

### 3. Configure Chunking Settings
![Configure Chunking Settings](snapshots/3.Configure%20Chunking%20Settings.png)
*Customize text processing parameters and preview*

### 4. Select Pages to Process
![Select Pages to Process](snapshots/4.Select%20Pages%20to%20Process.png)
*Choose specific page ranges from uploaded PDF*

### 5. Manage Topics
![Manage Topics](snapshots/5.Manage%20Topics.png)
*Review AI-detected topics and add custom topics*

### 6. Select Question Types and Difficulty
![Select Question Types and Difficulty](snapshots/6.Select%20Question%20Types%20and%20Difficulty.png)
*Choose question formats and difficulty levels*

### 7. Generated Questions Results View
![Generated Questions Results View](snapshots/7.Generated%20Questions%20Results%20View.png)
*Display AI-generated questions organized by type and topic*

### 8. Export Options and Download Interface
![Export Options and Download Interface](snapshots/8.Export%20Options%20and%20Download%20Interface.png)
*Download questions in CSV or PDF formats*

## 👨‍💻 Author

**Aradhya Pavan**



