from flask import Flask, render_template, request, redirect, url_for, send_file, session, jsonify
from werkzeug.utils import secure_filename
import os
import hashlib
import torch
import pandas as pd
from io import BytesIO
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from dotenv import load_dotenv
from phi.agent import Agent
from phi.model.mistral import MistralChat
from pypdf import PdfReader  
from transformers import AutoTokenizer, AutoModel
import faiss
import pickle
import json
from datetime import datetime
import time
import random
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.schema import Document

load_dotenv()

app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "default_secret_key")
APP_ROOT = os.path.abspath(os.path.dirname(__file__))
"""HF Spaces runs apps inside an iframe. Default Flask cookie policy (Lax)
can drop session cookies in third-party iframes, causing session loss between
requests. These settings keep the session cookie available in the iframe."""
app.config['SESSION_COOKIE_SAMESITE'] = 'None'
app.config['SESSION_COOKIE_SECURE'] = True
app.config['UPLOAD_FOLDER'] = os.path.join(APP_ROOT, 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB limit

# Faiss Configuration
FAISS_INDEX_DIR = os.path.join(APP_ROOT, "faiss_indices")
EMBEDDING_DIM = 768  # all-mpnet-base-v2 embedding dimension

def _ensure_writable_directory(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        test_file = os.path.join(path, '.write_test')
        with open(test_file, 'w') as f:
            f.write('ok')
        os.remove(test_file)
        return path
    except Exception:
        fallback = os.path.join('/tmp', os.path.basename(path))
        os.makedirs(fallback, exist_ok=True)
        try:
            test_file = os.path.join(fallback, '.write_test')
            with open(test_file, 'w') as f:
                f.write('ok')
            os.remove(test_file)
            return fallback
        except Exception:
            return path

# Ensure writable directories exist when running under gunicorn, fallback to /tmp if needed
app.config['UPLOAD_FOLDER'] = _ensure_writable_directory(app.config['UPLOAD_FOLDER'])
FAISS_INDEX_DIR = _ensure_writable_directory(FAISS_INDEX_DIR)

# Chunking Configuration
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
CHUNKING_STRATEGY = "recursive"  # Options: "recursive", "character", "semantic"

# Initialize models once
tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-mpnet-base-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-mpnet-base-v2")

# Initialize Mistral client directly (like test.py)
from mistralai import Mistral

mistral_api_key = os.getenv("MISTRAL_API_KEY", "")
if not mistral_api_key:
    raise RuntimeError("MISTRAL_API_KEY not set. Please configure it as an environment variable.")
mistral_client = Mistral(api_key=mistral_api_key)

# API throttling configuration
API_RATE_LIMIT = {
    'requests_per_minute': 30,  # Adjust based on your API limits
    'min_delay': 2.0,  # Minimum delay between requests in seconds
    'max_delay': 5.0,  # Maximum delay between requests in seconds
    'max_retries': 3,  # Maximum number of retries
    'backoff_factor': 2.0  # Exponential backoff factor
}

# Track API usage
api_usage = {
    'last_request_time': 0,
    'request_count': 0,
    'window_start': time.time()
}

def throttle_api_request():
    """Implement API rate limiting and throttling."""
    current_time = time.time()
    
    # Reset counter if window has passed
    if current_time - api_usage['window_start'] >= 60:
        api_usage['request_count'] = 0
        api_usage['window_start'] = current_time
    
    # Check if we're within rate limits
    if api_usage['request_count'] >= API_RATE_LIMIT['requests_per_minute']:
        sleep_time = 60 - (current_time - api_usage['window_start'])
        print(f"⏳ Rate limit reached. Waiting {sleep_time:.1f} seconds...")
        time.sleep(sleep_time)
        api_usage['request_count'] = 0
        api_usage['window_start'] = time.time()
    
    # Add delay between requests
    time_since_last = current_time - api_usage['last_request_time']
    min_delay = API_RATE_LIMIT['min_delay']
    
    if time_since_last < min_delay:
        delay = min_delay - time_since_last + random.uniform(0, 1)  # Add some jitter
        print(f"⏳ Throttling: waiting {delay:.1f} seconds...")
        time.sleep(delay)
    
    api_usage['last_request_time'] = time.time()
    api_usage['request_count'] += 1

def make_api_request_with_retry(prompt, max_retries=None):
    """Make API request with retry mechanism and throttling."""
    if max_retries is None:
        max_retries = API_RATE_LIMIT['max_retries']
    
    for attempt in range(max_retries + 1):
        try:
            # Apply throttling
            throttle_api_request()
            
            print(f"🔄 Making API request (attempt {attempt + 1}/{max_retries + 1})")
            
            chat_response = mistral_client.chat.complete(
                model="mistral-large-latest",
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ]
            )
            
            print(f"✅ API request successful")
            return chat_response.choices[0].message.content.strip()
            
        except Exception as e:
            print(f"❌ API request failed (attempt {attempt + 1}): {str(e)}")
            
            if attempt < max_retries:
                # Exponential backoff with jitter
                delay = API_RATE_LIMIT['backoff_factor'] ** attempt + random.uniform(0, 1)
                print(f"⏳ Retrying in {delay:.1f} seconds...")
                time.sleep(delay)
            else:
                print(f"💥 All retry attempts failed")
                raise e

def compute_pdf_hash(pdf_path):
    """Compute MD5 hash for PDF file."""
    with open(pdf_path, "rb") as f:
        return hashlib.md5(f.read()).hexdigest()

def get_total_pages(pdf_path):
    """Get total pages and extract text using PyPDF2 for fast processing."""
    reader = PdfReader(pdf_path)
    documents = []
    for page_num, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        
        class Doc: 
            def __init__(self, content, page_number):
                self.content = content
                self.page_number = page_number
        
        doc = Doc(text, page_num)
        documents.append(doc)
    return len(documents), documents

def get_embeddings(text):
    """Generate embeddings for text."""
    tokens = tokenizer(text, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**tokens).last_hidden_state.mean(dim=1)
    return embeddings.squeeze(0).tolist()

def create_text_splitter(strategy="recursive"):
    """Create appropriate text splitter based on strategy."""
    if strategy == "recursive":
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )
    elif strategy == "character":
        return CharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separator="\n"
        )
    else:
        # Default to recursive
        return RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

def chunk_documents(documents, strategy="recursive"):
    """Chunk documents using LangChain text splitters."""
    text_splitter = create_text_splitter(strategy)
    
    chunked_docs = []
    for doc in documents:
        if doc.content.strip():  # Only process non-empty content
            # Create LangChain Document
            langchain_doc = Document(
                page_content=doc.content,
                metadata={"page_number": doc.page_number}
            )
            
            # Split the document
            chunks = text_splitter.split_documents([langchain_doc])
            
            # Convert back to our format
            for i, chunk in enumerate(chunks):
                class ChunkDoc:
                    def __init__(self, content, page_number, chunk_index):
                        self.content = content
                        self.page_number = page_number
                        self.chunk_index = chunk_index
                
                chunked_doc = ChunkDoc(
                    content=chunk.page_content,
                    page_number=chunk.metadata.get("page_number", doc.page_number),
                    chunk_index=i
                )
                chunked_docs.append(chunked_doc)
    
    return chunked_docs

def create_faiss_index_if_not_exists(index_name):
    """
    Create Faiss index and metadata file if they don't exist.
    """
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    
    index_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}.index")
    metadata_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}_metadata.pkl")
    
    if not os.path.exists(index_path):
        # Create a new Faiss index
        index = faiss.IndexFlatIP(EMBEDDING_DIM)  # Inner product for cosine similarity
        faiss.write_index(index, index_path)
        
        # Initialize metadata
        metadata = {
            'page_numbers': [],
            'pdf_hashes': [],
            'contents': [],
            'created_at': [],
            'chunk_indices': [],
            'chunk_ids': [],
            'upload_timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(metadata_path, 'wb') as f:
            pickle.dump(metadata, f)
    
    return index_path, metadata_path

def find_existing_index(pdf_hash):
    """
    Find existing index files for the same PDF hash.
    Returns the most recent index name if found, None otherwise.
    """
    if not os.path.exists(FAISS_INDEX_DIR):
        return None
    
    # Look for existing index files with the same PDF hash
    existing_files = []
    for filename in os.listdir(FAISS_INDEX_DIR):
        if filename.startswith(f"pdf_{pdf_hash[:8]}_") and filename.endswith("_metadata.pkl"):
            # Extract timestamp from filename
            try:
                timestamp_str = filename.replace(f"pdf_{pdf_hash[:8]}_", "").replace("_metadata.pkl", "")
                timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
                existing_files.append((timestamp, filename.replace("_metadata.pkl", "")))
            except:
                continue
    
    if existing_files:
        # Return the most recent one
        existing_files.sort(key=lambda x: x[0], reverse=True)
        return existing_files[0][1]
    
    return None

def page_exists_in_faiss(metadata_path, page_number):
    """Check if a specific page_number already exists in the Faiss index."""
    if not os.path.exists(metadata_path):
        return False
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    return page_number in metadata['page_numbers']

def process_pdf(pdf_path, page_start, page_end):
    """
    Processes a PDF and stores its embeddings in Faiss index
    while avoiding duplicates. Only processes pages [page_start, page_end].
    Uses LangChain chunking strategy for better text processing.
    Creates timestamped index files for each upload.
    """
    total_pages, documents = get_total_pages(pdf_path)
    pdf_hash = compute_pdf_hash(pdf_path)
    
    # Check if there's an existing index for this PDF
    existing_index = find_existing_index(pdf_hash)
    
    if existing_index:
        print(f"📁 Found existing index: {existing_index}")
        index_name = existing_index
    else:
        # Create new timestamped index name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        index_name = f"pdf_{pdf_hash[:8]}_{timestamp}"
        print(f"📁 Creating new index: {index_name}")
    
    index_path, metadata_path = create_faiss_index_if_not_exists(index_name)
    
    # Load existing index and metadata
    index = faiss.read_index(index_path)
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Get documents for the specified page range
    page_documents = documents[page_start - 1 : page_end]
    
    # Check which pages are new
    existing_pages = set(metadata.get('page_numbers', []))
    new_pages = []
    
    for doc in page_documents:
        if doc.page_number not in existing_pages:
            new_pages.append(doc)
    
    if not new_pages:
        print(f"📄 All pages {page_start}-{page_end} already processed, skipping...")
        return index_name
    
    # Chunk only the new documents using LangChain
    chunked_docs = chunk_documents(new_pages, CHUNKING_STRATEGY)
    
    print(f"📄 Processing {len(new_pages)} new pages -> {len(chunked_docs)} new chunks")
    
    # Process only new chunks
    for chunk_idx, chunk_doc in enumerate(chunked_docs):
        # Create unique identifier for this chunk
        chunk_id = f"{chunk_doc.page_number}_{chunk_doc.chunk_index}"
        
        # Skip if chunk already exists (double check)
        if chunk_id in metadata.get('chunk_ids', []):
            continue
            
        embedding = get_embeddings(chunk_doc.content)
        
        # Normalize embedding for cosine similarity
        embedding_np = torch.tensor(embedding).unsqueeze(0).numpy()
        faiss.normalize_L2(embedding_np)
        
        # Add to index
        index.add(embedding_np)
        
        # Update metadata
        metadata['page_numbers'].append(chunk_doc.page_number)
        metadata['pdf_hashes'].append(pdf_hash)
        metadata['contents'].append(chunk_doc.content)
        metadata['created_at'].append(pd.Timestamp.now())
        metadata['chunk_indices'].append(chunk_doc.chunk_index)
        
        # Add chunk_id to metadata if not exists
        if 'chunk_ids' not in metadata:
            metadata['chunk_ids'] = []
        metadata['chunk_ids'].append(chunk_id)
    
    # Save updated index and metadata
    faiss.write_index(index, index_path)
    with open(metadata_path, 'wb') as f:
        pickle.dump(metadata, f)
    
    print(f"✅ Processed {len(chunked_docs)} new chunks and stored in timestamped Faiss index")
    return index_name

def generate_topics(index_name, num_topics=10):
    """
    Analyze the content in Faiss index and return a concise list of distinct main topics (title-like),
    ignoring lines like "Based on the provided content..." or any extra fluff.
    These topics are displayed for the user but not used for question generation unless overridden.
    """
    metadata_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}_metadata.pkl")
    
    if not os.path.exists(metadata_path):
        return []
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Combine all content
    content = " ".join(metadata['contents'])[:5000]

    
    prompt = f"""
    You have the following PDF content (truncated to 5000 characters):
    ---
    {content}
    ---
    Identify up to {num_topics} DISTINCT key topics or concepts covered in this text.

    Requirements for each topic:
      - Must be a short, descriptive phrase or title (not a sentence).
      - Must be strictly derived from the PDF content (no guessing or external knowledge).
      - Do NOT prefix with "Based on the provided content..." or any filler text.
      - Return only bullet points (like "- Communication Fundamentals").
      - No duplicates or repeated phrases.
      - Keep them concise and relevant.

    Do NOT add extra commentary or summary beyond the bullet points.
    """

    # Use retry mechanism with throttling
    try:
        raw_output = make_api_request_with_retry(prompt)
    except Exception as e:
        print(f"Error generating topics: {e}")
        return ["Sample Topic 1", "Sample Topic 2", "Sample Topic 3"]

    # Split lines by newline
    lines = raw_output.split('\n')

    topics = []
    for line in lines:
        line = line.strip()
        # Remove any leading bullet-like symbols
        if line.startswith('- '):
            line = line[2:].strip()
        elif line.startswith('* '):
            line = line[2:].strip()
        # If line has a phrase we want to exclude
        if "Based on the provided content" in line:
            continue
        # Remove leftover markdown formatting
        line = line.replace('**', '').replace('###', '').strip()
        # Skip empty lines or filler
        if not line:
            continue
        topics.append(line)

    # Truncate to num_topics if too many
    topics = topics[:num_topics]
    return topics


def generate_questions(index_name, topics, question_types, question_counts):
    """
    topics: user-chosen topics only
    question_types: e.g. ["mcqs", "short_qa", "descriptive"]
    question_counts: e.g. 
      {
        'mcqs':        {'Easy': 2, 'Medium': 2, 'Hard': 2},
        'short_qa':    {'Easy': 2, 'Medium': 2, 'Hard': 2},
        'descriptive': {'Easy': 2, 'Medium': 2, 'Hard': 2}
      }
    """
    # 1) Pull text from Faiss metadata
    metadata_path = os.path.join(FAISS_INDEX_DIR, f"{index_name}_metadata.pkl")
    
    if not os.path.exists(metadata_path):
        return {}
    
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)
    
    # Combine and truncate
    doc_text = " ".join(metadata['contents'])[:5000]

    # 2) Prepare final results
    results = {
        "mcqs": {},
        "short_qa": {},
        "descriptive": {}
    }

    # 3) Loop over user-chosen topics, question types, difficulties
    for topic in topics:
        for q_type in question_types:
            results[q_type][topic] = {}
            for difficulty, count in question_counts[q_type].items():
                if count <= 0:
                    continue

                # 4) Build prompt with "use only words from doc" directive
                if q_type == "mcqs":
                    prompt = f"""
                    Generate {count} {difficulty} MCQs about the topic: "{topic}"
                    Use ONLY the exact wording from the document below. 
                    Do not add any outside knowledge or invented text.

                    Document content (truncated):
                    {doc_text}

                    Requirements for each MCQ:
                    - 4 options (A-D)
                    - Mark the correct answer clearly
                    - Label difficulty
                    - Format:
                      Q) ...
                      A) ...
                      B) ...
                      C) ...
                      D) ...
                      Answer: ...
                    """
                elif q_type == "short_qa":
                    prompt = f"""
                    Create {count} {difficulty} short-answer questions about the topic: "{topic}"
                    Use ONLY the exact wording from the document below. 
                    Do not add any outside knowledge or invented text.

                    Document content (truncated):
                    {doc_text}

                    Requirements:
                    - A 'Q)' line
                    - An 'Answer:' line with concise text 
                    - Label difficulty
                    """
                elif q_type == "descriptive":
                    prompt = f"""
                    Develop {count} {difficulty} descriptive questions about the topic: "{topic}"
                    Use ONLY the exact wording from the document below. 
                    Do not add any outside knowledge or invented text.

                    Document content (truncated):
                    {doc_text}

                    Requirements:
                    - A 'Q)' line
                    - An 'Expected Answer:' line with a detailed explanation
                    - Label difficulty
                    """

                # 5) Call the LLM using retry mechanism with throttling
                try:
                    results[q_type][topic][difficulty] = make_api_request_with_retry(prompt)
                except Exception as e:
                    print(f"Error generating questions for {topic} - {q_type} - {difficulty}: {e}")
                    results[q_type][topic][difficulty] = f"Error generating {difficulty} {q_type} questions for {topic}"

    return results


# --------------------------------------------------------------------
# Flask Routes
# --------------------------------------------------------------------
@app.route('/')
def index():
    """Home page."""
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload_pdf():
    if request.method == 'POST':
        if 'pdf' not in request.files:
            return redirect(url_for('upload_pdf'))
        file = request.files['pdf']
        if file and file.filename.endswith('.pdf'):
            filename = secure_filename(file.filename)
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            total_pages, _ = get_total_pages(save_path)
            session.clear()
            session['pdf_path'] = save_path
            session['total_pages'] = total_pages
            return redirect(url_for('select_pages'))
    return render_template('upload.html')

@app.route('/configure-chunking', methods=['GET', 'POST'])
def configure_chunking():
    """Configure chunking strategy and parameters."""
    global CHUNK_SIZE, CHUNK_OVERLAP, CHUNKING_STRATEGY
    
    if request.method == 'POST':
        CHUNK_SIZE = int(request.form.get('chunk_size', 1000))
        CHUNK_OVERLAP = int(request.form.get('chunk_overlap', 200))
        CHUNKING_STRATEGY = request.form.get('chunking_strategy', 'recursive')
        
        return redirect(url_for('upload_pdf'))
    
    return render_template('configure_chunking.html', 
                         chunk_size=CHUNK_SIZE,
                         chunk_overlap=CHUNK_OVERLAP,
                         chunking_strategy=CHUNKING_STRATEGY)

@app.route('/select-pages', methods=['GET', 'POST'])
def select_pages():
    total_pages = session.get('total_pages', 1)
    if request.method == 'POST':
        page_start = int(request.form['start'])
        page_end = int(request.form['end'])
        if page_start > page_end:
            page_start, page_end = page_end, page_start
        if page_start < 1:
            page_start = 1
        if page_start > total_pages:
            page_start = total_pages
        if page_end < page_start:
            page_end = page_start
        if page_end > total_pages:
            page_end = total_pages
        if (page_end - page_start + 1) > 30:
            page_end = page_start + 30 - 1
            if page_end > total_pages:
                page_end = total_pages
        session['page_start'] = page_start
        session['page_end'] = page_end
        return redirect(url_for('process_pdf_route'))
    return render_template('select_pages.html', total_pages=total_pages)

@app.route('/process-pdf')
def process_pdf_route():
    if 'pdf_path' not in session:
        return redirect(url_for('upload_pdf'))
    try:
        table_name = process_pdf(
            session.get('pdf_path'),
            session.get('page_start'),
            session.get('page_end')
        )
        session['index_name'] = table_name
        return redirect(url_for('manage_topics'))
    except Exception as e:
        return str(e), 500

@app.route('/manage-topics', methods=['GET', 'POST'])
def manage_topics():
    if 'index_name' not in session:
        return redirect(url_for('upload_pdf'))

    if request.method == 'POST':
        # Parse user-entered topics
        raw_input = request.form.get('manual_topics', '')
        manual_topics = [t.strip() for t in raw_input.split(',') if t.strip()]

        # Parse selected auto topics joined by hidden input (and support individual checkbox list as fallback)
        selected_joined = request.form.get('selected_auto_topics_joined', '')
        auto_selected = [t.strip() for t in selected_joined.split(',') if t.strip()]
        if not auto_selected:
            auto_selected = request.form.getlist('selected_auto_topics')

        # Merge and de-duplicate while preserving order
        merged = []
        for t in (auto_selected + manual_topics):
            if t and t not in merged:
                merged.append(t)

        # Store topics for question generation
        session['topics'] = merged
        return redirect(url_for('select_questions'))

    # On GET:
    # 1) If we haven't auto-detected topics yet, do it once
    if 'auto_topics' not in session:
        session['auto_topics'] = generate_topics(session['index_name'], num_topics=10)

    # 2) If no user topics yet, default to empty
    if 'topics' not in session:
        session['topics'] = []

    # Pass auto-detected to template for display only
    return render_template('manage_topics.html',
                           auto_topics=session['auto_topics'])


@app.route('/select-questions', methods=['GET', 'POST'])
def select_questions():
    if request.method == 'POST':
        q_types = request.form.getlist('q_types')
        question_counts = {
            'mcqs': {
                'Easy': int(request.form.get('mcqs_easy', 0)),
                'Medium': int(request.form.get('mcqs_medium', 0)),
                'Hard': int(request.form.get('mcqs_hard', 0))
            },
            'short_qa': {
                'Easy': int(request.form.get('short_qa_easy', 0)),
                'Medium': int(request.form.get('short_qa_medium', 0)),
                'Hard': int(request.form.get('short_qa_hard', 0))
            },
            'descriptive': {
                'Easy': int(request.form.get('descriptive_easy', 0)),
                'Medium': int(request.form.get('descriptive_medium', 0)),
                'Hard': int(request.form.get('descriptive_hard', 0))
            }
        }
        session['question_types'] = q_types
        session['question_counts'] = question_counts
        return redirect(url_for('show_results'))
    return render_template('select_questions.html')

@app.route('/results')
def show_results():
    if 'index_name' not in session:
        return redirect(url_for('upload_pdf'))
    results = generate_questions(
        session['index_name'],
        session.get('topics', []),
        session.get('question_types', []),
        session.get('question_counts', {})
    )
    session['results'] = results
    return render_template('results.html', results=results)

@app.route('/download/<format>')
def download(format):
    results = session.get('results', {})
    if not results:
        return "No results to download", 400

    # Only include selected question types with non-empty content
    selected_types = session.get('question_types', []) or []
    def has_content(d):
        return bool(d) and any(bool(v) for v in d.values())
    if selected_types:
        filtered = {qt: data for qt, data in results.items() if qt in selected_types and has_content(data)}
    else:
        filtered = {qt: data for qt, data in results.items() if has_content(data)}

    if not filtered:
        return "No content to export for the selected options", 400

    if format == 'csv':
        import csv
        import io
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["QuestionType", "Topic", "Difficulty", "Content"])
        for q_type, topic_data in filtered.items():
            for topic, diff_data in topic_data.items():
                for level, content in (diff_data or {}).items():
                    if not content:
                        continue
                    writer.writerow([q_type, topic, level, content])
        output.seek(0)
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/csv',
            as_attachment=True,
            download_name='questions.csv'
        )
    elif format == 'pdf':
        # Better looking PDF using Platypus
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.enums import TA_LEFT
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=50, rightMargin=50,
                                topMargin=50, bottomMargin=50)

        styles = getSampleStyleSheet()
        # Custom styles
        title = ParagraphStyle('title', parent=styles['Heading1'], fontSize=16, spaceAfter=10)
        h2 = ParagraphStyle('h2', parent=styles['Heading2'], textColor=colors.HexColor('#374151'), spaceBefore=12, spaceAfter=6)
        h3 = ParagraphStyle('h3', parent=styles['Heading3'], textColor=colors.HexColor('#4B5563'), spaceBefore=6, spaceAfter=4)
        body = ParagraphStyle('body', parent=styles['BodyText'], leading=14, fontSize=10)
        mono = ParagraphStyle('mono', parent=body, fontName='Helvetica', leftIndent=8)

        flow = []
        flow.append(Paragraph('AI Generated Questions', title))

        def normalize(text: str) -> str:
            tx = (text or '').replace('**', '').replace('###', '').replace('---', '')
            tx = tx.replace('\n', '<br/>')
            return tx

        # Prepare an ordered list of types to render
        ordered_types = [qt for qt in ['mcqs', 'short_qa', 'descriptive'] if qt in filtered]
        for idx, q_type in enumerate(ordered_types):
            topic_data = filtered[q_type]
            if not has_content(topic_data):
                continue
            if idx > 0:
                # Page break only between non-empty sections
                flow.append(PageBreak())
            flow.append(Spacer(1, 8))
            flow.append(Paragraph(f'Question Type: <b>{q_type.replace("_", " ").title()}</b>', h2))
            for topic, diff_data in (topic_data or {}).items():
                if not diff_data:
                    continue
                flow.append(Paragraph(f'Topic: <b>{topic}</b>', h3))
                for level, content in (diff_data or {}).items():
                    if not content:
                        continue
                    flow.append(Paragraph(f'Difficulty: <b>{level}</b>', styles['Italic']))
                    flow.append(Spacer(1, 4))
                    flow.append(Paragraph(normalize(content), mono))
                    flow.append(Spacer(1, 10))
                # Divider
                tbl = Table([[" "]], colWidths=[460])
                tbl.setStyle(TableStyle([
                    ('LINEBELOW', (0,0), (-1,-1), 0.25, colors.HexColor('#E5E7EB')),
                ]))
                flow.append(tbl)
                flow.append(Spacer(1, 10))

        doc.build(flow)
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name='questions.pdf'
        )
    return "Invalid format", 400

@app.route('/download_saved_pdf', methods=['POST'])
def download_saved_pdf():
    try:
        payload = request.get_json(silent=True) or {}
        items = payload.get('items', [])
        include_answers = bool(payload.get('include_answers', True))
        if not isinstance(items, list) or not items:
            return "No saved questions provided", 400

        # Build a polished PDF
        import io
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib import colors
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter,
                                leftMargin=50, rightMargin=50,
                                topMargin=50, bottomMargin=50)

        styles = getSampleStyleSheet()
        title = ParagraphStyle('title', parent=styles['Heading1'], fontSize=16, spaceAfter=12)
        h2 = ParagraphStyle('h2', parent=styles['Heading2'], textColor=colors.HexColor('#111827'), spaceBefore=10, spaceAfter=4)
        meta = ParagraphStyle('meta', parent=styles['BodyText'], textColor=colors.HexColor('#6B7280'), fontSize=9, spaceAfter=4)
        qstyle = ParagraphStyle('qstyle', parent=styles['BodyText'], leading=14, fontSize=10)
        optstyle = ParagraphStyle('optstyle', parent=qstyle, leftIndent=12)
        ansstyle = ParagraphStyle('ansstyle', parent=qstyle, textColor=colors.HexColor('#065F46'))

        def clean_lines(text: str, keep_answer: bool) -> list:
            # Normalize and split; remove Difficulty lines
            text = (text or '').replace('\r\n', '\n')
            lines = [l.strip() for l in text.split('\n')]
            cleaned = []
            for l in lines:
                if not l:
                    continue
                if l.lower().startswith('difficulty:'):
                    continue
                if not keep_answer and l.lower().startswith('answer:'):
                    continue
                cleaned.append(l)
            return cleaned

        flow = [Paragraph('Saved Questions', title)]

        # Group by topic for nicer layout
        from collections import defaultdict
        grouped = defaultdict(list)
        for it in items:
            grouped[(it.get('topic') or 'Untitled', it.get('type') or '')].append(it)

        for (topic, qtype), arr in grouped.items():
            flow.append(Paragraph(f'Topic: <b>{topic}</b>', h2))
            if qtype:
                flow.append(Paragraph(f'Type: {qtype.replace("_"," ").title()}', meta))

            for it in arr:
                lines = clean_lines(it.get('content', ''), include_answers)
                if not lines:
                    continue
                # Render question + options/answer
                # Detect first line that starts with 'Q' as question; otherwise print all
                for ln in lines:
                    if ln.lower().startswith('q'):
                        flow.append(Paragraph(ln, qstyle))
                    elif ln[:2] in ('A)', 'B)', 'C)', 'D)') or (len(ln) > 2 and ln[1:3] == ')'):
                        flow.append(Paragraph(ln, optstyle))
                    elif ln.lower().startswith('answer:'):
                        flow.append(Paragraph(ln, ansstyle))
                    else:
                        flow.append(Paragraph(ln, qstyle))
                # divider
                tbl = Table([[" "]], colWidths=[460])
                tbl.setStyle(TableStyle([('LINEBELOW', (0,0), (-1,-1), 0.25, colors.HexColor('#E5E7EB'))]))
                flow.append(Spacer(1, 6))
                flow.append(tbl)
                flow.append(Spacer(1, 8))

        doc.build(flow)
        buffer.seek(0)
        return send_file(buffer, mimetype='application/pdf', as_attachment=True,
                         download_name='saved_questions.pdf')
    except Exception as e:
        return str(e), 500

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(FAISS_INDEX_DIR, exist_ok=True)
    port = int(os.getenv("PORT", "7860"))
    app.run(host="0.0.0.0", port=port)
