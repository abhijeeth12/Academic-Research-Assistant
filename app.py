from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from dotenv import load_dotenv
import os
from serpapi import GoogleSearch
import requests
import PyPDF2
import io
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import re
import json
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer
# Add these new imports at the top
import pdfplumber
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer
from PIL import Image
import logging
import pdfplumber
import re
import io
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
from sklearn.feature_extraction.text import TfidfVectorizer 
from markupsafe import Markup
import pdfplumber
import torch
from transformers import LayoutLMv3ForTokenClassification, LayoutLMv3Tokenizer

# Suppress pypdf CropBox warnings
logging.getLogger("pypdf").setLevel(logging.ERROR)

load_dotenv()
app = Flask(__name__)
CORS(app)

# Configuration
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
layoutlm_model = LayoutLMv3ForTokenClassification.from_pretrained("microsoft/layoutlmv3-base")
layoutlm_tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
layoutlm_model = layoutlm_model.to('cuda' if torch.cuda.is_available() else 'cpu')
CHUNK_SIZE = 1000
TOP_K = 5
OLLAMA_API = "http://localhost:11434/api"
SIMILARITY_THRESHOLD = 0.7
MAX_SCRAPE_LINKS = 5  # Maximum PDF links to scrape from a single page
HEADING_SIMILARITY_THRESHOLD = 0.65
MIN_HEADING_LENGTH = 3
# Initialize models
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def clean_json_response(text):
    """Clean and extract JSON from LLama response"""
    text = re.sub(r'```(?:json)?\n?([\s\S]*?)```', r'\1', text)
    json_match = re.search(r'\[\s*\{[^\[\]]*\}\s*\]|\{[^{}]*\}', text)
    if json_match:
        return json_match.group(0)
    return None

def query_llama(prompt, model="llama3", expect_json=False):
    """Query LLama model with better error handling and JSON support"""
    # Improved print to show query position
    print(f"\nQuerying LLama with prompt:")
    print(f"First 150 chars: {prompt[:150]}")
    print(f"Last 150 chars: {prompt[-150:]}" if len(prompt) > 300 else "")
    # Rest of the function remains the same
    
    try:
        full_response = ""
        response = requests.post(
            f"{OLLAMA_API}/generate",
            json={
                "model": model,
                "prompt": prompt + (" Return only a valid JSON array." if expect_json else ""),
                "stream": False,
                "options": {
                    "temperature": 0.3,
                    "stop": ["\n", "\n\n"] if expect_json else None
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        result = response.json().get("response", "").strip()
        print(f"Raw response: {result}")
        
        if expect_json:
            try:
                matches = re.findall(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', result)
                if matches:
                    for match in matches:
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
                return json.loads(result)
            except Exception as e:
                print(f"JSON parsing failed: {e}")
                return [0, 1, 2]
        if expect_json:
            try:
                return json.loads(result)
            except json.JSONDecodeError:
                print("JSON parsing failed, returning empty dict")
                return {}
        return result
    except Exception as e:
        print(f"LLama query failed: {e}")
        if expect_json:
            return [0, 1, 2]
        return None

def extract_keywords(query):
    """Extract main keywords from query using LLM"""
    prompt = f"""Extract or generate the most important 1-3 keywords from this query that represent its core meaning. 
    Return ONLY a JSON array of strings with the keywords.
    
    Example: "What is a covalent bond?" → ["covalent bond"]
    Example: "Explain machine learning algorithms" → ["machine learning", "algorithms"]
    
    Query: {query}"""
    
    keywords = query_llama(prompt, expect_json=True)
    if isinstance(keywords, list) and len(keywords) > 0:
        return keywords
    # Fallback to simple extraction if LLM fails
    return [query.replace("?", "").strip().lower()]

def find_similar_words(keywords):
    """Find semantically similar words using embeddings"""
    similar_words = []
    for keyword in keywords:
        keyword_embed = embedding_model.encode([keyword])[0]
        
        vocab = [
            keyword,
            keyword.replace(" ", "-"),
            keyword.replace(" ", "_"),
            keyword + "s",
            "introduction to " + keyword,
            "fundamentals of " + keyword,
            "basics of " + keyword,
            "what is " + keyword,
            keyword + " definition",
            keyword + " concept",
            keyword + " pdf",
            keyword + " paper",
            keyword + " article",
            keyword + " research"
        ]
        
        vocab_embeds = embedding_model.encode(vocab)
        similarities = cosine_similarity([keyword_embed], vocab_embeds)[0]
        
        for word, sim in zip(vocab, similarities):
            if sim >= SIMILARITY_THRESHOLD and word not in similar_words:
                similar_words.append(word)
    
    return list(set(similar_words))

def is_pdf_url(url):
    """Check if URL directly points to a PDF"""
    if not url:
        return False
    return url.lower().endswith('.pdf')

def scrape_for_pdf_links(url):
    """Scrape a webpage to find PDF links"""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link['href']
            if href.lower().endswith('.pdf'):
                if not href.startswith('http'):
                    href = urljoin(url, href)
                pdf_links.append(href)
                if len(pdf_links) >= MAX_SCRAPE_LINKS:
                    break
        
        return pdf_links
    except Exception as e:
        print(f"Error scraping {url}: {e}")
        return []
# Add heading extraction function
def extract_headings_and_content(pdf_url=None, raw_text=None, heading_pattern=r"^[A-Z][A-Za-z0-9 ]{1,50}$"):
    """
    Extracts headings and their content from a PDF using pdfplumber.
    
    Args:
        pdf_url (str): URL of the PDF to download and process.
        raw_text (str): Fallback raw text to process if pdf_url is None.
        heading_pattern (str): Regex for identifying heading lines.
    
    Returns:
        dict: Mapping of heading -> content (string).
    """
    chunks = {}
    current_heading = None
    current_content = []

    if pdf_url:
        # Download PDF bytes
        response = requests.get(pdf_url, timeout=20)
        pdf_stream = io.BytesIO(response.content)
        pdf = pdfplumber.open(pdf_stream)
    else:
        # Simulate a single-page pdfplumber object from raw_text
        class FakePage: 
            def __init__(self, text): self._text = text
            @property
            def extract_text(self): return lambda: self._text
        pdf = [FakePage(raw_text or "")]  # one-page list

    for page in (pdf.pages if hasattr(pdf, "pages") else pdf):
        text = page.extract_text() or ""
        for line in text.split("\n"):
            line_stripped = line.strip()
            if re.match(heading_pattern, line_stripped):
                # Save previous heading
                if current_heading:
                    chunks[current_heading] = "\n".join(current_content).strip()
                current_heading = line_stripped
                current_content = []
            else:
                current_content.append(line)
    # Append the last heading
    if current_heading:
        chunks[current_heading] = "\n".join(current_content).strip()
    return chunks


def extract_metadata_from_url(url):
    """Try to extract metadata from the URL/page before downloading"""
    print(f"Attempting to extract metadata from URL: {url}")
    try:
        if "arxiv.org" in url:
            arxiv_id = url.split('/')[-1].replace('.pdf', '')
            metadata = requests.get(f"http://export.arxiv.org/api/query?id_list={arxiv_id}").text
            title = re.search(r'<title>([^<]+)</title>', metadata).group(1)
            authors = re.findall(r'<author>.*?<name>(.*?)</name>', metadata)
            author = ", ".join(authors) if authors else "Unknown"
            print(f"Extracted arxiv metadata - Title: {title}, Authors: {author}")
            return {"title": title, "author": author}
        
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type.lower():
            head = requests.get(url, headers={'Range': 'bytes=0-1000'}).content
            if b'%PDF-' in head:
                print("Found PDF metadata in headers")
                return {"title": url.split('/')[-1].replace('.pdf', '').replace('_', ' ').title(), "author": ""}
        
        print("No metadata found in URL, will try content extraction")
        return None
    except Exception as e:
        print(f"Metadata extraction error: {e}")
        return None

def extract_text_from_pdf(url):
    """Extract text from PDF with error handling"""
    try:
        response = requests.get(url, timeout=20)
        with io.BytesIO(response.content) as f:
            reader = PyPDF2.PdfReader(f)
            return " ".join(page.extract_text() for page in reader.pages if page.extract_text())
    except Exception as e:
        print(f"PDF extraction failed: {e}")
        return ""

def extract_metadata_from_text(text):
    """Use LLM to extract title/author from document text"""
    print("Extracting metadata from document text using LLama")
    prompt = f"""Analyze this text and extract:
    1. The document's title
    2. The author(s)
    3. The publication year if available
    
    Return ONLY a JSON object with 'title', 'author', and 'year' keys.
    
    Text: {text[:7500]}"""
    
    metadata = query_llama(prompt, expect_json=True)
    if metadata:
        print(f"Extracted metadata: {metadata}")
        return metadata
    return {"title": "", "author": "", "year": ""}

def chunk_text(text, chunk_size=1000):
    """Create context-preserving chunks with headings"""
    heading_content = extract_headings_and_content(text)
    chunks = []
    
    # Fallback if no headings found
    if not heading_content:
        print("No headings found, using standard chunking")
        sections = re.split(r'\n\s*\n', text)
        current_chunk = ""
        for section in sections:
            if len(current_chunk) + len(section) < chunk_size:
                current_chunk += "\n\n" + section
            else:
                chunks.append(current_chunk.strip())
                current_chunk = section
        if current_chunk:
            chunks.append(current_chunk.strip())
        return chunks
    
    for heading, content in heading_content.items():
        chunk = f"## {heading}\n{content}"
        while len(chunk) > chunk_size:
            split_pos = max(
                chunk.rfind('\n\n', 0, chunk_size),
                chunk.rfind('. ', 0, chunk_size)
            )
            if split_pos == -1:
                split_pos = chunk_size
            chunks.append(chunk[:split_pos].strip())
            chunk = f"## {heading}\n" + chunk[split_pos:].lstrip()
        chunks.append(chunk.strip())
    return chunks

def get_relevant_chunks(query, chunks):
    """Get most relevant chunks with similarity scores"""
    if not chunks:  # Handle empty chunks case
        print("⚠️ Warning: No chunks available for similarity calculation")
        return []
    
    # Ensure proper array dimensions
    query_embed = embedding_model.encode([query])
    chunk_embeds = embedding_model.encode(chunks)
    
    # Reshape arrays for sklearn compatibility
    if len(query_embed.shape) == 1:
        query_embed = query_embed.reshape(1, -1)
    if len(chunk_embeds.shape) == 1:
        chunk_embeds = chunk_embeds.reshape(1, -1)
    
    try:
        sims = cosine_similarity(query_embed, chunk_embeds)[0]
        top_indices = np.argsort(sims)[-TOP_K:][::-1]
        return [(chunks[i], sims[i]) for i in top_indices]
    except ValueError as e:
        print(f"⚠️ Similarity calculation failed: {str(e)}")
        return []

def generate_summary(query, chunks):
    """Generate comprehensive summary using LLM"""
    print(f"Generating summary for query: {query} using {len(chunks)} chunks")
    
    chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = chunks[:5]
    
    context = ""
    for i, (chunk, score) in enumerate(top_chunks):
        context += f"Chunk {i+1} (score {score:.2f}): {chunk[:1000]}\n"
    
    prompt = f"Generate a comprehensive and detailed summary of the following text to answer '{query}'. Include key points, important details, and relevant examples. TEXT: {context}"
    
    print("Querying LLama for summary...")
    summary = query_llama(prompt)
    
    if not summary:
        return "Sorry, I couldn't generate a summary. Please try again."
    
    summary = summary.replace('```', '').strip()
    summary = re.sub(r'\n{3,}', '\n\n', summary)
    
    print("Summary generated successfully")
    return summary

def preprocess_summary(summary_text: str, title: str, source_url: str) -> str:
    """
    Turn raw summary_text into the HTML structure you showed:
      - <h2>Analysis Results</h2>
      - <h3>{title}</h3>
      - first **…** block as bold question
      - remaining text as paragraphs and <ul>/<li>
      - View Source Document link
    """
    # 1. Escape the LLM output
    summary = Markup.escape(summary_text)

    # 2. Extract first **…** question
    m = re.match(r'\s*\*\*(?P<q>[^*]+)\*\*\s*(?P<rest>.+)', summary, flags=re.S)
    if m:
        question, rest = m.group('q'), m.group('rest')
    else:
        question, rest = None, summary

    # 3. Convert remaining **…** → <strong>…</strong>
    rest = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', rest)

    # 4. Wrap "- " lines in <ul>
    lines = rest.splitlines()
    html_body, in_ul = [], False
    for line in lines:
        if line.strip().startswith('- '):
            if not in_ul:
                html_body.append('<ul>')
                in_ul = True
            html_body.append(f'  <li>{line.strip()[2:].strip()}</li>')
        else:
            if in_ul:
                html_body.append('</ul>')
                in_ul = False
            if line.strip():
                html_body.append(f'<p>{line.strip()}</p>')
    if in_ul:
        html_body.append('</ul>')

    # 5. Assemble final HTML
    parts = [
        '<div class="analysis-results">',
        '  <h2>Analysis Results</h2>',
    ]
    if title:
        parts.append(f'  <h3>{Markup.escape(title)}</h3>')
    if question:
        parts.append(f'  <p><strong>{Markup.escape(question)}</strong></p>')
    parts += [f'  {ln}' for ln in html_body]
    parts.append(f'  <p><a href="{Markup.escape(source_url)}" target="_blank">View Source Document</a></p>')
    parts.append('</div>')

    return '\n'.join(parts)

def rank_documents(query, documents):
    """Rank documents using semantic similarity with embeddings"""
    return documents
        
    print(f"\nRanking {len(documents)} documents for query: {query}")
    
    try:
        doc_texts = []
        for doc in documents:
            doc_text = f"{doc['title']} "
            if doc.get('year'):
                doc_text += f"({doc['year']}) "
            if doc['author'] != "Unknown":
                doc_text += f"by {doc['author']} "
            doc_text += f"[{doc['type']}]"
            doc_texts.append(doc_text)
        
        query_embedding = embedding_model.encode([query])[0]
        doc_embeddings = embedding_model.encode(doc_texts)
        
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        ranked_pairs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        ranked_indices = [i for i, _ in ranked_pairs]
        
        print(f"Ranked order with scores:")
        for idx, (doc_idx, score) in enumerate(ranked_pairs):
            print(f"{idx+1}. {documents[doc_idx]['title']} (score: {score:.3f})")
        
        return [documents[i] for i in ranked_indices]
        
    except Exception as e:
        print(f"Error in embedding-based ranking: {e}")
        return documents

def search_documents_with_keywords(query, selected_types=["PDF"]):
    """Search documents using keyword-based approach"""
    keywords = extract_keywords(query)
    print(f"Extracted keywords: {keywords}")
    
    similar_words = find_similar_words(keywords)
    print(f"Similar words: {similar_words}")
    
    search_terms = keywords
    unique_terms = list(set(search_terms))
    
    all_results = []
    
    for term in unique_terms:
        search_query = f"{term} PDF"
        print(f"Search query: {search_query}")
        try:
            search = GoogleSearch({
                "q": search_query, 
                "api_key": SERP_API_KEY, 
                "num": 10
            })
            results = search.get_dict().get("organic_results", [])
            
            for result in results:
                url = result.get('link')
                if not url:
                    continue
                
                if is_pdf_url(url):
                    all_results.append({
                        'url': url,
                        'title': result.get('title', url),
                        'source': 'direct_pdf',
                        'search_term': term
                    })
                else:
                    pdf_links = scrape_for_pdf_links(url)
                    for pdf_url in pdf_links:
                        all_results.append({
                            'url': pdf_url,
                            'title': result.get('title', pdf_url) or pdf_url.split('/')[-1],
                            'source': 'scraped_pdf',
                            'search_term': term,
                            'origin_url': url
                        })
            
        except Exception as e:
            print(f"Search failed for term '{term}': {e}")
            continue
    
    unique_results = {}
    for result in all_results:
        url = result['url']
        if url not in unique_results or result['source'] == 'direct_pdf':
            unique_results[url] = result
    
    return list(unique_results.values())

@app.route('/')
def index():
    return render_template('index.html')

@app.route("/search", methods=["POST"])
def search_documents():
    data = request.get_json()
    query = data.get("query", "")
    selected_types = data.get("types", ["PDF"])
    
    print(f"\n=== Starting Document Search ===\nQuery: {query}\nTypes: {selected_types}")
    
    try:
        raw_documents = search_documents_with_keywords(query, selected_types)
        print(f"Found {len(raw_documents)} initial results")
        
        documents = []
        for doc in raw_documents:
            try:
                metadata = extract_metadata_from_url(doc['url']) or {}
                
                if not metadata.get("title"):
                    text = extract_text_from_pdf(doc['url'])
                    if text:
                        metadata.update(extract_metadata_from_text(text))
                
                documents.append({
                    "url": doc['url'],
                    "title": metadata.get("title", doc['url'].split('/')[-1]),
                    "author": metadata.get("author", "Unknown"),
                    "year": metadata.get("year", ""),
                    "type": "PDF",
                    "source": doc.get('source', 'unknown'),
                    "origin_url": doc.get('origin_url', ''),
                    "search_term": doc.get('search_term', '')
                })
            except Exception as e:
                print(f"Failed to process document {doc['url']}: {e}")
                continue
        
        ranked_documents = rank_documents(query, documents)
        
        return jsonify({"documents": ranked_documents})
        
    except Exception as e:
        print(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/analyze", methods=["POST"])
def analyze_document():
    # Validate request
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400

    try:
        data = request.get_json()
    except Exception as e:
        return jsonify({"error": "Invalid JSON format"}), 400

    # Check required fields
    if not data or 'url' not in data:
        return jsonify({"error": "Missing required field 'url'"}), 400

    url = data.get('url', '').strip()
    query = data.get('query', '').strip()
    if not url:
        return jsonify({"error": "URL cannot be empty"}), 400

    print(f"\nStarting analysis for URL: {url}")
    print(f"User query: {query}")

    try:
        # 1. Download and extract text
        print("Downloading document...")
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            return jsonify({"error": f"Failed to download document: {str(e)}"}), 400

        # 2. Extract text from PDF
        print("Extracting text from PDF...")
        try:
            text = extract_text_from_pdf(url)
            if not text:
                return jsonify({"error": "No text content found in document"}), 400
        except Exception as e:
            return jsonify({"error": f"PDF text extraction failed: {str(e)}"}), 400

        # 3. Extract document structure
        print("Extracting document structure...")
        pdf_content = response.content  # Get the binary content
        text = extract_text_from_pdf(url)  # Keep text extraction
        heading_content = extract_headings_and_content(pdf_url=url, raw_text=text)
        headings = list(heading_content.keys())
        # 4. Keyword extraction (original prompt preserved)
        prompt = f"""Analyze this query and extract key terms that should appear in the answer.
Return ONLY a comma-separated list of 5-7 keywords/phrases.

Query: {query}"""
        
        keyword_string = query_llama(prompt)
        keywords = [k.strip() for k in keyword_string.split(",")] if keyword_string else []
        search_terms = f"{query} {' '.join(keywords)}"
        
        # 5. Try heading-based retrieval first
        relevant_excerpts = []
        if headings:
            print("Attempting heading-based retrieval...")
            heading_embeddings = embedding_model.encode(headings)
            search_embedding = embedding_model.encode([search_terms])
            
            similarities = cosine_similarity(search_embedding, heading_embeddings)[0]
            heading_scores = sorted(zip(headings, similarities), 
                                  key=lambda x: x[1], reverse=True)
            
            # Get headings above threshold
            relevant_headings = [h for h, s in heading_scores 
                               if s > HEADING_SIMILARITY_THRESHOLD][:3]
            
            if relevant_headings:
                print(f"Found relevant headings: {relevant_headings}")
                for heading in relevant_headings:
                    content = heading_content[heading]
                    relevant_excerpts.append(f"- {content[:300]}")

        # 6. Fallback to chunk-based retrieval if no good headings
        if not relevant_excerpts:
            print("Using chunk-based retrieval fallback...")
            chunks = chunk_text(text, pdf_url=url)
            relevant_chunks = get_relevant_chunks(search_terms, chunks)
            relevant_excerpts = [f"- {chunk[0][:300]}" for chunk in relevant_chunks]

        # 7. Generate summary (original prompt preserved)
        summary_prompt = f"""USER QUERY: {query}

RELEVANT EXCERPTS:
{chr(10).join(relevant_excerpts)}

properly format excerpts length of 100-200 words if possible and skip any introduction part, your response is directly going to be displayed on my website"""

        summary = query_llama(summary_prompt)
        if not summary:
            return jsonify({"error": "Failed to generate summary"}), 500

        # Clean and format summary
        summary = summary.replace('```', '').strip()
        summary = re.sub(r'\n{3,}', '\n\n', summary)

        # 8. Extract metadata
        metadata = {"title": "", "author": "", "year": ""}
        try:
            metadata.update(extract_metadata_from_text(text[:5000]))
        except Exception as e:
            print(f"Metadata extraction warning: {str(e)}")
                 # … same as before …

        # Clean and format summary
        summary = summary.replace('```', '').strip()
        summary = re.sub(r'\n{3,}', '\n\n', summary)
        # Clean raw markdown
        summary = summary.replace('```', '').strip()
        summary = re.sub(r'\n{3,}', '\n\n', summary)

     # 8. Extract metadata
        metadata = {"title": "", "author": "", "year": ""}
        try:
            metadata.update(extract_metadata_from_text(text[:5000]))
        except Exception as e:
            print(f"Metadata extraction warning: {str(e)}")

        # 9. Transform the cleaned summary into your HTML layout
        formatted_html = preprocess_summary(
            summary_text=summary,
            title=metadata.get("title", ""),
            source_url=url
        )

        return jsonify({
            "success": True,
            "summary": summary,
            "metadata": metadata,
            "query": query,
            "stats": {
                "total_chunks": len(heading_content) if relevant_excerpts else len(chunks),
                "relevant_chunks": len(relevant_excerpts),
                "top_relevance": float(max(similarities)) if headings else float(relevant_chunks[0][1]) if relevant_chunks else 0
            }
        })

    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Unexpected error: {str(e)}\n{error_trace}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
