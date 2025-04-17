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

load_dotenv()
app = Flask(__name__)
CORS(app)

# Configuration
SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000
TOP_K = 5
OLLAMA_API = "http://localhost:11434/api"
SIMILARITY_THRESHOLD = 0.7
MAX_SCRAPE_LINKS = 5  # Maximum PDF links to scrape from a single page

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
    print(f"\nQuerying LLama with prompt:\n{prompt[:200]}...")
    
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
        return result
    except Exception as e:
        print(f"LLama query failed: {e}")
        if expect_json:
            return [0, 1, 2]
        return None

def extract_keywords(query):
    """Extract main keywords from query using LLM"""
    prompt = f"""Extract the most important 1-3 keywords from this query that represent its core meaning. 
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

def chunk_text(text):
    """Create larger semantic chunks with overlap"""
    print("Creating text chunks...")
    sections = re.split(r'\n\s*\n|(?=[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}:)|(?=\d+\.\s+[A-Z])', text)
    sections = [s.strip() for s in sections if s.strip()]
    
    chunks = []
    current_chunk = ""
    overlap_size = CHUNK_SIZE // 4
    
    for section in sections:
        if len(current_chunk) + len(section) < CHUNK_SIZE:
            current_chunk += "\n\n" + section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            if current_chunk and len(current_chunk) > overlap_size:
                current_chunk = current_chunk[-overlap_size:] + "\n\n" + section
            else:
                current_chunk = section
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    print(f"Created {len(chunks)} chunks with average size {sum(len(c) for c in chunks)/len(chunks) if chunks else 0:.0f} chars")
    return chunks

def get_relevant_chunks(query, chunks):
    """Get most relevant chunks with similarity scores"""
    query_embed = embedding_model.encode([query])
    chunk_embeds = embedding_model.encode(chunks)
    sims = cosine_similarity(query_embed, chunk_embeds)[0]
    top_indices = np.argsort(sims)[-TOP_K:][::-1]
    return [(chunks[i], sims[i]) for i in top_indices]

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

def rank_documents(query, documents):
    """Rank documents using semantic similarity with embeddings"""
    if len(documents) <= 1:
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
    
    search_terms = keywords + similar_words
    unique_terms = list(set(search_terms))
    
    all_results = []
    
    for term in unique_terms:
        search_query = f"{term} PDF"
        
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
            with io.BytesIO(response.content) as f:
                reader = PyPDF2.PdfReader(f)
                text = " ".join(page.extract_text() for page in reader.pages if page.extract_text())
        except Exception as e:
            return jsonify({"error": f"PDF text extraction failed: {str(e)}"}), 400

        if not text:
            return jsonify({"error": "No text content found in document"}), 400

        # 3. Process text chunks
        print("Processing text chunks...")
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "Failed to create text chunks"}), 400

        # 4. Find relevant chunks
        relevant_chunks = get_relevant_chunks(query, chunks)
        if not relevant_chunks:
            return jsonify({"error": "No relevant content found for the query"}), 404

        # 5. Generate summary
        print("Generating summary...")
        chunks_context = "\n\n".join(
            f"CHUNK {i+1} (Relevance: {score:.2f}):\n{chunk[:1000]}"
            for i, (chunk, score) in enumerate(relevant_chunks)
        )

        prompt = f"""USER QUERY: {query}

DOCUMENT EXCERPTS:
{chunks_context}

INSTRUCTIONS:
1. Focus on directly answering: "{query}"
2. Use only the provided excerpts
3. Include key points and supporting details
4. Maintain original meaning accurately
5. Use clear, academic language

SUMMARY:"""

        summary = query_llama(prompt)
        if not summary:
            return jsonify({"error": "Failed to generate summary"}), 500

        # Clean summary
        summary = summary.replace('```', '').strip()
        summary = re.sub(r'\n{3,}', '\n\n', summary)

        # 6. Extract metadata
        metadata = {"title": "", "author": "", "year": ""}
        try:
            metadata.update(extract_metadata_from_text(text[:5000]))  # Only use first part for metadata
        except Exception as e:
            print(f"Metadata extraction warning: {e}")

        return jsonify({
            "success": True,
            "summary": summary,
            "metadata": metadata,
            "query": query,
            "stats": {
                "total_chunks": len(chunks),
                "relevant_chunks": len(relevant_chunks),
                "top_relevance": float(relevant_chunks[0][1]) if relevant_chunks else 0
            }
        })

    except Exception as e:
        print(f"Unexpected error: {str(e)}", exc_info=True)
        return jsonify({"error": "Internal server error"}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)
