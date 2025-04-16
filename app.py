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

load_dotenv()
app = Flask(__name__)
CORS(app)

# Configuration
DOCUMENT_TYPES = {
    "PDF": "filetype:pdf",
    "PPT": "filetype:ppt OR filetype:pptx",
    "Word": "filetype:doc OR filetype:docx"
}

SERP_API_KEY = os.getenv("SERPAPI_API_KEY")
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 1000  # Larger chunks for better context
TOP_K = 5  # More chunks for richer summary
OLLAMA_API = "http://localhost:11434/api"

# Initialize model
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def clean_json_response(text):
    """Clean and extract JSON from LLama response"""
    # Remove markdown code blocks
    text = re.sub(r'```(?:json)?\n?([\s\S]*?)```', r'\1', text)
    
    # Try to find JSON array or object
    json_match = re.search(r'\[\s*\{[^\[\]]*\}\s*\]|\{[^{}]*\}', text)
    if json_match:
        return json_match.group(0)
    return None

def query_llama(prompt, model="llama3", expect_json=False):
    """Query LLama model with better error handling and JSON support"""
    print(f"\nQuerying LLama with prompt:\n{prompt[:200]}...")
    
    try:
        # Use streaming to get complete response
        full_response = ""
        response = requests.post(
            f"{OLLAMA_API}/generate",
            json={
                "model": model,
                "prompt": prompt + (" Return only a valid JSON array." if expect_json else ""),
                "stream": False,  # Disable streaming for complete response
                "options": {
                    "temperature": 0.3,
                    "stop": ["\n", "\n\n"] if expect_json else None  # Stop at newlines for JSON
                }
            },
            timeout=120
        )
        response.raise_for_status()
        
        # Get response text
        result = response.json().get("response", "").strip()
        print(f"Raw response: {result}")
        
        if expect_json:
            try:
                # Try to find a JSON array in the response
                matches = re.findall(r'\[\s*\d+\s*,\s*\d+\s*,\s*\d+\s*\]', result)
                if matches:
                    # Take the first valid JSON array found
                    for match in matches:
                        try:
                            return json.loads(match)
                        except json.JSONDecodeError:
                            continue
                
                # If no array found, try parsing the whole response
                return json.loads(result)
            except Exception as e:
                print(f"JSON parsing failed: {e}")
                # Return a default ranking if parsing fails
                return [0, 1, 2]
        return result
    except Exception as e:
        print(f"LLama query failed: {e}")
        if expect_json:
            return [0, 1, 2]  # Return default ranking
        return None

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
        
        # Try to get metadata from page headers
        response = requests.head(url, allow_redirects=True)
        content_type = response.headers.get('content-type', '')
        if 'pdf' in content_type.lower():
            # Get PDF metadata without downloading full file
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
    Be thorough in finding author names and full title.
    If any field is unknown, use empty string.
    
    Example response format:
    {{
        "title": "Global Warming: A Comprehensive Study",
        "author": "John Smith, Jane Doe",
        "year": "2023"
    }}
    
    Text: {text[:7500]}"""
    
    metadata = query_llama(prompt, expect_json=True)
    if metadata:
        print(f"Extracted metadata: {metadata}")
        return metadata
    return {"title": "", "author": "", "year": ""}

def chunk_text(text):
    """Create larger semantic chunks with overlap"""
    print("Creating text chunks...")
    # First split into sections/paragraphs
    sections = re.split(r'\n\s*\n|(?=[A-Z][a-z]{2,}\s+[A-Z][a-z]{2,}:)|(?=\d+\.\s+[A-Z])', text)
    sections = [s.strip() for s in sections if s.strip()]
    
    chunks = []
    current_chunk = ""
    overlap_size = CHUNK_SIZE // 4  # 25% overlap for context continuity
    
    for section in sections:
        if len(current_chunk) + len(section) < CHUNK_SIZE:
            current_chunk += "\n\n" + section
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            # Keep some overlap from previous chunk
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
    
    # Sort chunks by relevance score and take top 5
    chunks.sort(key=lambda x: x[1], reverse=True)
    top_chunks = chunks[:5]
    
    # Create a simple context string
    context = ""
    for i, (chunk, score) in enumerate(top_chunks):
        # Take first 1000 chars of each chunk for more context
        context += f"Chunk {i+1} (score {score:.2f}): {chunk[:1000]}\n"
    
    # Detailed summary prompt
    prompt = f"Generate a comprehensive and detailed summary of the following text to answer '{query}'. Include key points, important details, and relevant examples. Aim for a thorough analysis while maintaining clarity. TEXT: {context}"
    
    print("Querying LLama for summary...")
    summary = query_llama(prompt)
    
    if not summary:
        return "Sorry, I couldn't generate a summary. Please try again."
    
    # Clean up the summary
    summary = summary.replace('```', '').strip()
    summary = re.sub(r'\n{3,}', '\n\n', summary)  # Remove excess newlines
    
    print("Summary generated successfully")
    return summary

@app.route('/')
def index():
    return render_template('index.html')

def rank_documents(query, documents):
    """Rank documents using semantic similarity with embeddings"""
    if len(documents) <= 1:
        return documents
        
    print(f"\nRanking {len(documents)} documents for query: {query}")
    
    try:
        # Create document representations that combine title and metadata
        doc_texts = []
        for doc in documents:
            # Combine title with metadata for better matching
            doc_text = f"{doc['title']} "
            if doc.get('year'):
                doc_text += f"({doc['year']}) "
            if doc['author'] != "Unknown":
                doc_text += f"by {doc['author']} "
            doc_text += f"[{doc['type']}]"
            doc_texts.append(doc_text)
        
        # Get embeddings for query and documents
        query_embedding = embedding_model.encode([query])[0]
        doc_embeddings = embedding_model.encode(doc_texts)
        
        # Calculate similarities
        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        # Create (index, similarity) pairs and sort by similarity
        ranked_pairs = sorted(enumerate(similarities), key=lambda x: x[1], reverse=True)
        ranked_indices = [i for i, _ in ranked_pairs]
        
        print(f"Ranked order with scores:")
        for idx, (doc_idx, score) in enumerate(ranked_pairs):
            print(f"{idx+1}. {documents[doc_idx]['title']} (score: {score:.3f})")
        
        # Return documents in ranked order
        return [documents[i] for i in ranked_indices]
        
    except Exception as e:
        print(f"Error in embedding-based ranking: {e}")
        return documents

@app.route("/search", methods=["POST"])
def search_documents():
    data = request.get_json()
    query = data.get("query", "")
    selected_types = data.get("types", ["PDF"])  # Default to PDF
    
    print(f"\n=== Starting Document Search ===\nQuery: {query}\nTypes: {selected_types}")
    
    # Build search query
    type_filters = [DOCUMENT_TYPES[t] for t in selected_types]
    search_query = f"{query} ({' OR '.join(type_filters)})"
    
    # Get search results
    try:
        search = GoogleSearch({"q": search_query, "api_key": SERP_API_KEY, "num": 15})
        urls = [r.get('link') for r in search.get_dict().get("organic_results", [])]
        print(f"Found {len(urls)} initial results")
    except Exception as e:
        print(f"Search failed: {e}")
        return jsonify({"error": str(e)}), 500
    
    # Get metadata for each document
    documents = []
    for url in urls:
        try:
            # Try to get metadata from URL first
            metadata = extract_metadata_from_url(url) or {}
            
            # If no metadata from URL, extract text and use LLM
            if not metadata.get("title"):
                text = extract_text_from_pdf(url)
                if text:
                    metadata.update(extract_metadata_from_text(text))
            
            documents.append({
                "url": url,
                "title": metadata.get("title", url.split('/')[-1]),
                "author": metadata.get("author", "Unknown"),
                "year": metadata.get("year", ""),
                "type": next((t for t in DOCUMENT_TYPES if url.lower().endswith(tuple(t.lower().split()))), "Other")
            })
        except Exception as e:
            print(f"Failed to process document {url}: {e}")
            continue
    
    # Rank documents using LLama
    ranked_documents = rank_documents(query, documents)
    
    return jsonify({"documents": ranked_documents})

@app.route("/analyze", methods=["POST"])
def analyze_document():
    print("\n=== Starting Document Analysis ===\n")
    data = request.get_json()
    url = data.get("url")
    query = data.get("query", "")
    print(f"Analyzing document: {url}\nUser query: {query}")
    
    try:
        # Extract full text
        text = extract_text_from_pdf(url)
        if not text:
            return jsonify({"error": "Failed to extract text from document"}), 400
        
        print(f"Extracted {len(text)} characters of text")
        
        # Create chunks
        chunks = chunk_text(text)
        if not chunks:
            return jsonify({"error": "Failed to create text chunks"}), 400
        
        # Get relevant chunks
        relevant_chunks = get_relevant_chunks(query, chunks)
        if not relevant_chunks:
            return jsonify({"error": "No relevant content found"}), 400
        
        print(f"Found {len(relevant_chunks)} relevant chunks")
        
        # Generate summary
        summary = generate_summary(query, relevant_chunks)
        
        # Extract metadata
        metadata = extract_metadata_from_text(text)
        
        response = {
            "summary": summary,
            "metadata": metadata,
            "chunks_analyzed": len(chunks),
            "relevant_chunks": len(relevant_chunks)
        }
        
        print("Analysis completed successfully")
        return jsonify(response)
        
    except Exception as e:
        print(f"Analysis failed: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)