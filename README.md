Academic Research Assistant 🎓🔍
A Flask-based web application that helps researchers find, analyze, and summarize academic documents using AI-powered search and natural language processing.

Key Features ✨
Smart Document Search: Finds relevant PDFs, PPTs, and Word documents using SerpAPI

Metadata Extraction: Automatically detects document titles, authors, and publication years

Semantic Analysis: Uses sentence transformers to find the most relevant content

AI Summarization: Generates comprehensive summaries using Llama3

Document Classification: Organizes results by document type (PDFs, PPTs, etc.)

Technology Stack 🛠️
Backend: Python Flask

Search API: SerpAPI (Google Search)

NLP Models:

Sentence Transformers (all-MiniLM-L6-v2) for semantic search

Llama3 (via Ollama) for summarization and metadata extraction

Document Processing: PyPDF2 for text extraction

Frontend: HTML/CSS/JavaScript (not included in this backend code)

How It Works ⚙️
Search Phase:

User enters research query and selects document types

System searches academic sources using SerpAPI

Extracts metadata from URLs and document headers

Analysis Phase:

Downloads selected documents

Extracts and chunks text content

Finds most relevant sections using cosine similarity

Generates comprehensive summaries with Llama3

Results:

Presents organized results with:

Document metadata (title, author, source)

AI-generated summary

Relevance scores

Setup Instructions 🚀
Prerequisites
Python 3.8+

Ollama with Llama3 installed (ollama pull llama3)

SerpAPI account (for search functionality)

Installation
bash
Copy
git clone https://github.com/yourusername/academic-research-assistant.git
cd academic-research-assistant
pip install -r requirements.txt
Configuration
Create a .env file with your API keys:

env
Copy
SERPAPI_API_KEY=your_serpapi_key
Running the Application
bash
Copy
# Start Ollama in another terminal
ollama serve

# Run the Flask application
python app.py
API Endpoints 🌐
POST /search - Search for academic documents

Parameters: query, types (array of document types)

POST /analyze - Analyze a specific document

Parameters: url, query

Example Usage 📖
python
Copy
# Search for machine learning papers
response = requests.post("http://localhost:5000/search", json={
    "query": "machine learning in healthcare",
    "types": ["PDF", "Research"]
})

# Analyze a specific document
response = requests.post("http://localhost:5000/analyze", json={
    "url": "https://arxiv.org/pdf/2103.12345.pdf",
    "query": "machine learning applications in radiology"
})
Future Enhancements 🔮
Add support for more document types (ePub, HTML)

Implement citation extraction

Add reference management integration

Develop browser extension version

Contributing 🤝
Contributions are welcome! Please open an issue or submit a pull request.

This description highlights the key aspects of your project while making it appealing to potential users and contributors. You can adjust the specific details based on your exact implementation and goals for the project.
